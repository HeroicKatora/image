use core::num::NonZeroUsize;

use crate::{ImageError, ImageResult};
use crate::error::{ImageFormatHint, UnsupportedError, UnsupportedErrorKind};
use crate::image::{GenericImageView as _, GenericImage as _};
use crate::color::{ColorType, ExtendedColorType};
use crate::dynimage::DynamicImage;
use crate::flat::{View, SampleLayout};
use crate::traits::{EncodableLayout, Pixel};

use bytemuck::{cast_slice, cast_slice_mut};

/// Implementation detail, maximum alignment of all supported primitives.
type MaxAlign = u64;


/// A byte buffer that can hold a byte representation of known color/texel types.
///
/// This is not a pixel matrix. The main purpose is that of compatibility: convert to and from the
/// layouts in `DynamicImage`, which we can compute with. Also, interface with all decoders,
/// (foreign) functions, and as a compatibility buffer. As such, it offers a view on the raw bytes
/// of the internal texel grid.
///
/// # Usage
///
/// Note that there are multiple constructors. Firstly, conversion via the `From` trait for simple
/// cases. These constructors copy pixel data and will preserve the layout of the original data.
/// Secondly, explicit methods that copy pixel data but might deviate from the layout guarantees as
/// documented on them. Assigning to an existing `ImageBytes` with its [`paint`] method will also
/// preserve its current layout, converting the assigned data into it.
///
/// ```rust
/// use image::buffer::ImageBytes;
/// ```
///
/// # Design
///
/// Internally, it holds matrices of 'texels' (texture units) which may express multiple pixels at
/// once (grouped together or sub-sampled). Additionally, the texel channels maybe be spread out
/// through multiple image planes instead of being grouped together.
///
/// For any planar layout there is an equivalent non-planar layout to which we can normalize.
#[derive(Clone)]
pub struct ImageBytes {
    /// The bytes (or integer contents) of the buffer.
    /// The length is at least as large as the layout requires.
    buffer: Vec<MaxAlign>,
    /// The byte layout of the buffer.
    layout: Layout,
    /// The underlying, extended 'color type'.
    color: ExtendedColorType,
}

/// The layout of a simple pixel matrix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ColorLayout {
    /// The simple color type of the pixel.
    pub color: ColorType,
    /// The width, number of pixels horizontally.
    pub width: u32,
    /// The height, number of pixels vertically.
    pub height: u32,
}

/// A private, inner type for layouts.
/// We do not want to expose the struct- or enum-ness of this type. If we are to expose it to the
/// outside (e.g. for the sake of layout comparison and manipulation) it should be wrapped and
/// renamed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Layout {
    /// The total byte usage of this layout.
    len: usize,
    /// The first plane of the image.
    /// For texel based images this is the only plane containing texel chunks in a matrix.
    main_plane: SamplePlane,
    /// The individual image plane parameters if they exist.
    planes: [Option<SamplePlane>; 3],
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Number of channels of the color type.
    /// This equals the number of planes in a planar layout.
    channels: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct SamplePlane {
    channel_stride: NonZeroUsize,
    width_stride: NonZeroUsize,
    height_stride: NonZeroUsize,
    offset: usize,
}

impl ImageBytes {
    /// Create an empty, row-major matrix of pixel elements.
    ///
    /// # Panics
    ///
    /// Panic when the layout would overflow the address space, or the allocation fails. You may
    /// use [`ColorLayout::byte_len()`] to perform size checks before calling this method.
    pub fn with_row_matrix(layout: ColorLayout) -> Self {
        let color = layout.color.into();
        let layout = Layout::with_row_matrix(layout)
            .unwrap_or_else(|| address_space_error());
        Self::with_bytes(layout, &[], color)
    }

    /// Create a buffer, copying a view.
    ///
    /// Note: the channel type is constrained by a private, sealed trait. This is by design, to
    /// avoid exposing the underlying methods for casting to/from bytes. It's implemented for all
    /// common integer types, in particular the ones found underlying `DynamicImage` variants.
    ///
    /// The exact layout is preserved if the pixels of the view do not alias. Otherwise, it is
    /// converted to a row-major layout with implementation defined row stride.
    ///
    /// This method requires passing a [`View`] of a pixel matrix with standard layout.
    ///
    /// # Panics
    ///
    /// This method panics if the allocation fails, or if the non-aliasing buffer would exceed the
    /// available address space.
    pub fn with_view<P, C>(img: View<&'_ [C], P>) -> ImageBytes
    where
        P: Pixel<Subpixel=C>,
        [C]: EncodableLayout,
    {
        let color = <P as Pixel>::COLOR_TYPE;

        if !img.flat().has_aliased_samples() {
            // Great, we have a standard pixel matrix.
            let layout = Layout::with_strides(img.flat().layout);
            let bytes = img.image_slice().as_bytes();
            Self::with_bytes(layout, bytes, color.into())
        } else {
            // Normalize our pixel layout.
            let layout = Layout::with_row_matrix(ColorLayout {
                color,
                width: img.width(),
                height: img.height(),
            });

            let layout = layout.unwrap_or_else(|| address_space_error());
            let mut buffer = Self::with_bytes(layout, &[], color.into());
            let layout = buffer.layout.layout_of(&buffer.layout.main_plane);

            for (x, y, c) in img.pixels() {
                let start = layout.in_bounds_index(0, x, y);
                let data = c.channels().as_bytes();
                buffer.as_bytes_mut()[start..][..data.len()]
                    .copy_from_slice(data);
            }

            buffer
        }
    }

    /// Get a reference to all raw bytes of the image.
    ///
    /// Note: The bytes are aligned stricter than alignment `1`, however the exact details are not
    /// considered to be stable.
    pub fn as_bytes(&self) -> &[u8] {
        &cast_slice(&self.buffer)[..self.layout.len]
    }

    /// Get a mutable reference to all raw bytes of the image.
    ///
    /// Note: The bytes are aligned stricter than alignment `1`, however the exact details are not
    /// considered to be stable.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut cast_slice_mut(&mut self.buffer)[..self.layout.len]
    }

    /// Get the number of pixels horizontally.
    pub fn width(&self) -> u32 {
        self.layout.width
    }

    /// Get the number of pixels vertically.
    pub fn height(&self) -> u32 {
        self.layout.height
    }

    fn equivalent_color_type(&self) -> ColorType {
        use ExtendedColorType::*;
        match self.color {
            L1 | L2 | L4 | L8 => ColorType::L8,
            La1 | La2 | La4 | La8 => ColorType::La8,
            _ => todo!(),
        }
    }

    /// Overwrite pixel contents with the right-hand-side.
    pub fn paint(&mut self, _: &Self) -> ImageResult<()> {
        Err(ImageError::Unsupported(UnsupportedError::from_format_and_kind(
            ImageFormatHint::Unknown,
            UnsupportedErrorKind::GenericFeature("no paint operations are implemented yet.".into())
        )))
    }

    /// An internal method to pre-initialize a buffer with a specific layout.
    /// Not exposed due to keeping `Layout` an implementation detail and an unclear design.
    /// Should bytes be at least or at most as long as layout?
    fn with_bytes(layout: Layout, bytes: &[u8], color: ExtendedColorType) -> Self {
        let prelen = layout.len.min(bytes.len());
        let mut buffer = vec![0 as MaxAlign; byte_len_as_max_align(layout.len)];
        cast_slice_mut(&mut buffer)[..prelen].copy_from_slice(&bytes[..prelen]);
        ImageBytes {
            buffer,
            layout,
            color,
        }
    }
}

impl From<&'_ DynamicImage> for ImageBytes {
    fn from(img: &'_ DynamicImage) -> ImageBytes {
        let (width, height) = img.dimensions();
        let color = img.color().into();
        let layout = Layout::with_row_matrix(ColorLayout { color, width, height })
            // Due to the input type we already a matrix, proving it fits into memory.
            .expect("Layout of a `DynamicImage` is valid");
        ImageBytes::with_bytes(layout, img.as_bytes(), color.into())
    }
}

impl ColorLayout {
    fn byte_len(&self) -> Option<usize> {
        // Ensure this all fits in `usize`.
        fn matrix_size(el: usize, width: u32, height: u32) -> Option<usize> {
            use core::convert::TryFrom;
            let width = usize::try_from(width).ok()?;
            let height = usize::try_from(height).ok()?;
            width.checked_mul(height)?.checked_mul(el)
        }

        let bytes = usize::from(self.color.bytes_per_pixel());
        matrix_size(bytes, self.width, self.height)
    }
}

impl Layout {
    /// A row-major matrix of pixel texels.
    fn with_row_matrix(layout: ColorLayout) -> Option<Self> {
        let len = layout.byte_len()?;
        let channels = layout.color.channel_count();
        let layout = SampleLayout::row_major_packed(channels, layout.width, layout.height);
        Some(Self::with_strides(layout))
    }

    /// From a pre-determined matrix layout, single plane.
    /// Must not have any aliased pixels and be valid for memory.
    fn with_strides(layout: SampleLayout) -> Self {
        let len = layout.min_length()
            .expect("called on non-aliased, in-memory layout");
        Layout {
            len,
            main_plane: Self::plane_of(&layout),
            planes: [None; 3],
            channels: layout.channels,
            width: layout.width,
            height: layout.height,
        }
    }

    fn plane_of(layout: &SampleLayout) -> SamplePlane {
        let channel_stride = NonZeroUsize::new(layout.channel_stride)
            .expect("zero width stride is forbidden aliased layout");
        let width_stride = NonZeroUsize::new(layout.width_stride)
            .expect("zero width stride is forbidden aliased layout");
        let height_stride = NonZeroUsize::new(layout.height_stride)
            .expect("zero height stride is forbidden aliased layout");
        SamplePlane {
            channel_stride,
            width_stride,
            height_stride,
            offset: 0,
        }
    }

    /// Turn a plane into a full layout.
    fn layout_of(&self, plane: &SamplePlane) -> SampleLayout {
        SampleLayout {
            channels: self.channels,
            width: self.width,
            height: self.height,
            channel_stride: plane.channel_stride.get(),
            width_stride: plane.width_stride.get(),
            height_stride: plane.height_stride.get(),
        }
    }
}

fn byte_len_as_max_align(bytes: usize) -> usize {
    const SIZE: usize = core::mem::size_of::<MaxAlign>();
    bytes / SIZE + usize::from(bytes % SIZE != 0)
}

#[cold]
fn address_space_error() -> ! {
    panic!("too large for address space.")
}

#[cfg(test)]
mod tests {
    use crate::flat::FlatSamples; 
    use crate::{DynamicImage, GenericImageView, Rgba, RgbaImage};

    use super::ImageBytes;

    #[test]
    fn dyn_layout() {
        let dynimg = DynamicImage::ImageRgba8(RgbaImage::from_fn(13, 17, |x, y| {
            let (x, y) = (x as u8, y as u8);
            Rgba::<u8>([x, y, x.wrapping_add(y), x.wrapping_sub(y),])
        }));

        let (width, height) = dynimg.dimensions();
        let bytes = ImageBytes::from(&dynimg);

        assert_eq!(bytes.width(), width);
        assert_eq!(bytes.height(), height);
        assert_eq!(dynimg.as_bytes(), bytes.as_bytes());
    }

    #[test]
    fn flat_layout() {
        let dynimg = DynamicImage::ImageRgba8(RgbaImage::from_fn(13, 17, |x, y| {
            let (x, y) = (x as u8, y as u8);
            Rgba::<u8>([x, y, x.wrapping_add(y), x.wrapping_sub(y),])
        }));

        let samples = dynimg.as_flat_samples_u8()
            .expect("u8 image");
        let view = samples
            .as_view::<Rgba<u8>>()
            .expect("valid view");

        let (width, height) = dynimg.dimensions();
        let bytes = ImageBytes::with_view(view);

        assert_eq!(bytes.width(), width);
        assert_eq!(bytes.height(), height);
        assert_eq!(dynimg.as_bytes(), bytes.as_bytes());
    }


    #[test]
    fn flat_aliased() {
        let color = Rgba([0, 1, 2, 3]);

        let samples = FlatSamples::with_monocolor(&color, 13, 17);
        let view = samples
            .as_view::<Rgba<u8>>()
            .expect("valid view");

        let (width, height) = view.dimensions();
        let bytes = ImageBytes::with_view(view);

        assert_eq!(bytes.width(), width);
        assert_eq!(bytes.height(), height);
        assert!(samples.as_slice() != bytes.as_bytes());
    }
}
