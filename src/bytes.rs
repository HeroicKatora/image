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

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Layout {
    len: usize,
    pixel_width: u32,
    pixel_height: u32,
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

            for (x, y, c) in img.pixels() {
                todo!()
            }

            buffer
        }
    }

    /// Get a reference to all raw bytes of the image.
    ///
    /// Note: The bytes are aligned stricter than alignment `1`, however the exact details are not
    /// considered to be stable.
    pub fn as_bytes(&self) -> &[u8] {
        cast_slice(&self.buffer)
    }

    /// Get a mutable reference to all raw bytes of the image.
    ///
    /// Note: The bytes are aligned stricter than alignment `1`, however the exact details are not
    /// considered to be stable.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        cast_slice_mut(&mut self.buffer)
    }

    /// Get the number of pixels horizontally.
    pub fn width(&self) -> u32 {
        self.layout.pixel_width
    }

    /// Get the number of pixels vertically.
    pub fn height(&self) -> u32 {
        self.layout.pixel_height
    }

    /// Get the number of bytes needed per pixel, if the pixel were represented individually.
    ///
    /// For the sake of this calculation, each pixel will be padded to a whole byte.
    pub fn bytes_per_pixel(&self) -> u8  {
        todo!("Calculate a 'normalized' color.")
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
        Some(Layout {
            len,
            pixel_width: layout.width,
            pixel_height: layout.height,
        })
    }

    fn with_strides(layout: SampleLayout) -> Self {
        // Yes, this isn't in-bounds. It's fine. See [SampleLayout::min_length] for details.
        // This is more efficient since we assume it fits in memory.
        let len = layout.in_bounds_index(0, layout.width, layout.height);
        Layout {
            len,
            pixel_width: layout.width,
            pixel_height: layout.height,
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
