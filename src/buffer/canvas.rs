//! A byte-buffer based image descriptor.
use crate::{ImageDecoder, ImageDecoderRect, ImageResult};
use crate::color::{ColorType, ExtendedColorType as TexelKind};
use crate::flat::FlatSamples;
use crate::traits::{Pixel, PixelWithColorType};

use canvas::canvas::{CanvasRef, CanvasMut};
use canvas::layout::{Layout as CanvasLayout, Matrix, MatrixBytes, MatrixLayout, Raster};

pub struct Canvas {
    inner: canvas::Canvas<Layout>,
}

/// Represents a single matrix like layer of an image.
pub struct LayerRef<'data, P: Pixel> {
    inner: CanvasRef<'data, Matrix<P::Subpixel>>,
}

/// Represents a single mutable matrix like layer of an image.
pub struct LayerMut<'data, P: Pixel> {
    inner: CanvasMut<'data, Matrix<P::Subpixel>>,
}

pub struct SetFromDecoder<'canvas, 'decoder, I> {
    canvas: &'canvas mut Canvas,
    decoder: I,
    decoder_data: core::marker::PhantomData<&'decoder mut()>,
}

/// The layout of an image except for metadata.
#[derive(Clone)]
pub struct Layout {
    /// The texel in non-planar fashion.
    texel: TexelKind,
    color: Option<ColorType>,
    texel_width: u32,
    texel_height: u32,
    pixel_with: u32,
    pixel_height: u32,
    bytes_per_pixel: usize,
    layers: Box<[usize]>,
    // TODO: color space information.
}

impl CanvasLayout for Layout {
    fn byte_len(&self) -> usize {
        todo!()
    }
}

impl MatrixLayout for Layout {
    fn matrix(&self) -> MatrixBytes {
        todo!()
    }
}

impl Layout {
    pub const fn empty(texel: TexelKind) -> Self {
        todo!()
    }

    pub fn with_decoder<'a>(decoder: &impl ImageDecoder<'a>) -> Self {
        let (width, height) = decoder.dimensions();
        let texel = decoder.original_color_type();
        todo!()
    }

    /// Returns the width of the underlying image in pixels.
    pub fn width(&self) -> u32 {
        todo!()
    }

    /// Returns the height of the underlying image in pixels.
    pub fn height(&self) -> u32 {
        todo!()
    }

    /// Retrieve the color information, if the canvas is a simple pixel colored matrix.
    pub fn color_type(&self) -> Option<ColorType> {
        self.color
    }
}

impl Canvas {
    /// Create an empty image that will use the indicated texels.
    ///
    /// This will _not_ allocate.
    pub fn empty(texel: TexelKind) -> Self {
        Canvas {
            inner: canvas::Canvas::new(Layout::empty(texel))
        }
    }

    /// Allocate and Initialize a new canvas by decoding.
    pub fn with_decoder<'a>(decoder: impl ImageDecoder<'a>) -> ImageResult<Self> {
        let mut canvas = Canvas::empty(TexelKind::A8);
        canvas.set_from_decoder(decoder)
            .read_image()?;
        Ok(canvas)
    }

    /// Overwrite the layout, allocate if necessary, and clear the image.
    pub fn set_layout(&mut self, layout: Layout) {
        self.set_layout_conservative(layout);
        self.inner.as_bytes_mut().fill(0);
    }

    /// Overwrite the layout, allocate if necessary, _do not_ clear the image.
    pub fn set_layout_conservative(&mut self, layout: Layout) {
        *self.inner.layout_mut_unguarded() = layout;
        self.inner.ensure_layout();
    }

    /// Start overwriting the byte data.
    ///
    /// Returns an object to perform the actual IO. The byte buffer is re-allocated according to
    /// the decoders requirements immediately.
    pub fn set_from_decoder<'a, I: ImageDecoder<'a>>(&mut self, decoder: I) -> SetFromDecoder<'_, 'a, I> {
        let layout = Layout::with_decoder(&decoder);
        self.set_layout_conservative(layout);
        SetFromDecoder {
            canvas: self,
            decoder,
            decoder_data: core::marker::PhantomData,
        }
    }

    /// Return this image's pixels as a native endian byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// Return this image's pixels as a mutable native endian byte slice.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    pub fn as_flat_samples_u8(&self) -> Option<FlatSamples<&[u8]>> {
        None
    }

    pub fn as_flat_samples_u16(&self) -> Option<FlatSamples<&[u16]>> {
        None
    }

    pub fn as_flat_samples_f32(&self) -> Option<FlatSamples<&[u16]>> {
        None
    }

    /// Return this image's pixels as a byte vector.
    pub fn into_bytes(self) -> Vec<u8> {
        // We can not reuse the allocation of `canvas`.
        self.as_bytes().to_owned()
    }

    /// Convert into a dynamic image.
    ///
    /// Potentially performs texel conversions, color conversions, and a separate allocation.
    pub fn into_image(self) -> DynamicImage {
        // Determine to what and how to convert. Works directly on the (untyped) internal buffer.
        //
        // For the CPU SIMD-capable case we perform this in three stages:
        // - Remove planarity and merge to texel clusters.
        // - Expand subsampled texel clusters into pixels.
        // - Color convert to the target colors.
        let into_color_type = match todo!() {
        };
    }

    /// Get a reference to the data as a single, strongly typed matrix.
    ///
    /// This returns `None` if the canvas' color is not compatible with the provided pixel type,
    /// i.e. the `COLOR_TYPE` constant of the type does not equal the inner `color`.
    pub fn as_ref<P: PixelWithColorType>(&self) -> Option<LayerRef<P>> {
        if self.color_type() == Some(P::COLOR_TYPE) {
            todo!()
        } else {
            None
        }
    }

    /// Get a mutable reference to the data as a single, strongly typed matrix.
    pub fn as_mut<P: PixelWithColorType>(&self) -> Option<LayerMut<P>> {
        if self.color_type() == Some(P::COLOR_TYPE) {
            todo!()
        } else {
            None
        }
    }

    /// Retrieve the color information, if the canvas is a simple pixel colored matrix.
    pub fn color_type(&self) -> Option<ColorType> {
        self.inner.layout().color_type()
    }

    /// TODO: In this form?
    pub fn texel_type(&self) -> TexelKind {
        self.inner.layout().texel
    }
}

impl<'canvas, 'decoder, I> SetFromDecoder<'canvas, 'decoder, I> {
    /// Read the full image data in a single step, if possible.
    pub fn read_image(self) -> ImageResult<()>
    where
        I: ImageDecoder<'decoder>,
    {
        self.decoder.read_image(self.canvas.as_bytes_mut())
    }
}
