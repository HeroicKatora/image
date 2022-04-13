//! A byte-buffer based image descriptor.
use crate::color::{ColorType, ExtendedColorType as TexelKind};
use crate::traits::{Pixel, PixelWithColorType};

use canvas::canvas::{CanvasRef, CanvasMut};
use canvas::layout::{Layout as CanvasLayout, Matrix, MatrixBytes, MatrixLayout, Raster};

pub struct Canvas {
    inner: canvas::Canvas<Layout>,
}

pub struct LayerRef<'data, P: Pixel> {
    inner: CanvasRef<'data, Matrix<P::Subpixel>>,
}

pub struct LayerMut<'data, P: Pixel> {
    inner: CanvasMut<'data, Matrix<P::Subpixel>>,
}

/// The layout of an image except for metadata.
#[derive(Clone)]
pub struct Layout {
    /// The texel in non-planar fashion.
    texel: TexelKind,
    texel_width: u32,
    texel_height: u32,
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

impl Canvas {
    /// Retrieve the color information, if the canvas is a simple pixel colored matrix.
    pub fn color_type(&self) -> Option<ColorType> {
        todo!()
    }

    /// FIXME: really?
    pub fn texel_type(&self) -> TexelKind {
        todo!()
    }

    /// Get a reference to the data as a single, strongly typed matrix.
    ///
    /// This returns `None` if the canvas' color is not compatible with the provided pixel type,
    /// i.e. the `COLOR_TYPE` constant of the type does not equal the inner `color`.
    pub fn as_ref<P: PixelWithColorType>(&self) -> Option<LayerRef<P>> {
        todo!()
    }

    /// Get a mutable reference to the data as a single, strongly typed matrix.
    pub fn as_mut<P: PixelWithColorType>(&self) -> Option<LayerRef<P>> {
        todo!()
    }
}
