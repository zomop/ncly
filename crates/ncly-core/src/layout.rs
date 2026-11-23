//! A strided view description: shape, strides and storage offset.
//!
//! [`Layout`] is the single source of truth for "where does logical element
//! `(i, j, k)` live". A contiguous tensor has `offset == 0` and canonical
//! strides; a transposed view swaps strides; a broadcast operand uses `0`
//! strides. Because every kernel walks a `Layout`, one implementation handles
//! all three without copying.

use crate::error::{AxisError, NclyError, Result};
use crate::shape::{Shape, Strides};

/// Shape, strides and storage offset of a strided view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Strides,
    offset: usize,
}

impl Layout {
    /// Build a layout from its parts.
    ///
    /// The caller is responsible for keeping `offset` inside the storage that
    /// the layout describes; [`Layout::span`] reports the highest address the
    /// layout can touch so that slicing code can check it.
    pub fn new(shape: Shape, strides: Strides, offset: usize) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }

    /// Contiguous row-major layout of `shape`.
    pub fn contiguous(shape: Shape) -> Self {
        let strides = shape.canonical_strides();
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// The shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The strides.
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// The storage offset of the first logical element.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// True when the layout is contiguous and has no offset.
    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.shape.is_canonical(&self.strides)
    }

    /// Smallest flat index that can be addressed by this layout.
    pub fn min_offset(&self) -> usize {
        let mut min = self.offset as isize;
        for (axis, &len) in self.shape.dims().iter().enumerate() {
            if len == 0 {
                return self.offset;
            }
            let stride = self.strides.get(axis).unwrap_or(0);
            if stride < 0 {
                min += stride * (len as isize - 1);
            }
        }
        min as usize
    }

    /// Largest flat index (exclusive upper bound) addressable by this layout.
    pub fn span(&self) -> usize {
        let mut max = self.offset as isize;
        for (axis, &len) in self.shape.dims().iter().enumerate() {
            if len == 0 {
                return self.offset;
            }
            let stride = self.strides.get(axis).unwrap_or(0);
            max += stride * (len as isize - 1);
        }
        max as usize + 1
    }

    /// Flat storage index of a logical index.
    pub fn flat(&self, index: &[usize]) -> Result<usize> {
        if index.len() != self.shape.rank() {
            return Err(NclyError::invalid_shape(
                "index",
                format!(
                    "index of rank {} used on a tensor of rank {}",
                    index.len(),
                    self.shape.rank()
                ),
            ));
        }
        let mut flat = self.offset as isize;
        for (axis, &i) in index.iter().enumerate() {
            let len = self.shape.dims()[axis];
            if i >= len {
                return Err(NclyError::IndexOutOfBounds {
                    index: index.to_vec(),
                    shape: self.shape.to_string(),
                });
            }
            flat += i as isize * self.strides.get(axis).unwrap_or(0);
        }
        Ok(flat as usize)
    }

    /// Transpose by reversing all axes.
    pub fn transpose(&self) -> Self {
        let mut dims = self.shape.dims().to_vec();
        dims.reverse();
        let mut strides = self.strides.as_slice().to_vec();
        strides.reverse();
        Self {
            shape: Shape::new(dims),
            strides: Strides::new(strides),
            offset: self.offset,
        }
    }

    /// Swap two axes.
    pub fn swap_axes(&self, a: usize, b: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if a >= rank || b >= rank {
            return Err(AxisError::AxisOutOfBounds {
                axis: a.max(b),
                rank,
            }
            .into());
        }
        let mut dims = self.shape.dims().to_vec();
        dims.swap(a, b);
        let mut strides = self.strides.as_slice().to_vec();
        strides.swap(a, b);
        Ok(Self {
            shape: Shape::new(dims),
            strides: Strides::new(strides),
            offset: self.offset,
        })
    }

    /// Broadcast this layout to `target`, using zero strides on new axes.
    pub fn broadcast_to(&self, target: &Shape) -> Result<Self> {
        let plan = crate::shape::Broadcast::plan_with_strides(&self.shape, &self.strides, target)?;
        Ok(Self {
            shape: target.clone(),
            strides: Strides::new(plan.strides),
            offset: self.offset,
        })
    }

    /// Reshape a *contiguous* layout. Strided layouts must be materialised first.
    pub fn reshape(&self, target: &[usize]) -> Result<Self> {
        if !self.is_contiguous() {
            return Err(NclyError::invalid_shape(
                "reshape",
                "layout is not contiguous",
            ));
        }
        let shape = self.shape.reshape_to(target)?;
        Ok(Self::contiguous(shape))
    }

    /// Narrow one axis to `[start, start + len)`.
    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> Result<Self> {
        let dim = self.shape.dim(axis)?;
        if start + len > dim {
            return Err(NclyError::invalid_shape(
                "narrow",
                format!("range {start}..{} exceeds axis length {dim}", start + len),
            ));
        }
        let mut dims = self.shape.dims().to_vec();
        dims[axis] = len;
        let offset = self.offset as isize + start as isize * self.strides.get(axis).unwrap_or(0);
        Ok(Self {
            shape: Shape::new(dims),
            strides: self.strides.clone(),
            offset: offset as usize,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_span_matches_size() {
        let layout = Layout::contiguous(Shape::new([2, 3]));
        assert!(layout.is_contiguous());
        assert_eq!(layout.span(), 6);
        assert_eq!(layout.flat(&[1, 2]).unwrap(), 5);
    }

    #[test]
    fn transpose_swaps_strides() {
        let layout = Layout::contiguous(Shape::new([2, 3])).transpose();
        assert_eq!(layout.shape(), &Shape::new([3, 2]));
        assert_eq!(layout.strides(), &Strides::new([1isize, 3]));
        assert!(!layout.is_contiguous());
        assert_eq!(layout.flat(&[2, 1]).unwrap(), 5);
    }

    #[test]
    fn broadcast_layout_uses_zero_strides() {
        let layout = Layout::contiguous(Shape::new([3, 1]))
            .broadcast_to(&Shape::new([2, 3, 4]))
            .unwrap();
        // `(3, 1)` has canonical strides `(1, 1)`; the surviving axis keeps
        // stride 1 and the two expanded axes get stride 0.
        assert_eq!(layout.strides(), &Strides::new([0isize, 1, 0]));
        assert_eq!(layout.span(), 3);
        assert_eq!(layout.flat(&[1, 2, 3]).unwrap(), 2);
    }

    #[test]
    fn broadcast_keeps_the_strides_of_a_transposed_view() {
        // (2, 3) transposed to (3, 2) has strides (1, 3); broadcasting must
        // preserve that, not fall back to canonical strides.
        let view = Layout::contiguous(Shape::new([2, 3])).transpose();
        let layout = view.broadcast_to(&Shape::new([5, 3, 2])).unwrap();
        assert_eq!(layout.strides(), &Strides::new([0isize, 1, 3]));
        assert_eq!(layout.flat(&[4, 2, 1]).unwrap(), 2 + 3);
    }

    #[test]
    fn narrow_moves_offset() {
        let layout = Layout::contiguous(Shape::new([4, 5]))
            .narrow(0, 1, 2)
            .unwrap();
        assert_eq!(layout.shape(), &Shape::new([2, 5]));
        assert_eq!(layout.offset(), 5);
    }

    #[test]
    fn out_of_bounds_reports_index() {
        let layout = Layout::contiguous(Shape::new([2, 2]));
        let err = layout.flat(&[0, 5]).unwrap_err();
        assert!(matches!(err, NclyError::IndexOutOfBounds { .. }));
    }
}
