//! Shape and stride algebra.
//!
//! A [`Shape`] is a vector of axis lengths. NCLY uses *row-major* (C order) as
//! its default memory layout, so the canonical strides of a shape are computed
//! right to left:
//!
//! ```text
//! shape   = (2, 3, 4)
//! strides = (12, 4, 1)
//! ```
//!
//! A zero length axis contributes a `0` stride, which keeps the layout
//! *canonical* even for empty tensors: no two logical positions alias the same
//! element unless the tensor is empty, which [`Shape::is_canonical`] relies on.

use crate::error::{AxisError, NclyError, Result};
use core::fmt;

/// The maximum number of dimensions supported by the tensor engine.
pub const MAX_RANK: usize = 8;

/// Axis lengths of an n-dimensional value.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a shape from axis lengths.
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// A shape with no axes: exactly one element.
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    /// A shape whose axes all have the same length.
    pub fn from_elem(len: usize, rank: usize) -> Self {
        Self {
            dims: vec![len; rank],
        }
    }

    /// The axis lengths.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Mutable access to the axis lengths; used by constructors and reshapes.
    pub fn dims_mut(&mut self) -> &mut Vec<usize> {
        &mut self.dims
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// True when the shape has no axes.
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// True when at least one axis has length zero.
    pub fn is_empty(&self) -> bool {
        self.size() == 0 && !self.is_scalar()
    }

    /// Length of `axis`.
    pub fn dim(&self, axis: usize) -> Result<usize> {
        self.dims.get(axis).copied().ok_or(NclyError::Axis(
            AxisError::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            },
        ))
    }

    /// Canonical row-major strides for this shape.
    pub fn canonical_strides(&self) -> Strides {
        Strides::canonical(&self.dims)
    }

    /// Whether `strides` are canonical for this shape.
    pub fn is_canonical(&self, strides: &Strides) -> bool {
        self.canonical_strides() == *strides
    }

    /// Broadcast two shapes following NumPy's rules.
    ///
    /// Axes are aligned from the right. Two lengths are compatible when they are
    /// equal or when one of them is `1`, in which case the result takes the
    /// larger length.
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        broadcast_shapes([self, other])
    }

    /// Number of elements contributed by the axes strictly after `axis`.
    pub fn inner_size(&self, axis: usize) -> usize {
        self.dims[axis + 1..].iter().product()
    }

    /// Number of elements contributed by the axes strictly before `axis`.
    pub fn outer_size(&self, axis: usize) -> usize {
        self.dims[..axis].iter().product()
    }

    /// Shape with the given axis removed.
    pub fn without_axis(&self, axis: usize) -> Result<Shape> {
        if axis >= self.rank() {
            return Err(AxisError::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            }
            .into());
        }
        let mut dims = self.dims.clone();
        dims.remove(axis);
        Ok(Self { dims })
    }

    /// Shape with a new axis of length `len` inserted at `axis`.
    pub fn with_axis(&self, axis: usize, len: usize) -> Result<Shape> {
        if axis > self.rank() {
            return Err(AxisError::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            }
            .into());
        }
        let mut dims = self.dims.clone();
        dims.insert(axis, len);
        Ok(Self { dims })
    }

    /// Shape with every axis of length `1` removed.
    pub fn squeezed(&self) -> Shape {
        Self {
            dims: self.dims.iter().copied().filter(|&d| d != 1).collect(),
        }
    }

    /// Shape after applying `squeeze` to one specific axis.
    pub fn squeeze_axis(&self, axis: usize) -> Result<Shape> {
        if axis >= self.rank() {
            return Err(AxisError::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            }
            .into());
        }
        if self.dims[axis] != 1 {
            return Err(NclyError::invalid_shape(
                "squeeze",
                format!("axis {axis} has length {}", self.dims[axis]),
            ));
        }
        Ok(self.without_axis(axis)?)
    }

    /// Validate a reshape between the same element count.
    pub fn reshape_to(&self, target: impl Into<Vec<usize>>) -> Result<Shape> {
        let mut target = target.into();
        // A single inferred axis (`usize::MAX` sentinel) is resolved here.
        if let Some(position) = target.iter().position(|&d| d == usize::MAX) {
            let known: usize = target
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != position)
                .map(|(_, &d)| d)
                .product();
            if known == 0 {
                return Err(NclyError::invalid_shape("reshape", "cannot infer an axis"));
            }
            target[position] = self.size() / known;
        }
        let target = Shape::new(target);
        if target.rank() > MAX_RANK {
            return Err(AxisError::RankOverflow { max_rank: MAX_RANK }.into());
        }
        if target.size() != self.size() {
            return Err(NclyError::invalid_shape(
                "reshape",
                format!(
                    "cannot reshape {} elements into shape {}",
                    self.size(),
                    target
                ),
            ));
        }
        Ok(target)
    }

    /// Iterate over every index of the shape in row-major order.
    pub fn indices(&self) -> Indices<'_> {
        Indices {
            shape: self,
            counter: vec![0; self.rank()],
            remaining: self.size(),
        }
    }

    /// Flat row-major offset of `index`, or `None` when out of bounds.
    pub fn offset_of(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.rank() {
            return None;
        }
        let mut offset = 0usize;
        for (axis, (&i, &len)) in index.iter().zip(self.dims.iter()).enumerate() {
            if i >= len {
                return None;
            }
            offset += i * self.inner_size(axis);
        }
        Some(offset)
    }

    /// Decompose a flat row-major offset into an index vector.
    pub fn index_of(&self, mut offset: usize) -> Option<Vec<usize>> {
        if offset >= self.size() {
            return None;
        }
        let mut index = vec![0usize; self.rank()];
        for axis in 0..self.rank() {
            let stride = self.inner_size(axis);
            index[axis] = offset / stride;
            offset %= stride;
        }
        Some(index)
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.dims)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dim}")?;
        }
        f.write_str(")")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self { dims }
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.dims
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.dims
    }
}

impl IntoIterator for Shape {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.into_iter()
    }
}

impl<'a> IntoIterator for &'a Shape {
    type Item = &'a usize;
    type IntoIter = core::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.iter()
    }
}

/// Row-major (or arbitrary) strides.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Strides {
    strides: Vec<isize>,
}

impl Strides {
    /// Create strides from explicit values.
    pub fn new(strides: impl Into<Vec<isize>>) -> Self {
        Self {
            strides: strides.into(),
        }
    }

    /// Canonical row-major strides for `dims`.
    pub fn canonical(dims: &[usize]) -> Self {
        dyn_canonical_strides(dims)
    }

    /// The stride values.
    pub fn as_slice(&self) -> &[isize] {
        &self.strides
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.strides.len()
    }

    /// Stride of `axis`.
    pub fn get(&self, axis: usize) -> Option<isize> {
        self.strides.get(axis).copied()
    }

    /// True when every stride is non-negative.
    pub fn is_non_negative(&self) -> bool {
        self.strides.iter().all(|&s| s >= 0)
    }
}

impl fmt::Debug for Strides {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Strides({:?})", self.strides)
    }
}

impl fmt::Display for Strides {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        for (i, stride) in self.strides.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{stride}")?;
        }
        f.write_str(")")
    }
}

impl From<Vec<isize>> for Strides {
    fn from(strides: Vec<isize>) -> Self {
        Self { strides }
    }
}

impl<const N: usize> From<[isize; N]> for Strides {
    fn from(strides: [isize; N]) -> Self {
        Self {
            strides: strides.to_vec(),
        }
    }
}

impl AsRef<[isize]> for Strides {
    fn as_ref(&self) -> &[isize] {
        &self.strides
    }
}

/// Canonical row-major strides for `dims`.
pub fn dyn_canonical_strides(dims: &[usize]) -> Strides {
    let rank = dims.len();
    let mut strides = vec![0isize; rank];
    let mut acc = 1isize;
    for axis in (0..rank).rev() {
        strides[axis] = acc;
        acc *= dims[axis] as isize;
    }
    Strides { strides }
}

/// Iterator over the indices of a shape, in row-major order.
#[derive(Debug, Clone)]
pub struct Indices<'a> {
    shape: &'a Shape,
    counter: Vec<usize>,
    remaining: usize,
}

impl Iterator for Indices<'_> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let current = self.counter.clone();
        // Advance the mixed-radix counter.
        for axis in (0..self.counter.len()).rev() {
            self.counter[axis] += 1;
            if self.counter[axis] < self.shape.dims[axis] {
                break;
            }
            self.counter[axis] = 0;
        }
        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Indices<'_> {}

/// Description of how one operand maps onto a broadcast result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Broadcast {
    /// Strides of the operand expressed in the *result* shape.
    ///
    /// Axes that were original length `1` get stride `0`, which turns
    /// broadcasting into a plain stride lookup: index arithmetic needs no
    /// special case inside the kernels.
    pub strides: Vec<isize>,
}

impl Broadcast {
    /// Compute the broadcast plan from `shape` to `target`.
    pub fn plan(shape: &Shape, target: &Shape) -> Result<Self> {
        if shape.rank() > target.rank() {
            return Err(NclyError::invalid_shape(
                "broadcast",
                format!("source {shape} has higher rank than target {target}"),
            ));
        }
        let source = shape.canonical_strides();
        let pad = target.rank() - shape.rank();
        let mut strides = vec![0isize; target.rank()];
        for axis in 0..shape.rank() {
            let target_axis = axis + pad;
            let src_len = shape.dims()[axis];
            let dst_len = target.dims()[target_axis];
            if src_len == dst_len {
                strides[target_axis] = source.as_slice()[axis];
            } else if src_len == 1 {
                strides[target_axis] = 0;
            } else {
                return Err(AxisError::IncompatibleLength {
                    axis,
                    lhs: src_len,
                    rhs: dst_len,
                }
                .into());
            }
        }
        Ok(Self { strides })
    }
}

/// Broadcast a set of shapes into one result shape.
pub fn broadcast_shapes<'a, I>(shapes: I) -> Result<Shape>
where
    I: IntoIterator<Item = &'a Shape>,
{
    let mut result: Vec<usize> = Vec::new();
    for shape in shapes {
        let dims = shape.dims();
        if dims.len() > MAX_RANK {
            return Err(AxisError::RankOverflow { max_rank: MAX_RANK }.into());
        }
        if dims.len() > result.len() {
            let pad = dims.len() - result.len();
            let mut grown = vec![1usize; pad];
            grown.extend_from_slice(&result);
            result = grown;
        }
        let offset = result.len() - dims.len();
        for (axis, &len) in dims.iter().enumerate() {
            let target = &mut result[axis + offset];
            if *target == len || len == 1 {
                continue;
            }
            if *target == 1 {
                *target = len;
            } else {
                return Err(AxisError::IncompatibleLength {
                    axis,
                    lhs: *target,
                    rhs: len,
                }
                .into());
            }
        }
    }
    Ok(Shape::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_strides_are_row_major() {
        let shape = Shape::new([2, 3, 4]);
        assert_eq!(shape.canonical_strides(), Strides::new([12isize, 4, 1]));
        assert_eq!(shape.size(), 24);
    }

    #[test]
    fn scalar_shape_has_one_element() {
        let shape = Shape::scalar();
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.size(), 1);
        assert!(shape.is_scalar());
        assert!(!shape.is_empty());
        assert_eq!(shape.canonical_strides(), Strides::new(Vec::<isize>::new()));
    }

    #[test]
    fn empty_shape_is_not_scalar() {
        let shape = Shape::new([0, 3]);
        assert_eq!(shape.size(), 0);
        assert!(shape.is_empty());
        assert!(!shape.is_scalar());
    }

    #[test]
    fn broadcasting_follows_numpy_rules() {
        let a = Shape::new([3, 1, 5]);
        let b = Shape::new([4, 5]);
        assert_eq!(a.broadcast_with(&b).unwrap(), Shape::new([3, 4, 5]));
        assert_eq!(
            Shape::new([1]).broadcast_with(&Shape::scalar()).unwrap(),
            Shape::new([1])
        );
        assert!(Shape::new([2, 3])
            .broadcast_with(&Shape::new([2, 4]))
            .is_err());
    }

    #[test]
    fn broadcast_plan_zeroes_expanded_axes() {
        let plan = Broadcast::plan(&Shape::new([3, 1]), &Shape::new([2, 3, 4])).unwrap();
        assert_eq!(plan.strides, vec![0, 4, 0]);
    }

    #[test]
    fn index_roundtrip() {
        let shape = Shape::new([2, 3, 4]);
        for (flat, index) in shape.indices().enumerate() {
            assert_eq!(shape.offset_of(&index), Some(flat));
            assert_eq!(shape.index_of(flat), Some(index));
        }
        assert_eq!(shape.offset_of(&[2, 0, 0]), None);
    }

    #[test]
    fn reshape_rejects_size_change() {
        let shape = Shape::new([2, 3]);
        assert_eq!(shape.reshape_to([3, 2]).unwrap(), Shape::new([3, 2]));
        assert!(shape.reshape_to([4, 2]).is_err());
    }

    #[test]
    fn reshape_infers_axis() {
        let shape = Shape::new([2, 3, 4]);
        assert_eq!(shape.reshape_to([usize::MAX, 4]).unwrap(), Shape::new([6, 4]));
    }

    #[test]
    fn squeeze_and_unsqueeze() {
        let shape = Shape::new([1, 3, 1]);
        assert_eq!(shape.squeezed(), Shape::new([3]));
        assert_eq!(shape.without_axis(0).unwrap(), Shape::new([3, 1]));
        assert_eq!(shape.with_axis(3, 7).unwrap(), Shape::new([1, 3, 1, 7]));
        assert!(shape.squeeze_axis(1).is_err());
    }
}
