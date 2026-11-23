//! Tensor type, construction and view manipulation.

use ncly_core::{Layout, NclyError, Result, Shape, Strides};
use std::fmt;
use std::sync::Arc;

pub(crate) type Storage = Arc<Vec<f64>>;

/// An n-dimensional, strided, reference-counted array of `f64`.
///
/// Cheap to clone (one `Arc` bump). Views created with [`Tensor::transpose`],
/// [`Tensor::narrow`] or [`Tensor::broadcast_to`] share storage with the source
/// and never copy.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) storage: Storage,
    layout: Layout,
}

impl Tensor {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build a tensor from a row-major buffer.
    ///
    /// # Errors
    ///
    /// Returns [`NclyError::InvalidShape`] when `data.len()` differs from the
    /// number of elements implied by `shape`.
    pub fn from_vec(data: Vec<f64>, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape);
        if shape.size() != data.len() {
            return Err(NclyError::invalid_shape(
                "from_vec",
                format!(
                    "shape {} requires {} elements, got {}",
                    shape,
                    shape.size(),
                    data.len()
                ),
            ));
        }
        let layout = Layout::contiguous(shape);
        Ok(Self {
            storage: Arc::new(data),
            layout,
        })
    }

    /// Build a tensor from a row-major slice.
    pub fn from_slice(data: &[f64], shape: impl Into<Vec<usize>>) -> Result<Self> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Wrap an existing buffer without copying.
    pub fn from_arc(storage: Arc<Vec<f64>>, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape);
        if shape.size() != storage.len() {
            return Err(NclyError::invalid_shape(
                "from_arc",
                format!(
                    "shape {} requires {} elements, got {}",
                    shape,
                    shape.size(),
                    storage.len()
                ),
            ));
        }
        let layout = Layout::contiguous(shape);
        Ok(Self { storage, layout })
    }

    /// A 0-d tensor holding `value`.
    pub fn scalar(value: f64) -> Self {
        Self {
            storage: Arc::new(vec![value]),
            layout: Layout::contiguous(Shape::scalar()),
        }
    }

    /// A tensor of the given shape filled with `value`.
    pub fn full(shape: impl Into<Vec<usize>>, value: f64) -> Self {
        let shape = Shape::new(shape);
        let data = vec![value; shape.size()];
        Self {
            storage: Arc::new(data),
            layout: Layout::contiguous(shape),
        }
    }

    /// A tensor of zeros.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        Self::full(shape, 0.0)
    }

    /// A tensor of ones.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Self {
        Self::full(shape, 1.0)
    }

    /// Build a tensor from a function of the indices.
    ///
    /// ```
    /// use ncly_tensor::Tensor;
    /// let t = Tensor::from_fn([2, 2], |idx| (idx[0] * 2 + idx[1]) as f64);
    /// assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    /// ```
    pub fn from_fn(shape: impl Into<Vec<usize>>, f: impl Fn(&[usize]) -> f64) -> Self {
        let shape = Shape::new(shape);
        let mut data = Vec::with_capacity(shape.size());
        for index in shape.indices() {
            data.push(f(&index));
        }
        Self {
            storage: Arc::new(data),
            layout: Layout::contiguous(shape),
        }
    }

    /// The `n × n` identity matrix.
    pub fn eye(n: usize) -> Self {
        Self::from_fn([n, n], |idx| if idx[0] == idx[1] { 1.0 } else { 0.0 })
    }

    /// `n` evenly spaced values on `[start, stop)`, similar to `numpy.arange`
    /// with a floating point step.
    pub fn arange(start: f64, stop: f64, step: f64) -> Result<Self> {
        if step == 0.0 || !step.is_finite() {
            return Err(NclyError::invalid_shape("arange", "step must be finite and non-zero"));
        }
        let count = ((stop - start) / step).ceil();
        if count < 0.0 {
            return Err(NclyError::invalid_shape(
                "arange",
                "step has the wrong sign for the requested range",
            ));
        }
        let count = count as usize;
        let data = (0..count).map(|i| start + i as f64 * step).collect();
        Self::from_vec(data, [count])
    }

    /// `count` values in `[0, 1)` from a deterministic SplitMix64 generator.
    ///
    /// A reproducible generator is used instead of a `rand` dependency so that
    /// tests, benchmarks and documentation examples produce identical numbers
    /// on every platform.
    pub fn uniform(shape: impl Into<Vec<usize>>, seed: u64) -> Self {
        let shape = Shape::new(shape);
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let data: Vec<f64> = (0..shape.size())
            .map(|_| (next() >> 11) as f64 / (1u64 << 53) as f64)
            .collect();
        Self {
            storage: Arc::new(data),
            layout: Layout::contiguous(shape),
        }
    }

    /// Standard normal samples via the Box–Muller transform.
    pub fn randn(shape: impl Into<Vec<usize>>, seed: u64) -> Self {
        let shape = Shape::new(shape);
        let size = shape.size();
        let uniform = Self::uniform([size], seed);
        let values = uniform.as_slice_unchecked();
        let mut data = Vec::with_capacity(size);
        let mut i = 0;
        while i + 1 < size {
            let u1 = values[i].max(f64::MIN_POSITIVE);
            let u2 = values[i + 1];
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            data.push(r * theta.cos());
            data.push(r * theta.sin());
            i += 2;
        }
        if i < size {
            let u1 = values[i].max(f64::MIN_POSITIVE);
            let r = (-2.0 * u1.ln()).sqrt();
            data.push(r * std::f64::consts::TAU.cos());
        }
        Self {
            storage: Arc::new(data),
            layout: Layout::contiguous(shape),
        }
    }

    /// Wrap a layout and storage; used by the engine and by `ncly-autodiff`.
    pub fn from_parts(storage: Storage, layout: Layout) -> Self {
        Self { storage, layout }
    }

    // -----------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------

    /// The shape.
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// The strides.
    pub fn strides(&self) -> &Strides {
        self.layout.strides()
    }

    /// The layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.shape().size()
    }

    /// True when the tensor has no elements (and is not 0-d).
    pub fn is_empty(&self) -> bool {
        self.shape().is_empty()
    }

    /// True when the tensor is a contiguous view starting at offset 0.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// True when the tensor holds exactly one element.
    pub fn is_scalar(&self) -> bool {
        self.len() == 1 && self.rank() == 0
    }

    /// Length of `axis`.
    pub fn dim(&self, axis: usize) -> Result<usize> {
        self.shape().dim(axis)
    }

    /// Borrowed storage.
    pub fn storage(&self) -> &Arc<Vec<f64>> {
        &self.storage
    }

    // -----------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------

    /// Read one element.
    pub fn get(&self, index: &[usize]) -> Result<f64> {
        let flat = self.layout.flat(index)?;
        Ok(self.storage[flat])
    }

    /// The single element of a 0-d tensor.
    pub fn scalar_value(&self) -> Result<f64> {
        if self.len() != 1 {
            return Err(NclyError::invalid_shape(
                "scalar_value",
                format!("expected exactly one element, found {}", self.len()),
            ));
        }
        Ok(self.storage[self.layout.offset()])
    }

    /// Read one element without bounds checks beyond the debug assertion.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `index` has the wrong rank or is out of
    /// bounds. Prefer [`Tensor::get`] in library code.
    pub fn get_unchecked(&self, index: &[usize]) -> f64 {
        debug_assert_eq!(index.len(), self.rank());
        let mut flat = self.layout.offset() as isize;
        for (axis, &i) in index.iter().enumerate() {
            debug_assert!(i < self.shape().dims()[axis]);
            flat += i as isize * self.strides().get(axis).unwrap_or(0);
        }
        self.storage[flat as usize]
    }

    // -----------------------------------------------------------------
    // Materialisation
    // -----------------------------------------------------------------

    /// Copy the logical contents into a fresh contiguous row-major buffer.
    ///
    /// This is the bridge between the strided world and algorithms that need
    /// dense, canonical data (LU, QR, sparse conversion, ...).
    pub fn to_contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        let mut data = Vec::with_capacity(self.len());
        self.for_each_flat(|flat| data.push(self.storage[flat]));
        match Self::from_vec(data, self.shape().dims().to_vec()) {
            Ok(tensor) => tensor,
            Err(_) => Self::zeros(self.shape().dims().to_vec()),
        }
    }

    /// Copy the logical contents into a row-major `Vec`.
    pub fn to_vec(&self) -> Vec<f64> {
        let mut data = Vec::with_capacity(self.len());
        self.for_each_flat(|flat| data.push(self.storage[flat]));
        data
    }

    /// The underlying slice when the tensor is contiguous.
    ///
    /// # Panics
    ///
    /// Panics when the tensor is a non-contiguous view; call
    /// [`Tensor::to_contiguous`] first.
    pub fn as_slice_unchecked(&self) -> &[f64] {
        assert!(
            self.is_contiguous(),
            "as_slice_unchecked requires a contiguous tensor; call to_contiguous() first"
        );
        &self.storage[..self.len()]
    }

    /// The underlying slice, or `None` for strided views.
    pub fn as_slice(&self) -> Option<&[f64]> {
        if self.is_contiguous() {
            Some(&self.storage[..self.len()])
        } else {
            None
        }
    }

    /// Iterate over storage offsets in logical row-major order.
    ///
    /// This is the single traversal primitive every kernel is built on: it
    /// handles strides, offsets and broadcasting without materialising an index
    /// vector per element.
    pub(crate) fn for_each_flat<F: FnMut(usize)>(&self, mut f: F) {
        let rank = self.rank();
        if self.len() == 0 {
            return;
        }
        let dims = self.shape().dims();
        let strides = self.strides().as_slice();
        let base = self.layout.offset() as isize;
        if rank == 0 {
            f(base as usize);
            return;
        }
        let mut counter = vec![0usize; rank];
        let mut flat = base;
        loop {
            f(flat as usize);
            // Increment the mixed-radix counter from the last axis.
            let mut axis = rank - 1;
            loop {
                counter[axis] += 1;
                flat += strides[axis];
                if counter[axis] < dims[axis] {
                    break;
                }
                flat -= (dims[axis] as isize) * strides[axis];
                counter[axis] = 0;
                if axis == 0 {
                    return;
                }
                axis -= 1;
            }
        }
    }

    // -----------------------------------------------------------------
    // Views
    // -----------------------------------------------------------------

    /// Reverse the axis order without copying.
    pub fn transpose(&self) -> Tensor {
        Self {
            storage: Arc::clone(&self.storage),
            layout: self.layout.transpose(),
        }
    }

    /// Swap two axes without copying.
    pub fn swap_axes(&self, a: usize, b: usize) -> Result<Tensor> {
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout: self.layout.swap_axes(a, b)?,
        })
    }

    /// Move `axis` to position `to` without copying.
    pub fn moved_axis(&self, axis: usize, to: usize) -> Result<Tensor> {
        let rank = self.rank();
        if axis >= rank || to >= rank {
            return Err(ncly_core::AxisError::AxisOutOfBounds {
                axis: axis.max(to),
                rank,
            }
            .into());
        }
        let mut order: Vec<usize> = (0..rank).collect();
        let moved = order.remove(axis);
        order.insert(to, moved);
        let mut dims = Vec::with_capacity(rank);
        let mut strides = Vec::with_capacity(rank);
        for &axis in &order {
            dims.push(self.shape().dims()[axis]);
            strides.push(self.strides().get(axis).unwrap_or(0));
        }
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout: Layout::new(Shape::new(dims), Strides::new(strides), self.layout.offset()),
        })
    }

    /// Broadcast to `shape` without copying, using zero strides.
    pub fn broadcast_to(&self, shape: impl Into<Vec<usize>>) -> Result<Tensor> {
        let target = Shape::new(shape);
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout: self.layout.broadcast_to(&target)?,
        })
    }

    /// Reshape. Requires a contiguous tensor.
    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Result<Tensor> {
        let target: Vec<usize> = shape.into();
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout: self.layout.reshape(&target)?,
        })
    }

    /// Slice `axis` to `[start, start + len)` without copying.
    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> Result<Tensor> {
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout: self.layout.narrow(axis, start, len)?,
        })
    }

    /// Insert an axis of length 1 at `axis`.
    pub fn unsqueeze(&self, axis: usize) -> Result<Tensor> {
        if axis > self.rank() {
            return Err(ncly_core::AxisError::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            }
            .into());
        }
        let mut dims = self.shape().dims().to_vec();
        let mut strides = self.strides().as_slice().to_vec();
        // A length-1 axis can use any stride; zero keeps it broadcast-safe.
        let stride = if axis < strides.len() { strides[axis] * dims[axis] as isize } else { 1 };
        dims.insert(axis, 1);
        strides.insert(axis, stride);
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout: Layout::new(Shape::new(dims), Strides::new(strides), self.layout.offset()),
        })
    }

    /// Remove all axes of length 1 without copying.
    pub fn squeeze(&self) -> Tensor {
        let dims: Vec<usize> = self.shape().dims().iter().copied().filter(|&d| d != 1).collect();
        let strides: Vec<isize> = self
            .shape()
            .dims()
            .iter()
            .zip(self.strides().as_slice())
            .filter(|(&d, _)| d != 1)
            .map(|(_, &s)| s)
            .collect();
        Self {
            storage: Arc::clone(&self.storage),
            layout: Layout::new(Shape::new(dims), Strides::new(strides), self.layout.offset()),
        }
    }

    /// Flatten to 1-d without copying (requires contiguity).
    pub fn flatten(&self) -> Result<Tensor> {
        self.reshape([self.len()])
    }

    // -----------------------------------------------------------------
    // Mutation of owned data
    // -----------------------------------------------------------------

    /// Mutable access to a unique buffer, materialising a copy when shared.
    ///
    /// Returns a tensor whose storage is guaranteed unique, so callers can use
    /// [`Tensor::as_mut_slice`] to write into it.
    pub fn into_unique(self) -> Tensor {
        let contiguous = self.to_contiguous();
        match Arc::try_unwrap(contiguous.storage) {
            Ok(data) => Tensor {
                storage: Arc::new(data),
                layout: contiguous.layout,
            },
            Err(shared) => Tensor {
                storage: Arc::new((*shared).clone()),
                layout: contiguous.layout,
            },
        }
    }

    /// Mutable slice of a unique, contiguous tensor.
    pub fn as_mut_slice(&mut self) -> Result<&mut [f64]> {
        if !self.is_contiguous() {
            return Err(NclyError::invalid_shape(
                "as_mut_slice",
                "tensor is not contiguous",
            ));
        }
        let len = self.len();
        let data = Arc::make_mut(&mut self.storage);
        Ok(&mut data[..len])
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("contiguous", &self.is_contiguous())
            .finish_non_exhaustive()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={}, data={:?}", self.shape(), self.to_vec())?;
        write!(f, ")")
    }
}

impl PartialEq for Tensor {
    /// Logical equality: shapes must match and every element must be equal.
    ///
    /// Layouts may differ (a transposed view compares equal to the contiguous
    /// result of transposing it).
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        if self.is_contiguous() && other.is_contiguous() {
            return self.as_slice() == other.as_slice();
        }
        self.to_vec() == other.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_validates_size() {
        assert!(Tensor::from_vec(vec![1.0, 2.0, 3.0], [2, 2]).is_err());
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        assert_eq!(t.len(), 4);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.strides(), &Strides::new([2isize, 1]));
    }

    #[test]
    fn scalar_tensor() {
        let t = Tensor::scalar(4.0);
        assert!(t.is_scalar());
        assert_eq!(t.scalar_value().unwrap(), 4.0);
    }

    #[test]
    fn transpose_is_a_view() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let tt = t.transpose();
        assert_eq!(Arc::strong_count(&t.storage), 2);
        assert_eq!(tt.shape(), &Shape::new([3, 2]));
        assert_eq!(tt.get(&[2, 1]).unwrap(), 6.0);
        assert_eq!(tt.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn for_each_flat_matches_to_vec() {
        let t = Tensor::from_vec((0..24).map(f64::from).collect(), [2, 3, 4]).unwrap();
        let view = t.transpose().narrow(0, 1, 2).unwrap();
        let expected: Vec<f64> = view.to_vec();
        let mut seen = Vec::new();
        view.for_each_flat(|flat| seen.push(view.storage[flat]));
        assert_eq!(seen, expected);
    }

    #[test]
    fn eye_and_from_fn() {
        assert_eq!(Tensor::eye(2).to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            Tensor::from_fn([2, 2], |i| (i[0] + i[1]) as f64).to_vec(),
            vec![0.0, 1.0, 1.0, 2.0]
        );
    }

    #[test]
    fn arange_counts_correctly() {
        assert_eq!(Tensor::arange(0.0, 1.0, 0.25).unwrap().len(), 4);
        assert!(Tensor::arange(0.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn uniform_is_deterministic_and_bounded() {
        let a = Tensor::uniform([100], 42);
        let b = Tensor::uniform([100], 42);
        assert_eq!(a, b);
        assert!(a.as_slice().unwrap().iter().all(|&v| (0.0..1.0).contains(&v)));
        let c = Tensor::uniform([100], 43);
        assert_ne!(a, c);
    }

    #[test]
    fn randn_has_reasonable_moments() {
        let t = Tensor::randn([20_000], 7);
        let v = t.as_slice().unwrap();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }

    #[test]
    fn narrow_and_unsqueeze() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let row = t.narrow(0, 1, 1).unwrap();
        assert_eq!(row.shape(), &Shape::new([1, 2]));
        assert_eq!(row.to_vec(), vec![3.0, 4.0]);
        let unsq = row.unsqueeze(0).unwrap();
        assert_eq!(unsq.shape(), &Shape::new([1, 1, 2]));
        assert_eq!(unsq.to_vec(), vec![3.0, 4.0]);
        assert!(t.narrow(0, 1, 2).is_err());
    }

    #[test]
    fn moved_axis_keeps_data_order_semantics() {
        let t = Tensor::from_vec((0..24).map(f64::from).collect(), [2, 3, 4]).unwrap();
        let moved = t.moved_axis(0, 2).unwrap();
        assert_eq!(moved.shape(), &Shape::new([3, 4, 2]));
        // moving axis 0 to the end is equivalent to transposing (2,3,4) -> (3,4,2)
        assert_eq!(moved.get(&[1, 2, 0]).unwrap(), t.get(&[0, 1, 2]).unwrap());
    }

    #[test]
    fn squeeze_removes_singletons() {
        let t = Tensor::zeros([1, 3, 1]);
        let s = t.squeeze();
        assert_eq!(s.shape(), &Shape::new([3]));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn empty_tensors_are_valid() {
        let t = Tensor::zeros([0, 3]);
        assert!(t.is_empty());
        assert_eq!(t.to_vec().len(), 0);
    }
}
