//! Python bindings for NCLY.
//!
//! The extension module is named `ncly` and is built with maturin. It exposes a
//! small, explicit API: a [`Tensor`] class for data and element-wise math, a
//! [`GradTensor`] participating in reverse-mode differentiation, and module level
//! helpers for linear algebra and sparse formats.
//!
//! # Error mapping
//!
//! Every fallible Rust call becomes a Python exception:
//!
//! | Rust error | Python exception |
//! | --- | --- |
//! | shape / index problems | `ValueError` |
//! | singular matrices | `ZeroDivisionError` |
//! | non-convergence | `ArithmeticError` |
//! | malformed sparse data | `ValueError` |
//!
//! This keeps Python code idiomatic: `try: ... except ValueError:` works without
//! importing a custom exception hierarchy.
//!
//! # Memory
//!
//! `Tensor` owns a reference-counted `Vec<f64>`; slicing operations return views
//! that share it. `GradTensor` holds a handle into a shared autodiff tape, so
//! building an expression graph is `O(nodes)` Python objects but a single Rust
//! `Arc` per tape.

use ncly_core::NclyError;
use ncly_tensor::Tensor as RsTensor;
use pyo3::exceptions::{PyArithmeticError, PyIndexError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::PyList;

fn to_py_err(error: NclyError) -> PyErr {
    match &error {
        NclyError::IndexOutOfBounds { .. } => PyIndexError::new_err(error.to_string()),
        NclyError::Singular { .. } => PyZeroDivisionError::new_err(error.to_string()),
        NclyError::Convergence { .. } => PyArithmeticError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn fallible<T>(result: Result<T, NclyError>) -> PyResult<T> {
    result.map_err(to_py_err)
}

/// An `n`-dimensional array of `f64`.
///
/// ```python
/// import ncly
/// a = ncly.Tensor([[1.0, 2.0], [3.0, 4.0]])
/// b = a.exp()
/// print(b.sum())
/// ```
#[pyclass(name = "Tensor", module = "ncly")]
#[derive(Clone)]
pub struct PyTensor {
    inner: RsTensor,
}

impl PyTensor {
    fn wrap(inner: RsTensor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTensor {
    /// ``Tensor(data, shape=None)``
    ///
    /// Build from a Python sequence (nested lists describe the shape) or from a
    /// flat sequence plus an explicit shape.
    #[new]
    #[pyo3(signature = (data, shape=None))]
    fn new(data: &Bound<'_, PyAny>, shape: Option<Vec<usize>>) -> PyResult<Self> {
        let (flat, inferred) = flatten_sequence(data)?;
        let shape = match shape {
            Some(shape) => shape,
            None => inferred,
        };
        let tensor = fallible(RsTensor::from_vec(flat, shape))?;
        Ok(Self::wrap(tensor))
    }

    /// Shape as a tuple of axis lengths.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.rank()
    }

    /// Number of elements.
    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }

    /// Whether the data is stored contiguously.
    #[getter]
    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    /// Row-major values as a flat list.
    fn tolist(&self) -> Vec<f64> {
        self.inner.to_vec()
    }

    /// Nested Python lists.
    fn to_nested(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| nest(py, &self.inner.to_vec(), self.inner.shape().dims(), 0))
    }

    fn __len__(&self) -> usize {
        self.inner.shape().dims().first().copied().unwrap_or(0)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, data={:?})",
            self.inner.shape().dims(),
            self.inner.to_vec()
        )
    }

    /// Item access: ``t[i, j]`` or ``t[i]`` for a sub-tensor.
    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            if let Ok(single) = index.extract::<usize>() {
                if self.inner.rank() == 0 {
                    return Err(PyIndexError::new_err("0-d tensor cannot be indexed"));
                }
                let len = self.inner.shape().dims()[0];
                if single >= len {
                    return Err(PyIndexError::new_err(format!(
                        "index {single} out of range for axis of length {len}"
                    )));
                }
                if self.inner.rank() == 1 {
                    let value = fallible(self.inner.get(&[single]))?;
                    return Ok(value.into_pyobject(py)?.into_any().unbind());
                }
                let view = fallible(self.inner.narrow(0, single, 1))?.squeeze();
                return Ok(Py::new(py, Self::wrap(view))?.into_any());
            }
            let indices: Vec<usize> = index.extract()?;
            let value = fallible(self.inner.get(&indices))?;
            Ok(value.into_pyobject(py)?.into_any().unbind())
        })
    }

    // --- element-wise arithmetic -------------------------------------

    fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::wrap(fallible(self.inner.add(&other.inner))?))
    }

    fn __sub__(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::wrap(fallible(self.inner.sub(&other.inner))?))
    }

    fn __mul__(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::wrap(fallible(self.inner.mul(&other.inner))?))
    }

    fn __truediv__(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::wrap(fallible(self.inner.div(&other.inner))?))
    }

    fn __neg__(&self) -> Self {
        Self::wrap(self.inner.neg())
    }

    /// Element-wise addition (explicit form, mirrors the Rust API).
    fn add(&self, other: &PyTensor) -> PyResult<Self> {
        self.__add__(other)
    }

    /// Element-wise subtraction.
    fn sub(&self, other: &PyTensor) -> PyResult<Self> {
        self.__sub__(other)
    }

    /// Element-wise multiplication.
    fn mul(&self, other: &PyTensor) -> PyResult<Self> {
        self.__mul__(other)
    }

    /// Element-wise division.
    fn div(&self, other: &PyTensor) -> PyResult<Self> {
        self.__truediv__(other)
    }

    /// `e^x`.
    fn exp(&self) -> Self {
        Self::wrap(self.inner.exp())
    }

    /// Natural logarithm.
    fn ln(&self) -> Self {
        Self::wrap(self.inner.ln())
    }

    /// Square root.
    fn sqrt(&self) -> Self {
        Self::wrap(self.inner.sqrt())
    }

    /// Sine.
    fn sin(&self) -> Self {
        Self::wrap(self.inner.sin())
    }

    /// Cosine.
    fn cos(&self) -> Self {
        Self::wrap(self.inner.cos())
    }

    /// Hyperbolic tangent.
    fn tanh(&self) -> Self {
        Self::wrap(self.inner.tanh())
    }

    /// Absolute value.
    fn abs(&self) -> Self {
        Self::wrap(self.inner.abs())
    }

    /// `x^p` for a real exponent.
    fn powf(&self, exponent: f64) -> Self {
        Self::wrap(self.inner.powf(exponent))
    }

    /// Multiply by a scalar.
    fn scale(&self, factor: f64) -> Self {
        Self::wrap(self.inner.muls(factor))
    }

    /// Sum of all elements.
    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    /// Mean of all elements.
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Largest element.
    fn max(&self) -> f64 {
        self.inner.max()
    }

    /// Smallest element.
    fn min(&self) -> f64 {
        self.inner.min()
    }

    /// Sum along `axis`.
    fn sum_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self::wrap(fallible(self.inner.sum_axis(axis))?))
    }

    /// Mean along `axis`.
    fn mean_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self::wrap(fallible(self.inner.mean_axis(axis))?))
    }

    /// Reshape.
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self::wrap(fallible(self.inner.reshape(shape))?))
    }

    /// Transpose (reverses the axis order).
    fn transpose(&self) -> Self {
        Self::wrap(self.inner.transpose())
    }

    /// Matrix product of the last two axes.
    fn matmul(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::wrap(fallible(self.inner.matmul(&other.inner))?))
    }

    /// Frobenius norm.
    fn norm(&self) -> f64 {
        self.inner.norm()
    }

    /// True when any element is NaN.
    fn has_nan(&self) -> bool {
        self.inner.has_nan()
    }

    /// Mark this tensor as requiring gradients and return a graph handle.
    ///
    /// The returned object supports arithmetic and a `backward()` call; see
    /// [`GradTensor`].
    fn requires_grad(&self) -> PyGradTensor {
        PyGradTensor::from_tensor(self.inner.clone())
    }
}

/// A tensor that participates in reverse-mode automatic differentiation.
///
/// ```python
/// import ncly
/// x = ncly.Tensor([1.0, 2.0, 3.0]).requires_grad()
/// y = (x * x).sum()
/// y.backward()
/// print(x.grad)
/// ```
#[pyclass(name = "GradTensor", module = "ncly")]
pub struct PyGradTensor {
    inner: ncly_autodiff::Variable,
}

impl PyGradTensor {
    fn from_tensor(tensor: RsTensor) -> Self {
        Self {
            inner: ncly_autodiff::Variable::from_tensor(tensor, "leaf"),
        }
    }

    /// Build a leaf on the caller's tape (so gradients reach the original leaf).
    fn on_same_tape(&self, tensor: RsTensor) -> Self {
        Self {
            inner: ncly_autodiff::Variable::on_tape(self.inner.tape(), tensor, "leaf"),
        }
    }
}

#[pymethods]
impl PyGradTensor {
    /// ``GradTensor(data)`` — a new leaf with its own tape.
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (flat, shape) = flatten_sequence(data)?;
        let tensor = fallible(RsTensor::from_vec(flat, shape))?;
        Ok(Self::from_tensor(tensor))
    }

    /// Current value as a `Tensor`.
    fn value(&self) -> PyTensor {
        PyTensor::wrap(self.inner.value())
    }

    /// Accumulated gradient as a `Tensor`.
    fn grad(&self) -> PyTensor {
        PyTensor::wrap(self.inner.grad())
    }

    /// Shape of the underlying value.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.value().shape().dims().to_vec()
    }

    fn __add__(&self, other: &PyGradTensor) -> Self {
        Self {
            inner: self.inner.add(&other.inner),
        }
    }

    fn __sub__(&self, other: &PyGradTensor) -> Self {
        Self {
            inner: self.inner.sub(&other.inner),
        }
    }

    fn __mul__(&self, other: &PyGradTensor) -> Self {
        Self {
            inner: self.inner.mul(&other.inner),
        }
    }

    fn __truediv__(&self, other: &PyGradTensor) -> Self {
        Self {
            inner: self.inner.div(&other.inner),
        }
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    /// `e^self`.
    fn exp(&self) -> Self {
        Self {
            inner: self.inner.exp(),
        }
    }

    /// `ln(self)`.
    fn ln(&self) -> Self {
        Self {
            inner: self.inner.ln(),
        }
    }

    /// `sin(self)`.
    fn sin(&self) -> Self {
        Self {
            inner: self.inner.sin(),
        }
    }

    /// `cos(self)`.
    fn cos(&self) -> Self {
        Self {
            inner: self.inner.cos(),
        }
    }

    /// `tanh(self)`.
    fn tanh(&self) -> Self {
        Self {
            inner: self.inner.tanh(),
        }
    }

    /// `self^n`.
    fn powi(&self, exponent: i32) -> Self {
        Self {
            inner: self.inner.powi(exponent),
        }
    }

    /// Multiply by a constant.
    fn scale(&self, factor: f64) -> Self {
        Self {
            inner: self.inner.scale(factor),
        }
    }

    /// Sum of all elements, as a 0-d differentiable value.
    fn sum(&self) -> Self {
        Self {
            inner: self.inner.sum(),
        }
    }

    /// Mean of all elements.
    fn mean(&self) -> Self {
        Self {
            inner: self.inner.mean(),
        }
    }

    /// Sum along one axis.
    fn sum_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self {
            inner: fallible(self.inner.sum_axis(axis))?,
        })
    }

    /// Matrix product.
    fn matmul(&self, other: &PyGradTensor) -> PyResult<Self> {
        Ok(Self {
            inner: fallible(self.inner.matmul(&other.inner))?,
        })
    }

    /// Reshape.
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: fallible(self.inner.reshape(shape))?,
        })
    }

    /// Reverse-mode sweep starting from this value.
    fn backward(&self) {
        self.inner.backward();
    }

    /// Reset accumulated gradients for this value's tape.
    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    /// Root mean square error against `target` (convenience for examples).
    fn mse(&self, target: &PyGradTensor) -> PyResult<Self> {
        let inner = fallible(ncly_autodiff::ops::mse(&self.inner, &target.inner))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "GradTensor(shape={:?}, value={:?})",
            self.inner.value().shape().dims(),
            self.inner.value().to_vec()
        )
    }

    /// Clone as a constant on the same tape (useful for shifting values).
    fn constant(&self, value: f64) -> Self {
        self.on_same_tape(RsTensor::scalar(value))
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Solve `A x = b` for a square `A` given as nested lists.
#[pyfunction]
fn solve(a: Vec<Vec<f64>>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    let n = a.len();
    let mut flat = Vec::with_capacity(n * n);
    for row in &a {
        if row.len() != n {
            return Err(PyValueError::new_err("matrix must be square"));
        }
        flat.extend_from_slice(row);
    }
    fallible(ncly_linalg::solve(&flat, &b, n))
}

/// Least-squares solution of `A x ≈ b` for an `m × n` matrix.
#[pyfunction]
fn lstsq(a: Vec<Vec<f64>>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    let m = a.len();
    let n = a.first().map(Vec::len).unwrap_or(0);
    let mut flat = Vec::with_capacity(m * n);
    for row in &a {
        if row.len() != n {
            return Err(PyValueError::new_err("matrix rows must have equal length"));
        }
        flat.extend_from_slice(row);
    }
    if b.len() != m {
        return Err(PyValueError::new_err("b has the wrong length"));
    }
    fallible(ncly_linalg::lstsq(&flat, &b, m, n))
}

/// Inverse of a square matrix given as nested lists.
#[pyfunction]
fn inverse(a: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let n = a.len();
    let mut flat = Vec::with_capacity(n * n);
    for row in &a {
        if row.len() != n {
            return Err(PyValueError::new_err("matrix must be square"));
        }
        flat.extend_from_slice(row);
    }
    let inverted = fallible(ncly_linalg::inverse(&flat, n))?;
    Ok(inverted.chunks(n).map(<[f64]>::to_vec).collect())
}

/// Eigenvalues of a symmetric matrix, ascending.
#[pyfunction]
fn eigvalsh(a: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let n = a.len();
    let mut flat = Vec::with_capacity(n * n);
    for row in &a {
        if row.len() != n {
            return Err(PyValueError::new_err("matrix must be square"));
        }
        flat.extend_from_slice(row);
    }
    let eigen = fallible(ncly_linalg::eigen::symmetric(&flat, n))?;
    Ok(eigen.values)
}

/// Sparse CSR matrix constructed from coordinate lists.
#[pyclass(name = "CsrMatrix", module = "ncly")]
pub struct PyCsr {
    inner: ncly_sparse::CsrMatrix,
}

#[pymethods]
impl PyCsr {
    #[new]
    fn new(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> PyResult<Self> {
        let inner = fallible(ncly_sparse::CsrMatrix::from_triplets(
            rows,
            cols,
            &row_indices,
            &col_indices,
            &values,
        ))?;
        Ok(Self { inner })
    }

    /// Number of stored entries.
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Shape as `(rows, cols)`.
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Sparse matrix–vector product.
    fn spmv(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        fallible(self.inner.try_spmv(&x))
    }

    /// Dense row-major copy as nested lists.
    fn to_dense(&self) -> Vec<Vec<f64>> {
        let cols = self.inner.cols();
        self.inner
            .to_dense()
            .chunks(cols.max(1))
            .map(<[f64]>::to_vec)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CsrMatrix(shape={:?}, nnz={})",
            self.inner.shape(),
            self.inner.nnz()
        )
    }
}

/// Render a flat buffer as nested Python lists.
fn nest(py: Python<'_>, data: &[f64], dims: &[usize], offset: usize) -> PyResult<PyObject> {
    if dims.is_empty() {
        return Ok(data[offset].into_pyobject(py)?.into_any().unbind());
    }
    if dims.len() == 1 {
        let values: Vec<f64> = (0..dims[0])
            .map(|i| data[offset + i])
            .collect();
        return Ok(PyList::new(py, values)?.into_any().unbind());
    }
    let stride: usize = dims[1..].iter().product();
    let list = PyList::empty(py);
    for i in 0..dims[0] {
        list.append(nest(py, data, &dims[1..], offset + i * stride)?)?;
    }
    Ok(list.into_any().unbind())
}

/// Flatten a (possibly nested) Python sequence, inferring its shape.
fn flatten_sequence(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let mut values = Vec::new();
    let mut shape = Vec::new();
    collect(data, &mut values, &mut shape, 0)?;
    Ok((values, shape))
}

fn collect(
    object: &Bound<'_, PyAny>,
    values: &mut Vec<f64>,
    shape: &mut Vec<usize>,
    depth: usize,
) -> PyResult<()> {
    if let Ok(value) = object.extract::<f64>() {
        if depth != shape.len() {
            return Err(PyValueError::new_err("ragged nested sequence"));
        }
        values.push(value);
        return Ok(());
    }
    let sequence = object
        .try_iter()
        .map_err(|_| PyValueError::new_err("expected a number or a sequence of numbers"))?;
    let items: Vec<Bound<'_, PyAny>> = sequence.collect::<PyResult<Vec<_>>>()?;
    let length = items.len();
    if depth < shape.len() {
        if shape[depth] != length {
            return Err(PyValueError::new_err("ragged nested sequence"));
        }
    } else if depth == shape.len() {
        shape.push(length);
    } else {
        return Err(PyValueError::new_err("ragged nested sequence"));
    }
    for item in items {
        collect(&item, values, shape, depth + 1)?;
    }
    Ok(())
}

/// Python module `ncly`.
#[pymodule]
fn ncly(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensor>()?;
    module.add_class::<PyGradTensor>()?;
    module.add_class::<PyCsr>()?;
    module.add_function(wrap_pyfunction!(solve, module)?)?;
    module.add_function(wrap_pyfunction!(lstsq, module)?)?;
    module.add_function(wrap_pyfunction!(inverse, module)?)?;
    module.add_function(wrap_pyfunction!(eigvalsh, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
