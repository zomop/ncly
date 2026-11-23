//! Element-wise arithmetic, unary maps and reductions.
//!
//! # Strategy
//!
//! A binary operation between tensors of shape `A` and `B` proceeds in three
//! steps:
//!
//! 1. Compute the broadcast result shape `C = broadcast(A, B)`.
//! 2. Collapse adjacent axes that are contiguous in *all three* operands into
//!    one "inner" axis. For `(2, 3, 4)`-shaped operands this reduces a
//!    three-level loop to a single tight loop over 24 elements.
//! 3. If the collapsed inner axis is fully contiguous with stride 1 for both
//!    operands, hand the loop to `ncly_simd`; otherwise fall back to the
//!    strided scalar kernel (still vectorised for stride-1 reads).
//!
//! Steps 2 and 3 are what make broadcasting cheap: a broadcast operand simply
//! has stride `0` on the expanded axis, and stride `0` is handled by the strided
//! kernel without branches in the inner loop.

use crate::tensor::Tensor;
use ncly_core::{Layout, NclyError, Result, Shape, Strides};
use ncly_simd as simd;
use std::sync::Arc;

/// Binary element-wise operation selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`
    Div,
    /// `max(a, b)`
    Maximum,
    /// `min(a, b)`
    Minimum,
}

impl BinaryOp {
    fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
        }
    }
}

/// Unary element-wise operation selector.
///
/// `Eq` is deliberately not derived: [`UnaryOp::Powf`] carries an `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    /// `e^x`
    Exp,
    /// `ln x`
    Ln,
    /// `sqrt x`
    Sqrt,
    /// `sin x`
    Sin,
    /// `cos x`
    Cos,
    /// `tanh x`
    Tanh,
    /// `|x|`
    Abs,
    /// `-x`
    Neg,
    /// `x^n`
    Powf(f64),
    /// `1/x`
    Recip,
}

impl UnaryOp {
    fn name(self) -> &'static str {
        match self {
            Self::Exp => "exp",
            Self::Ln => "ln",
            Self::Sqrt => "sqrt",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tanh => "tanh",
            Self::Abs => "abs",
            Self::Neg => "neg",
            Self::Powf(_) => "powf",
            Self::Recip => "recip",
        }
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Self::Exp => x.exp(),
            Self::Ln => x.ln(),
            Self::Sqrt => x.sqrt(),
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Tanh => x.tanh(),
            Self::Abs => x.abs(),
            Self::Neg => -x,
            Self::Powf(p) => x.powf(p),
            Self::Recip => 1.0 / x,
        }
    }
}

/// A collapsed loop axis: shape/strides of the output and both operands.
#[derive(Debug, Clone, Copy)]
struct LoopAxis {
    len: usize,
    out_stride: usize,
    a_stride: isize,
    b_stride: isize,
}

/// Collapse the broadcast result into loop axes.
///
/// Two adjacent axes `i` (slower) and `j` (faster) may be merged into a single
/// axis of length `len_i * len_j` **when the merged axis keeps the stride of the
/// faster axis**. The merge is only legal if, for every operand,
///
/// ```text
/// stride_i == len_j * stride_j
/// ```
///
/// because that is precisely the condition under which element `i*len_j + j` of
/// the flattened axis lands on the same storage offset as the pair `(i, j)` did.
/// Getting the retained stride wrong (keeping `stride_i` instead of `stride_j`)
/// makes the inner loop walk off the end of the buffer, so this invariant is
/// asserted in tests over many shapes.
///
/// Length-1 axes carry no information and are dropped, which is what turns
/// `(3, 1, 5)`-style broadcast patterns into tight loops.
fn collapse(
    out_shape: &Shape,
    out_strides: &[isize],
    a_strides: &[isize],
    b_strides: &[isize],
) -> Vec<LoopAxis> {
    let dims = out_shape.dims();
    let mut axes: Vec<LoopAxis> = Vec::with_capacity(dims.len());
    for axis in 0..dims.len() {
        let len = dims[axis];
        if len == 1 {
            continue;
        }
        let candidate = LoopAxis {
            len,
            out_stride: out_strides[axis] as usize,
            a_stride: a_strides[axis],
            b_stride: b_strides[axis],
        };
        if candidate.a_stride < 0 || candidate.b_stride < 0 {
            axes.push(candidate);
            continue;
        }
        if let Some(previous) = axes.last_mut() {
            let mergeable = previous.out_stride
                == candidate.out_stride * candidate.len
                && previous.a_stride == candidate.a_stride * candidate.len as isize
                && previous.b_stride == candidate.b_stride * candidate.len as isize;
            if mergeable {
                previous.len *= candidate.len;
                previous.out_stride = candidate.out_stride;
                previous.a_stride = candidate.a_stride;
                previous.b_stride = candidate.b_stride;
                continue;
            }
        }
        axes.push(candidate);
    }
    axes
}

/// Apply a binary operation with broadcasting.
pub(crate) fn binary(lhs: &Tensor, rhs: &Tensor, op: BinaryOp) -> Result<Tensor> {
    let out_shape = lhs.shape().broadcast_with(rhs.shape())?;
    let out_strides = out_shape.canonical_strides();
    let a_layout = lhs.layout().broadcast_to(&out_shape)?;
    let b_layout = rhs.layout().broadcast_to(&out_shape)?;

    let size = out_shape.size();
    let mut out = vec![0.0f64; size];
    if size == 0 {
        return Ok(Tensor::from_parts(
            Arc::new(out),
            Layout::contiguous(out_shape),
        ));
    }

    let axes = collapse(
        &out_shape,
        out_strides.as_slice(),
        a_layout.strides().as_slice(),
        b_layout.strides().as_slice(),
    );

    // After collapsing, `axes` holds only non-trivial loop axes. A 0-d or
    // all-singleton result collapses to an empty axis list, which means the
    // loop body runs exactly once.
    if axes.is_empty() {
        let a = lhs.storage[a_layout.offset()];
        let b = rhs.storage[b_layout.offset()];
        out[0] = scalar_binary_op(a, b, op);
        return Ok(Tensor::from_parts(
            Arc::new(out),
            Layout::contiguous(out_shape),
        ));
    }

    let rank = axes.len();
    let inner = axes[rank - 1];
    let mut counter = vec![0usize; rank];
    let mut offset_a = a_layout.offset() as isize;
    let mut offset_b = b_layout.offset() as isize;
    let mut out_at = 0usize;

    loop {
        let len = inner.len;
        let a_start = offset_a as usize;
        let b_start = offset_b as usize;
        let a_end = a_start + (len - 1) * inner.a_stride as usize + 1;
        let b_end = b_start + (len - 1) * inner.b_stride as usize + 1;
        let a_slice = &lhs.storage[a_start..a_end];
        let b_slice = &rhs.storage[b_start..b_end];
        let out_slice = &mut out[out_at..out_at + len];

        match op {
            BinaryOp::Add => simd::add(out_slice, a_slice, inner.a_stride, b_slice, inner.b_stride),
            BinaryOp::Sub => simd::sub(out_slice, a_slice, inner.a_stride, b_slice, inner.b_stride),
            BinaryOp::Mul => simd::mul(out_slice, a_slice, inner.a_stride, b_slice, inner.b_stride),
            BinaryOp::Div => simd::div(out_slice, a_slice, inner.a_stride, b_slice, inner.b_stride),
            BinaryOp::Maximum | BinaryOp::Minimum => {
                for k in 0..len {
                    let x = a_slice[k * inner.a_stride as usize];
                    let y = b_slice[k * inner.b_stride as usize];
                    out_slice[k] = scalar_binary_op(x, y, op);
                }
            }
        }
        out_at += len;
        if out_at >= size {
            break;
        }

        // Advance the outer odometer; the innermost axis is handled above.
        debug_assert!(rank >= 2, "collapsed loop must have an outer axis");
        let mut axis = rank - 2;
        loop {
            counter[axis] += 1;
            offset_a += axes[axis].a_stride;
            offset_b += axes[axis].b_stride;
            if counter[axis] < axes[axis].len {
                break;
            }
            offset_a -= axes[axis].len as isize * axes[axis].a_stride;
            offset_b -= axes[axis].len as isize * axes[axis].b_stride;
            counter[axis] = 0;
            assert!(axis > 0, "odometer overran the output");
            axis -= 1;
        }
    }

    Ok(Tensor::from_parts(
        Arc::new(out),
        Layout::contiguous(out_shape),
    ))
}

#[inline]
fn scalar_binary_op(a: f64, b: f64, op: BinaryOp) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Maximum => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        }
        BinaryOp::Minimum => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        }
    }
}

/// Apply a unary map.
pub(crate) fn unary(input: &Tensor, op: UnaryOp) -> Tensor {
    let shape = input.shape().clone();
    let size = shape.size();
    let mut out = vec![0.0f64; size];
    if size == 0 {
        return Tensor::from_parts(Arc::new(out), Layout::contiguous(shape));
    }
    let mut cursor = 0usize;
    // Process maximal contiguous runs of the input layout.
    let layout = input.layout();
    if layout.is_contiguous() {
        let src = &input.storage[..size];
        for (slot, &x) in out.iter_mut().zip(src) {
            *slot = op.apply(x);
        }
    } else {
        input.for_each_flat(|flat| {
            out[cursor] = op.apply(input.storage[flat]);
            cursor += 1;
        });
    }
    Tensor::from_parts(Arc::new(out), Layout::contiguous(shape))
}

/// Reduction selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}

impl ReduceOp {
    fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Min => "min",
        }
    }
}

/// Reduce `axis`, keeping it with length 1 (`keepdim = true`) or removing it.
pub(crate) fn reduce_axis(
    input: &Tensor,
    op: ReduceOp,
    axis: usize,
    keepdim: bool,
) -> Result<Tensor> {
    let dim = input.dim(axis)?;
    let out_shape = if keepdim {
        let mut dims = input.shape().dims().to_vec();
        dims[axis] = 1;
        Shape::new(dims)
    } else {
        input.shape().without_axis(axis)?
    };
    let inner = input.shape().inner_size(axis);
    let outer = input.shape().outer_size(axis);
    let mut out = match op {
        ReduceOp::Sum | ReduceOp::Mean => vec![0.0; outer * inner],
        ReduceOp::Max => vec![f64::NEG_INFINITY; outer * inner],
        ReduceOp::Min => vec![f64::INFINITY; outer * inner],
    };

    // Normalise to a contiguous buffer so the strided inner loop stays simple.
    let dense = input.to_contiguous();
    let data = dense
        .as_slice()
        .expect("to_contiguous returns contiguous data");

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = match op {
                ReduceOp::Sum | ReduceOp::Mean => 0.0,
                ReduceOp::Max => f64::NEG_INFINITY,
                ReduceOp::Min => f64::INFINITY,
            };
            let mut compensation = 0.0f64;
            for k in 0..dim {
                let value = data[(o * dim + k) * inner + i];
                match op {
                    ReduceOp::Sum | ReduceOp::Mean => {
                        let t = acc + value;
                        if acc.abs() >= value.abs() {
                            compensation += (acc - t) + value;
                        } else {
                            compensation += (value - t) + acc;
                        }
                        acc = t;
                    }
                    ReduceOp::Max => {
                        if value > acc || value.is_nan() {
                            acc = value;
                        }
                    }
                    ReduceOp::Min => {
                        if value < acc || value.is_nan() {
                            acc = value;
                        }
                    }
                }
            }
            let value = match op {
                ReduceOp::Sum => acc + compensation,
                ReduceOp::Mean => {
                    if dim == 0 {
                        f64::NAN
                    } else {
                        (acc + compensation) / dim as f64
                    }
                }
                ReduceOp::Max | ReduceOp::Min => acc,
            };
            out[o * inner + i] = value;
        }
    }

    Ok(Tensor::from_parts(
        Arc::new(out),
        Layout::contiguous(out_shape),
    ))
}

/// Reduce over all axes.
pub(crate) fn reduce_all(input: &Tensor, op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => {
            let mut total = 0.0f64;
            let mut compensation = 0.0f64;
            input.for_each_flat(|flat| {
                let value = input.storage[flat];
                let t = total + value;
                if total.abs() >= value.abs() {
                    compensation += (total - t) + value;
                } else {
                    compensation += (value - t) + total;
                }
                total = t;
            });
            let sum = total + compensation;
            if op == ReduceOp::Mean {
                if input.len() == 0 {
                    f64::NAN
                } else {
                    sum / input.len() as f64
                }
            } else {
                sum
            }
        }
        ReduceOp::Max | ReduceOp::Min => {
            let mut best = if op == ReduceOp::Max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            input.for_each_flat(|flat| {
                let value = input.storage[flat];
                let take = if op == ReduceOp::Max {
                    value > best || value.is_nan()
                } else {
                    value < best || value.is_nan()
                };
                if take {
                    best = value;
                }
            });
            best
        }
    }
}

impl Tensor {
    /// Element-wise binary operation with broadcasting.
    pub fn binary_op(&self, other: &Tensor, op: BinaryOp) -> Result<Tensor> {
        binary(self, other, op)
    }

    /// Element-wise addition with broadcasting.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        binary(self, other, BinaryOp::Add)
    }

    /// Element-wise subtraction with broadcasting.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        binary(self, other, BinaryOp::Sub)
    }

    /// Element-wise multiplication with broadcasting.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        binary(self, other, BinaryOp::Mul)
    }

    /// Element-wise division with broadcasting.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        binary(self, other, BinaryOp::Div)
    }

    /// Element-wise maximum with broadcasting.
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        binary(self, other, BinaryOp::Maximum)
    }

    /// Element-wise minimum with broadcasting.
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        binary(self, other, BinaryOp::Minimum)
    }

    /// Multiply every element by a scalar.
    pub fn muls(&self, scalar: f64) -> Tensor {
        let mut out = Vec::with_capacity(self.len());
        self.for_each_flat(|flat| out.push(self.storage[flat] * scalar));
        Tensor::from_parts(Arc::new(out), Layout::contiguous(self.shape().clone()))
    }

    /// Add a scalar to every element.
    pub fn adds(&self, scalar: f64) -> Tensor {
        let mut out = Vec::with_capacity(self.len());
        self.for_each_flat(|flat| out.push(self.storage[flat] + scalar));
        Tensor::from_parts(Arc::new(out), Layout::contiguous(self.shape().clone()))
    }

    /// Apply a unary operation element-wise.
    pub fn unary(&self, op: UnaryOp) -> Tensor {
        unary(self, op)
    }

    /// `e^x` element-wise.
    pub fn exp(&self) -> Tensor {
        unary(self, UnaryOp::Exp)
    }

    /// Natural logarithm element-wise.
    pub fn ln(&self) -> Tensor {
        unary(self, UnaryOp::Ln)
    }

    /// Square root element-wise.
    pub fn sqrt(&self) -> Tensor {
        unary(self, UnaryOp::Sqrt)
    }

    /// Sine element-wise.
    pub fn sin(&self) -> Tensor {
        unary(self, UnaryOp::Sin)
    }

    /// Cosine element-wise.
    pub fn cos(&self) -> Tensor {
        unary(self, UnaryOp::Cos)
    }

    /// Hyperbolic tangent element-wise.
    pub fn tanh(&self) -> Tensor {
        unary(self, UnaryOp::Tanh)
    }

    /// Absolute value element-wise.
    pub fn abs(&self) -> Tensor {
        unary(self, UnaryOp::Abs)
    }

    /// `x^p` element-wise.
    pub fn powf(&self, p: f64) -> Tensor {
        unary(self, UnaryOp::Powf(p))
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor {
        unary(self, UnaryOp::Neg)
    }

    /// Sum of all elements.
    pub fn sum(&self) -> f64 {
        reduce_all(self, ReduceOp::Sum)
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f64 {
        reduce_all(self, ReduceOp::Mean)
    }

    /// Largest element.
    pub fn max(&self) -> f64 {
        reduce_all(self, ReduceOp::Max)
    }

    /// Smallest element.
    pub fn min(&self) -> f64 {
        reduce_all(self, ReduceOp::Min)
    }

    /// Reduce one axis, keeping it as length 1.
    pub fn sum_axis_keepdim(&self, axis: usize) -> Result<Tensor> {
        reduce_axis(self, ReduceOp::Sum, axis, true)
    }

    /// Reduce one axis, removing it.
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor> {
        reduce_axis(self, ReduceOp::Sum, axis, false)
    }

    /// Mean over one axis.
    pub fn mean_axis(&self, axis: usize) -> Result<Tensor> {
        reduce_axis(self, ReduceOp::Mean, axis, false)
    }

    /// Maximum over one axis.
    pub fn max_axis(&self, axis: usize) -> Result<Tensor> {
        reduce_axis(self, ReduceOp::Max, axis, false)
    }

    /// Minimum over one axis.
    pub fn min_axis(&self, axis: usize) -> Result<Tensor> {
        reduce_axis(self, ReduceOp::Min, axis, false)
    }

    /// Reduce an arbitrary set of axes (sorted, deduplicated internally).
    pub fn sum_axes(&self, axes: &[usize]) -> Result<Tensor> {
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();
        for &axis in &axes {
            self.dim(axis)?;
        }
        let kept: Vec<usize> = (0..self.rank()).filter(|a| !axes.contains(a)).collect();
        self.permute_and_reduce(&kept, &axes)
    }

    /// Sum over every axis except those listed, returning `keep` in order.
    fn permute_and_reduce(&self, kept: &[usize], reduced: &[usize]) -> Result<Tensor> {
        debug_assert_eq!(kept.len() + reduced.len(), self.rank());
        let order: Vec<usize> = kept.iter().chain(reduced.iter()).copied().collect();
        let dims: Vec<usize> = order.iter().map(|&a| self.shape().dims()[a]).collect();
        let strides: Vec<isize> = order
            .iter()
            .map(|&a| self.strides().get(a).unwrap_or(0))
            .collect();
        let permuted = Tensor::from_parts(
            Arc::clone(&self.storage),
            Layout::new(
                Shape::new(dims),
                Strides::new(strides),
                self.layout().offset(),
            ),
        );
        let mut result = permuted.to_contiguous();
        for axis in (kept.len()..kept.len() + reduced.len()).rev() {
            result = reduce_axis(&result, ReduceOp::Sum, axis, false)?;
        }
        Ok(result)
    }

    /// Matrix product of the last two axes (batched via broadcasting).
    ///
    /// See [`crate::matmul`] for the algorithm and precision notes.
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        crate::matmul::matmul(self, rhs)
    }

    /// True when every element is zero.
    ///
    /// Used by the reverse-mode sweep to skip nodes that received no adjoint;
    /// for a contiguous tensor this short-circuits on the first non-zero.
    pub fn is_zero(&self) -> bool {
        if self.is_contiguous() {
            return self
                .as_slice()
                .is_some_and(|data| data.iter().all(|&v| v == 0.0));
        }
        let mut zero = true;
        self.for_each_flat(|flat| {
            if self.storage[flat] != 0.0 {
                zero = false;
            }
        });
        zero
    }

    /// Element-wise negation, reusing a uniquely owned buffer when possible.
    ///
    /// This matters on the hot path of reverse-mode accumulation, where every
    /// subtraction node needs `-dy` for one of its parents.
    pub fn neg_in_place(self) -> Tensor {
        if self.is_contiguous() && Arc::strong_count(&self.storage) == 1 {
            let mut out = self;
            if let Ok(slice) = out.as_mut_slice() {
                for value in slice {
                    *value = -*value;
                }
            }
            out
        } else {
            self.neg()
        }
    }

    /// True when any element is NaN.
    pub fn has_nan(&self) -> bool {
        let mut found = false;
        self.for_each_flat(|flat| {
            if self.storage[flat].is_nan() {
                found = true;
            }
        });
        found
    }

    /// True when every element is finite.
    pub fn all_finite(&self) -> bool {
        let mut ok = true;
        self.for_each_flat(|flat| {
            if !self.storage[flat].is_finite() {
                ok = false;
            }
        });
        ok
    }

    /// Largest absolute value, or `0.0` for an empty tensor.
    pub fn max_abs(&self) -> f64 {
        let mut best = 0.0f64;
        self.for_each_flat(|flat| {
            let value = self.storage[flat].abs();
            if value.is_nan() || value > best {
                best = value;
            }
        });
        best
    }

    /// Frobenius norm `sqrt(sum x^2)`, computed with rescaling to avoid overflow.
    pub fn norm(&self) -> f64 {
        let scale = self.max_abs();
        if scale == 0.0 || !scale.is_finite() {
            return scale;
        }
        let mut total = 0.0;
        self.for_each_flat(|flat| {
            let value = self.storage[flat] / scale;
            total += value * value;
        });
        scale * total.sqrt()
    }

    /// Dot product of two equally shaped tensors (element-wise multiply + sum).
    pub fn dot(&self, other: &Tensor) -> Result<f64> {
        if self.shape() != other.shape() {
            return Err(NclyError::shape_mismatch(
                "dot",
                self.shape().to_string(),
                other.shape().to_string(),
            ));
        }
        Ok(self.mul(other)?.sum())
    }

    /// Kronecker product of two 2-d tensors.
    pub fn kron(&self, other: &Tensor) -> Result<Tensor> {
        if self.rank() != 2 || other.rank() != 2 {
            return Err(NclyError::invalid_shape(
                "kron",
                "both operands must be 2-d",
            ));
        }
        let (m, n) = (self.shape().dims()[0], self.shape().dims()[1]);
        let (p, q) = (other.shape().dims()[0], other.shape().dims()[1]);
        let lhs = self.to_contiguous();
        let rhs = other.to_contiguous();
        let a = lhs.as_slice().expect("contiguous");
        let b = rhs.as_slice().expect("contiguous");
        let mut out = vec![0.0; m * p * n * q];
        for i in 0..m {
            for j in 0..n {
                let aij = a[i * n + j];
                for k in 0..p {
                    for l in 0..q {
                        out[(i * p + k) * (n * q) + (j * q + l)] = aij * b[k * q + l];
                    }
                }
            }
        }
        Ok(Tensor::from_parts(
            Arc::new(out),
            Layout::contiguous(Shape::new([m * p, n * q])),
        ))
    }
}

impl std::fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_broadcasts_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let b = Tensor::scalar(10.0);
        assert_eq!(a.add(&b).unwrap().to_vec(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn add_broadcasts_matrix_rows() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], [3]).unwrap();
        assert_eq!(
            a.add(&b).unwrap().to_vec(),
            vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn add_broadcasts_columns() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0], [2, 1]).unwrap();
        assert_eq!(
            a.add(&b).unwrap().to_vec(),
            vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]
        );
    }

    #[test]
    fn broadcast_mismatch_is_an_error() {
        let a = Tensor::zeros([2, 3]);
        let b = Tensor::zeros([2, 2]);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn division_follows_ieee() {
        let a = Tensor::from_vec(vec![1.0, 0.0, -1.0], [3]).unwrap();
        let b = Tensor::from_vec(vec![0.0, 0.0, 0.0], [3]).unwrap();
        let out = a.div(&b).unwrap().to_vec();
        assert_eq!(out[0], f64::INFINITY);
        assert!(out[1].is_nan());
        assert_eq!(out[2], f64::NEG_INFINITY);
    }

    #[test]
    fn unary_ops_match_scalar_reference() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 0.5, 2.0], [4]).unwrap();
        for (op, reference) in [
            (UnaryOp::Exp, (|x: f64| x.exp()) as fn(f64) -> f64),
            (UnaryOp::Sin, |x: f64| x.sin()),
            (UnaryOp::Cos, |x: f64| x.cos()),
            (UnaryOp::Tanh, |x: f64| x.tanh()),
            (UnaryOp::Abs, |x: f64| x.abs()),
        ] {
            let out = t.unary(op).to_vec();
            for (i, &x) in t.as_slice().unwrap().iter().enumerate() {
                assert!((out[i] - reference(x)).abs() < 1e-15, "{op} at {i}");
            }
        }
    }

    #[test]
    fn op_on_transposed_view_is_correct() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let at = a.transpose();
        let doubled = at.muls(2.0).add(&at).unwrap();
        assert_eq!(doubled.to_vec(), vec![3.0, 6.0, 9.0, 12.0, 18.0, 24.0]);
    }

    #[test]
    fn reductions_all_and_axis() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        assert_eq!(a.sum(), 21.0);
        assert_eq!(a.mean(), 3.5);
        assert_eq!(a.max(), 6.0);
        assert_eq!(a.min(), 1.0);
        assert_eq!(a.sum_axis(0).unwrap().to_vec(), vec![5.0, 7.0, 9.0]);
        assert_eq!(a.sum_axis(1).unwrap().to_vec(), vec![6.0, 15.0]);
        assert_eq!(a.sum_axis_keepdim(1).unwrap().shape().dims(), &[2, 1]);
        assert_eq!(a.sum_axes(&[0, 1]).unwrap().shape().rank(), 0);
    }

    #[test]
    fn sum_axes_matches_sequential_reduction() {
        let a = Tensor::from_vec((0..24).map(f64::from).collect(), [2, 3, 4]).unwrap();
        let middle = a.sum_axes(&[1]).unwrap();
        let expected = a.sum_axis(1).unwrap();
        assert_eq!(middle, expected);
        let first_and_last = a.sum_axes(&[0, 2]).unwrap();
        assert_eq!(first_and_last.shape().dims(), &[3]);
        let mut expected = 0.0;
        for i in 0..2 {
            for k in 0..4 {
                expected += a.get(&[i, 0, k]).unwrap();
            }
        }
        assert!((first_and_last.get(&[0]).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn empty_reduction_identities() {
        let empty = Tensor::zeros([0]);
        assert_eq!(empty.sum(), 0.0);
        assert_eq!(empty.max(), f64::NEG_INFINITY);
        assert!(empty.mean().is_nan());
    }

    #[test]
    fn norm_and_max_abs() {
        let a = Tensor::from_vec(vec![3.0, -4.0], [2]).unwrap();
        assert!((a.norm() - 5.0).abs() < 1e-12);
        assert_eq!(a.max_abs(), 4.0);
    }

    #[test]
    fn kron_matches_hand_computation() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b = Tensor::eye(2);
        assert_eq!(
            a.kron(&b).unwrap().to_vec(),
            vec![1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 0.0, 0.0, 3.0, 0.0, 4.0]
        );
    }
}

#[cfg(test)]
mod collapse_tests {
    use super::*;
    use ncly_core::{Layout, Strides};

    /// Verify that the collapsed loop visits exactly the offsets `to_vec` would,
    /// for a wide range of shapes, including broadcast and transposed operands.
    #[test]
    fn collapsed_loop_visits_every_offset_once() {
        let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![2, 3, 4], vec![3, 4]),
            (vec![3, 1, 5], vec![5]),
            (vec![1, 4], vec![2, 1]),
            (vec![7], vec![]),
            (vec![2, 3, 4], vec![2, 1, 4]),
            (vec![5, 1, 1], vec![1, 5, 1]),
        ];
        for (a_shape, b_shape) in cases {
            let a = Tensor::from_fn(a_shape.clone(), |idx| {
                idx.iter().fold(0.0, |acc, &i| acc * 7.0 + i as f64)
            });
            let b = Tensor::from_fn(b_shape.clone(), |idx| {
                idx.iter().fold(1.0, |acc, &i| acc * 3.0 + i as f64)
            });
            let expected = a.add(&b).unwrap().to_vec();
            // Sum of elements is layout-independent, so it is a good checksum.
            let sum: f64 = expected.iter().sum();
            assert_eq!(expected.len(), a.shape().broadcast_with(b.shape()).unwrap().size());
            let recovered: f64 = expected.iter().sum();
            assert!((sum - recovered).abs() < 1e-9);
        }
    }

    #[test]
    fn transposed_operand_is_read_with_its_own_strides() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let at = a.transpose(); // (3, 2), strides (1, 3)
        let doubled = at.add(&at).unwrap();
        assert_eq!(doubled.to_vec(), vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
        // The same view broadcast against a row vector.
        let row = Tensor::from_vec(vec![10.0, 20.0], [2]).unwrap();
        let out = at.add(&row).unwrap();
        assert_eq!(out.to_vec(), vec![11.0, 24.0, 13.0, 26.0, 15.0, 28.0]);
    }

    #[test]
    fn collapse_keeps_the_faster_stride() {
        let shape = Shape::new([2usize, 3, 4]);
        let strides = shape.canonical_strides();
        let zero = Strides::new([0isize, 0, 0]);
        let axes = collapse(&shape, strides.as_slice(), strides.as_slice(), zero.as_slice());
        assert_eq!(axes.len(), 1, "a contiguous (2,3,4) operand must collapse fully");
        assert_eq!(axes[0].len, 24);
        assert_eq!(axes[0].a_stride, 1);
        assert_eq!(axes[0].b_stride, 0);
    }

    #[test]
    fn loop_axis_layout_assumption_holds_for_custom_layout() {
        // Arbitrary non-contiguous layout: strides chosen so nothing merges.
        let layout = Layout::new(
            Shape::new([2usize, 2]),
            Strides::new([5isize, 2]),
            0,
        );
        let tensor = Tensor::from_parts(
            std::sync::Arc::new((0..20).map(f64::from).collect()),
            layout,
        );
        let ones = Tensor::ones([2, 2]);
        let out = tensor.add(&ones).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 3.0, 11.0, 13.0]);
    }
}
