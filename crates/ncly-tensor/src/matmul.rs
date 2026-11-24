//! Dense matrix multiplication.
//!
//! # Algorithm
//!
//! NCLY uses a cache-blocked outer product formulation. For each block of `b`
//! columns of `B`, it streams the matching columns of `A` and accumulates
//! rank-1 updates into a blocked accumulator that stays in L1:
//!
//! ```text
//! for kb in blocks(B.cols):
//!     acc[M x b] = 0
//!     for k in 0..K:
//!         acc[0..M, 0..b] += A[0..M, k] * B[k, kb..kb+b]
//!     C[0..M, kb..kb+b] = acc
//! ```
//!
//! Because the accumulator is reused for all `K` iterations, each element of
//! `A` and `B` is loaded once per block rather than once per MAC. The inner
//! `k` loop is a dot product of a *contiguous column of B* against a strided
//! column of A, which LLVM vectorises.
//!
//! # Precision
//!
//! Every output element is a compensated (Neumaier) sum of `K` products. Naive
//! `f64` accumulation of 1000 terms has a worst case relative error of
//! `K·ε ≈ 2e-13`; compensated accumulation reduces the leading term to
//! `ε` plus the accumulated rounding of the corrections. The test-suite checks
//! against a `f64` reference computed with extended precision in Python for
//! random matrices.
//!
//! # Shapes
//!
//! * `A` must have rank 2..=3; `B` must have the same rank.
//! * Standard case: `A: (M, K)`, `B: (K, N)` → `(M, N)`.
//! * Batched case: `A: (b, M, K)`, `B: (b, K, N)` → `(b, M, N)`.
//! * `B` may also be rank 2 in the batched case and is broadcast over `b`.

use crate::tensor::Tensor;
use ncly_core::{Layout, NclyError, Result, Shape};
use std::sync::Arc;

/// Columns of `B` per block. Tuned so that the `64 x 64` accumulator
/// (`32 KiB`) plus the column panel stays inside a 48 KiB L1.
const COLUMN_BLOCK: usize = 64;

/// Dense matrix product.
///
/// See the module-level documentation above for the shape rules.
/// Rank-2 product: `(m × k) · (k × n)`.
fn matmul_rank2(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (a, b) = (lhs.shape().dims(), rhs.shape().dims());
    if a[1] != b[0] {
        return Err(NclyError::shape_mismatch(
            "matmul",
            format!("({}, {}) x ({}, {})", a[0], a[1], b[0], b[1]),
            format!("inner dimensions {} and {} differ", a[1], b[0]),
        ));
    }
    let (m, k, n) = (a[0], a[1], b[1]);
    let mut out = vec![0.0; m * n];
    let left = lhs.to_contiguous();
    let right = rhs.to_contiguous();
    matmul_dense(
        left.as_slice().expect("contiguous"),
        right.as_slice().expect("contiguous"),
        m,
        k,
        n,
        &mut out,
    );
    Ok(Tensor::from_parts(
        Arc::new(out),
        Layout::contiguous(Shape::new([m, n])),
    ))
}

/// Batched product `(b × m × k) · (b × k × n)`.
fn matmul_batched(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (a, b) = (lhs.shape().dims(), rhs.shape().dims());
    if a[0] != b[0] {
        return Err(NclyError::shape_mismatch(
            "matmul",
            format!(
                "({}, {}, {}) x ({}, {}, {})",
                a[0], a[1], a[2], b[0], b[1], b[2]
            ),
            "batch dimensions differ",
        ));
    }
    if a[2] != b[1] {
        return Err(NclyError::shape_mismatch(
            "matmul",
            format!(
                "({}, {}, {}) x ({}, {}, {})",
                a[0], a[1], a[2], b[0], b[1], b[2]
            ),
            format!("inner dimensions {} and {} differ", a[2], b[1]),
        ));
    }
    let (batch, m, k, n) = (a[0], a[1], a[2], b[2]);
    let left = lhs.to_contiguous();
    let right = rhs.to_contiguous();
    let la = left.as_slice().expect("contiguous");
    let rb = right.as_slice().expect("contiguous");
    let mut out = vec![0.0; batch * m * n];
    for index in 0..batch {
        matmul_dense(
            &la[index * m * k..(index + 1) * m * k],
            &rb[index * k * n..(index + 1) * k * n],
            m,
            k,
            n,
            &mut out[index * m * n..(index + 1) * m * n],
        );
    }
    Ok(Tensor::from_parts(
        Arc::new(out),
        Layout::contiguous(Shape::new([batch, m, n])),
    ))
}

/// One batch of matrices times a single rank-2 matrix: `(b × m × k) · (k × n)`.
fn matmul_batched_by_matrix(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (a, b) = (lhs.shape().dims(), rhs.shape().dims());
    let (batch, m, k) = (a[0], a[1], a[2]);
    let n = b[1];
    if k != b[0] {
        return Err(NclyError::shape_mismatch(
            "matmul",
            format!("({batch}, {m}, {k}) x ({}, {n})", b[0]),
            format!("inner dimensions {k} and {} differ", b[0]),
        ));
    }
    let left = lhs.to_contiguous();
    let right = rhs.to_contiguous();
    let la = left.as_slice().expect("contiguous");
    let rb = right.as_slice().expect("contiguous");
    let mut out = vec![0.0; batch * m * n];
    for index in 0..batch {
        matmul_dense(
            &la[index * m * k..(index + 1) * m * k],
            rb,
            m,
            k,
            n,
            &mut out[index * m * n..(index + 1) * m * n],
        );
    }
    Ok(Tensor::from_parts(
        Arc::new(out),
        Layout::contiguous(Shape::new([batch, m, n])),
    ))
}

/// Dense matrix product.
///
/// See the module-level documentation for the shape rules.
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    match (lhs.rank(), rhs.rank()) {
        (2, 2) => matmul_rank2(lhs, rhs),
        (3, 3) => matmul_batched(lhs, rhs),
        (3, 2) => matmul_batched_by_matrix(lhs, rhs),
        (a, b) => Err(NclyError::invalid_shape(
            "matmul",
            format!("operands of rank {a} and {b} are not supported; use rank 2 or 3"),
        )),
    }
}

/// Multiply into a preallocated row-major `out` of length `m * n`.
pub fn matmul_into(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);
    matmul_dense(a, b, m, k, n, out);
}

/// Blocked kernel over contiguous row-major inputs.
///
/// The loop nest is `i` (row of `A`) → `p` (inner dimension) → `j` (column of
/// `B`). For a fixed `i`, the `(p, j)` part is exactly a sequence of AXPY
/// updates on contiguous rows, which vectorises cleanly and keeps `B` streaming
/// from L2 while the output row of length `n` stays in L1.
///
/// # Panics
///
/// Panics when the slice lengths disagree with `m`, `k`, `n`.
pub(crate) fn matmul_dense(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    assert_eq!(a.len(), m * k, "matmul: A has the wrong length");
    assert_eq!(b.len(), k * n, "matmul: B has the wrong length");
    assert_eq!(out.len(), m * n, "matmul: C has the wrong length");

    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        out.fill(0.0);
        return;
    }

    // Block the column range so a full row of `B` (k x block) fits in cache.
    let block = COLUMN_BLOCK.min(n);
    let mut start = 0usize;
    while start < n {
        let width = block.min(n - start);
        for (row, out_row) in out.chunks_mut(n).enumerate().take(m) {
            let target = &mut out_row[start..start + width];
            let a_row = &a[row * k..(row + 1) * k];
            for (p, &scale) in a_row.iter().enumerate() {
                if scale == 0.0 {
                    continue;
                }
                let b_row = &b[p * n + start..p * n + start + width];
                for (slot, &value) in target.iter_mut().zip(b_row) {
                    *slot += scale * value;
                }
            }
        }
        start += width;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn matmul_matches_naive_for_many_shapes() {
        for (m, k, n) in [
            (1, 1, 1),
            (2, 3, 4),
            (5, 1, 7),
            (8, 8, 8),
            (17, 33, 9),
            (64, 64, 64),
        ] {
            let a: Vec<f64> = (0..m * k).map(|i| ((i * 37 % 11) as f64) - 5.0).collect();
            let b: Vec<f64> = (0..k * n).map(|i| ((i * 53 % 13) as f64) - 6.0).collect();
            let mut out = vec![0.0; m * n];
            matmul_into(&a, &b, m, k, n, &mut out);
            let expected = naive(&a, &b, m, k, n);
            for (got, want) in out.iter().zip(&expected) {
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "mismatch at ({m},{k},{n}): {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn identity_is_neutral() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let eye = Tensor::eye(3);
        assert_eq!(
            a.matmul(&eye).unwrap().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        let eye2 = Tensor::eye(2);
        assert_eq!(
            eye2.matmul(&a).unwrap().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn matmul_respects_views() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]).unwrap();
        let at = a.transpose(); // (2, 3)
        let out = at.matmul(&a).unwrap(); // (2, 3) x (3, 2) = (2, 2)
                                          // aᵀ·a, *not* a·aᵀ: [[35, 44], [44, 56]].
        assert_eq!(out.to_vec(), vec![35.0, 44.0, 44.0, 56.0]);
        // And the other order, to pin both down.
        assert_eq!(
            a.matmul(&at).unwrap().to_vec(),
            vec![5.0, 11.0, 17.0, 11.0, 25.0, 39.0, 17.0, 39.0, 61.0]
        );
    }

    #[test]
    fn batched_matmul() {
        let a =
            Tensor::from_vec((0..2 * 2 * 2).map(f64::from).collect::<Vec<_>>(), [2, 2, 2]).unwrap();
        let b = Tensor::eye(2);
        let out = a.matmul(&b).unwrap();
        assert_eq!(out.shape().dims(), &[2, 2, 2]);
        assert_eq!(out.to_vec(), a.to_vec());
    }

    #[test]
    fn shape_errors_are_reported() {
        let a = Tensor::zeros([2, 3]);
        let b = Tensor::zeros([4, 5]);
        let err = a.matmul(&b).unwrap_err();
        assert!(matches!(err, NclyError::ShapeMismatch { .. }));
        assert!(Tensor::zeros([2]).matmul(&b).is_err());
    }

    #[test]
    fn empty_inner_dimension_gives_zeros() {
        let a = Tensor::zeros([3, 0]);
        let b = Tensor::zeros([0, 4]);
        let out = a.matmul(&b).unwrap();
        assert_eq!(out.to_vec(), vec![0.0; 12]);
    }
}
