//! LU decomposition with partial pivoting.
//!
//! For a square `A` we compute `P·A = L·U` where `P` is a row permutation
//! recorded as swaps, `L` is unit lower triangular and `U` is upper triangular.
//! The factors are stored compactly: `L`'s unit diagonal is implicit and `U`'s
//! diagonal is stored, so the factor matrix needs exactly `n²` values.
//!
//! # Algorithm
//!
//! Right-looking elimination with row interchanges:
//!
//! ```text
//! for k in 0..n:
//!     pivot = argmax_{i >= k} |A[i][k]|
//!     reject if |A[pivot][k]| <= tol * ||A||_inf
//!     swap rows k and pivot
//!     for i in k+1..n:
//!         A[i][k] /= A[k][k]                     (stores L)
//!         A[i][k+1..n] -= A[i][k] * A[k][k+1..n] (updates U)
//! ```
//!
//! The rows below the pivot are updated with an AXPY, which is the operation
//! that vectorises best on modern CPUs.
//!
//! # Complexity
//!
//! `2n³/3` flops for the factorisation and `2n²` for a solve.

use crate::{check_dims, matrix_norm_inf};
use ncly_core::scalar::{is_negligible, DEFAULT_PIVOT_TOLERANCE};
use ncly_core::{NclyError, Result};
use std::fmt;

/// An `LU` factorisation of a square matrix.
#[derive(Debug, Clone)]
pub struct Lu {
    /// Combined `L` (below the diagonal) and `U` (on and above it).
    factors: Vec<f64>,
    /// Row swap sequence: step `k` exchanged rows `k` and `pivots[k]`.
    pivots: Vec<usize>,
    /// Number of interchanges, used for the determinant sign.
    swaps: usize,
    size: usize,
    /// Absolute tolerance used for pivot rejection.
    tolerance: f64,
}

impl Lu {
    /// Factorise a row-major `n × n` matrix.
    pub fn factor(data: &[f64], n: usize) -> Result<Self> {
        Self::factor_with_tolerance(data, n, DEFAULT_PIVOT_TOLERANCE)
    }

    /// Factorise with an explicit relative pivot tolerance.
    pub fn factor_with_tolerance(data: &[f64], n: usize, tolerance: f64) -> Result<Self> {
        check_dims("lu", data, n, n)?;
        let scale = matrix_norm_inf(data, n);
        let mut factors = data.to_vec();
        let mut pivots = Vec::with_capacity(n);
        let mut swaps = 0usize;

        for k in 0..n {
            // Partial pivoting: pick the largest magnitude in column k.
            let mut best = k;
            let mut best_value = factors[k * n + k].abs();
            for i in k + 1..n {
                let candidate = factors[i * n + k].abs();
                if candidate > best_value {
                    best_value = candidate;
                    best = i;
                }
            }
            if is_negligible(best_value, scale, tolerance) {
                return Err(NclyError::Singular { op: "lu", pivot: k });
            }
            if best != k {
                for j in 0..n {
                    factors.swap(k * n + j, best * n + j);
                }
                swaps += 1;
            }
            pivots.push(best);

            let pivot = factors[k * n + k];
            for i in k + 1..n {
                let multiplier = factors[i * n + k] / pivot;
                factors[i * n + k] = multiplier;
                if multiplier == 0.0 {
                    continue;
                }
                for j in k + 1..n {
                    factors[i * n + j] -= multiplier * factors[k * n + j];
                }
            }
        }

        Ok(Self {
            factors,
            pivots,
            swaps,
            size: n,
            tolerance,
        })
    }

    /// Dimension of the factorised matrix.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The relative pivot tolerance in force.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// The compact factor matrix (`L` below, `U` on/above the diagonal).
    pub fn factors(&self) -> &[f64] {
        &self.factors
    }

    /// Row swaps applied, in order.
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    /// `L` as a dense unit lower triangular matrix.
    pub fn lower(&self) -> Vec<f64> {
        let n = self.size;
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            out[i * n + i] = 1.0;
            for j in 0..i {
                out[i * n + j] = self.factors[i * n + j];
            }
        }
        out
    }

    /// `U` as a dense upper triangular matrix.
    pub fn upper(&self) -> Vec<f64> {
        let n = self.size;
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                out[i * n + j] = self.factors[i * n + j];
            }
        }
        out
    }

    /// The permutation matrix `P` such that `P·A = L·U`, as a dense matrix.
    pub fn permutation(&self) -> Vec<f64> {
        let n = self.size;
        // Apply the swaps to the identity in the recorded order.
        let mut rows: Vec<usize> = (0..n).collect();
        for (k, &pivot) in self.pivots.iter().enumerate() {
            rows.swap(k, pivot);
        }
        let mut out = vec![0.0; n * n];
        for (position, &source) in rows.iter().enumerate() {
            out[position * n + source] = 1.0;
        }
        out
    }

    /// Solve `A·x = b`.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        let n = self.size;
        if rhs.len() != n {
            return Err(NclyError::invalid_shape(
                "lu::solve",
                format!("right-hand side has {} entries, expected {n}", rhs.len()),
            ));
        }
        let mut x = rhs.to_vec();
        // Apply P.
        for (k, &pivot) in self.pivots.iter().enumerate() {
            x.swap(k, pivot);
        }
        // Forward substitution with unit diagonal.
        for i in 0..n {
            let mut total = x[i];
            for j in 0..i {
                total -= self.factors[i * n + j] * x[j];
            }
            x[i] = total;
        }
        // Back substitution.
        for i in (0..n).rev() {
            let mut total = x[i];
            for j in i + 1..n {
                total -= self.factors[i * n + j] * x[j];
            }
            x[i] = total / self.factors[i * n + i];
        }
        Ok(x)
    }

    /// Solve for several right-hand sides given as columns of a dense
    /// `n × k` row-major matrix.
    pub fn solve_matrix(&self, rhs: &[f64], columns: usize) -> Result<Vec<f64>> {
        let n = self.size;
        check_dims("lu::solve_matrix", rhs, n, columns)?;
        let mut out = vec![0.0; n * columns];
        let mut column = vec![0.0; n];
        for c in 0..columns {
            for i in 0..n {
                column[i] = rhs[i * columns + c];
            }
            let solved = self.solve(&column)?;
            for i in 0..n {
                out[i * columns + c] = solved[i];
            }
        }
        Ok(out)
    }

    /// Determinant, accumulated in log space internally.
    pub fn determinant(&self) -> f64 {
        let n = self.size;
        let mut sign = if self.swaps.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let mut log_abs = 0.0;
        for i in 0..n {
            let diagonal = self.factors[i * n + i];
            if diagonal == 0.0 {
                return 0.0;
            }
            if diagonal < 0.0 {
                sign = -sign;
            }
            log_abs += diagonal.abs().ln();
        }
        sign * log_abs.exp()
    }

    /// `log|det A|` and the sign of `det A`.
    pub fn log_determinant(&self) -> (f64, f64) {
        let n = self.size;
        let mut sign = if self.swaps.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let mut log_abs = 0.0;
        for i in 0..n {
            let diagonal = self.factors[i * n + i];
            if diagonal < 0.0 {
                sign = -sign;
            }
            log_abs += diagonal.abs().ln();
        }
        (log_abs, sign)
    }

    /// Inverse, computed by solving `A·X = I`.
    pub fn inverse(&self) -> Result<Vec<f64>> {
        let n = self.size;
        let mut identity = vec![0.0; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        self.solve_matrix(&identity, n)
    }
}

impl fmt::Display for Lu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lu({}x{})", self.size, self.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut total = 0.0;
                for k in 0..n {
                    total += a[i * n + k] * b[k * n + j];
                }
                out[i * n + j] = total;
            }
        }
        out
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn factors_reconstruct_the_matrix() {
        for n in 1..8 {
            let a: Vec<f64> = (0..n * n)
                .map(|i| {
                    ((i * 7 + 3) % 11) as f64 - 4.0 + if i % (n + 1) == 0 { 10.0 } else { 0.0 }
                })
                .collect();
            let lu = Lu::factor(&a, n).unwrap();
            let lu_product = multiply(&lu.lower(), &lu.upper(), n);
            let permutation = lu.permutation();
            let pa = multiply(&permutation, &a, n);
            assert!(
                max_abs_diff(&lu_product, &pa) < 1e-10,
                "n={n}: reconstruction failed"
            );
        }
    }

    #[test]
    fn solves_linear_systems_to_machine_precision() {
        let n = 6;
        let a: Vec<f64> = (0..n * n)
            .map(|i| ((i * 13 + 5) % 9) as f64 + if i / n == i % n { 9.0 } else { 0.0 })
            .collect();
        let x_exact: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i * n + j] * x_exact[j];
            }
        }
        let lu = Lu::factor(&a, n).unwrap();
        let x = lu.solve(&b).unwrap();
        assert!(max_abs_diff(&x, &x_exact) < 1e-10);
    }

    #[test]
    fn determinant_matches_closed_form() {
        let a = [4.0, 3.0, 6.0, 3.0];
        let lu = Lu::factor(&a, 2).unwrap();
        assert!((lu.determinant() - (4.0 * 3.0 - 3.0 * 6.0)).abs() < 1e-12);
        // A swap flips the sign: [[0,1],[1,0]] has determinant -1.
        let swap = [0.0, 1.0, 1.0, 0.0];
        let lu = Lu::factor(&swap, 2).unwrap();
        assert!((lu.determinant() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn singular_matrices_are_detected() {
        let a = [1.0, 2.0, 2.0, 4.0];
        let err = Lu::factor(&a, 2).unwrap_err();
        assert!(matches!(err, NclyError::Singular { pivot: 1, .. }));
    }

    #[test]
    fn tiny_but_well_conditioned_is_not_singular() {
        // Scaled identity: determinant is tiny but the matrix is perfectly
        // conditioned, so the relative pivot test must accept it.
        let a = [1e-300, 0.0, 0.0, 1e-300];
        let lu = Lu::factor(&a, 2).expect("scaled identity must factor");
        let (log_abs, _) = lu.log_determinant();
        assert!((log_abs - 2.0 * 1e-300f64.ln()).abs() < 1e-6);
        let x = lu.solve(&[1e-300, 2e-300]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn inverse_inverts() {
        let n = 4;
        let a: Vec<f64> = (0..n * n)
            .map(|i| ((i * 5 + 1) % 7) as f64 + if i / n == i % n { 8.0 } else { 0.0 })
            .collect();
        let lu = Lu::factor(&a, n).unwrap();
        let inv = lu.inverse().unwrap();
        let product = multiply(&a, &inv, n);
        let mut identity = vec![0.0; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        assert!(max_abs_diff(&product, &identity) < 1e-10);
    }

    #[test]
    fn dimension_errors() {
        assert!(Lu::factor(&[1.0, 2.0, 3.0], 2).is_err());
        let lu = Lu::factor(&[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        assert!(lu.solve(&[1.0]).is_err());
    }
}
