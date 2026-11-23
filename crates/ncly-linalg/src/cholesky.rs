//! Cholesky factorisation for symmetric positive definite matrices.
//!
//! For spd `A` we compute the lower triangular `L` with `A = L·Lᵀ`. This is
//! roughly twice as fast as LU and, unlike LU, it *detects* indefiniteness: a
//! non-positive pivot proves `A` is not positive definite.
//!
//! # Algorithm
//!
//! ```text
//! for j in 0..n:
//!     L[j][j] = sqrt(A[j][j] - Σ_{k<j} L[j][k]²)
//!     for i in j+1..n:
//!         L[i][j] = (A[i][j] - Σ_{k<j} L[i][k] L[j][k]) / L[j][j]
//! ```

use crate::check_dims;
use ncly_core::scalar::DEFAULT_PIVOT_TOLERANCE;
use ncly_core::{NclyError, Result};

/// A Cholesky factorisation `A = L·Lᵀ`.
#[derive(Debug, Clone)]
pub struct Cholesky {
    lower: Vec<f64>,
    size: usize,
    tolerance: f64,
}

impl Cholesky {
    /// Factorise a symmetric positive definite row-major matrix.
    pub fn factor(data: &[f64], n: usize) -> Result<Self> {
        Self::factor_with_tolerance(data, n, DEFAULT_PIVOT_TOLERANCE)
    }

    /// Factorise with an explicit relative tolerance.
    pub fn factor_with_tolerance(data: &[f64], n: usize, tolerance: f64) -> Result<Self> {
        check_dims("cholesky", data, n, n)?;
        // Scale-tolerant test: the diagonal pivots of a matrix scaled by `s`
        // shrink by `s`, so compare against `s * trace / n`.
        let scale = {
            let trace: f64 = (0..n).map(|i| data[i * n + i].abs()).sum();
            if n == 0 {
                0.0
            } else {
                trace / n as f64
            }
        };
        let mut lower = vec![0.0f64; n * n];
        for j in 0..n {
            let mut sum = data[j * n + j];
            for k in 0..j {
                sum -= lower[j * n + k] * lower[j * n + k];
            }
            if sum <= tolerance * scale.max(f64::MIN_POSITIVE) || sum <= 0.0 {
                return Err(NclyError::Singular {
                    op: "cholesky",
                    pivot: j,
                });
            }
            let diagonal = sum.sqrt();
            lower[j * n + j] = diagonal;
            for i in j + 1..n {
                let mut value = data[i * n + j];
                for k in 0..j {
                    value -= lower[i * n + k] * lower[j * n + k];
                }
                lower[i * n + j] = value / diagonal;
            }
        }
        Ok(Self {
            lower,
            size: n,
            tolerance,
        })
    }

    /// The lower triangular factor as a dense row-major matrix.
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Dimension.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The relative tolerance in force.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Solve `A·x = b` using the two triangular systems.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        let n = self.size;
        if rhs.len() != n {
            return Err(NclyError::invalid_shape(
                "cholesky::solve",
                format!("right-hand side has {} entries, expected {n}", rhs.len()),
            ));
        }
        let mut y = rhs.to_vec();
        for i in 0..n {
            let mut total = y[i];
            for j in 0..i {
                total -= self.lower[i * n + j] * y[j];
            }
            y[i] = total / self.lower[i * n + i];
        }
        for i in (0..n).rev() {
            let mut total = y[i];
            for j in i + 1..n {
                total -= self.lower[j * n + i] * y[j];
            }
            y[i] = total / self.lower[i * n + i];
        }
        Ok(y)
    }

    /// `ln det A = 2 Σ ln L[i][i]`.
    pub fn log_determinant(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.size {
            total += self.lower[i * self.size + i].ln();
        }
        2.0 * total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_positive_definite_matrix() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        let a = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let n = 3;
        let chol = Cholesky::factor(&a, n).unwrap();
        let l = chol.lower();
        let mut product = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut total = 0.0;
                for k in 0..n {
                    total += l[i * n + k] * l[j * n + k];
                }
                product[i * n + j] = total;
            }
        }
        for (got, want) in product.iter().zip(&a) {
            assert!((got - want).abs() < 1e-10, "{got} vs {want}");
        }
    }

    #[test]
    fn solves_system() {
        let a = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let n = 3;
        let x_true = [1.0, 2.0, 3.0];
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i * n + j] * x_true[j];
            }
        }
        let x = Cholesky::factor(&a, n).unwrap().solve(&b).unwrap();
        for (got, want) in x.iter().zip(&x_true) {
            assert!((got - want).abs() < 1e-10);
        }
    }

    #[test]
    fn rejects_indefinite_matrices() {
        let a = [1.0, 2.0, 2.0, 1.0]; // eigenvalues 3 and -1
        let err = Cholesky::factor(&a, 2).unwrap_err();
        assert!(matches!(err, NclyError::Singular { pivot: 1, .. }));
    }

    #[test]
    fn determinant_matches_direct_product() {
        let a = [4.0, 0.0, 0.0, 9.0];
        let chol = Cholesky::factor(&a, 2).unwrap();
        assert!((chol.log_determinant() - 36.0f64.ln()).abs() < 1e-12);
    }
}
