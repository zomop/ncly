//! QR decomposition via Householder reflectors.
//!
//! For an `m × n` matrix (`m ≥ n`) we compute `A = Q·R` with `Q` orthogonal and
//! `R` upper triangular. Choosing reflectors over Gram–Schmidt matters: classical
//! Gram–Schmidt loses orthogonality as `κ(A)` grows, whereas Householder
//! reflection is backward stable for any conditioning.
//!
//! # Algorithm
//!
//! At step `k` we build `H_k = I - 2vvᵀ` that maps the trailing column onto a
//! multiple of `e_1`. Reflectors are accumulated so `Q = H_1 H_2 … H_n` without
//! ever forming dense intermediates:
//!
//! ```text
//! R = A
//! for k in 0..n:
//!     v = householder(R[k.., k])
//!     R[k.., k..] -= 2 v (vᵀ R[k.., k..])
//! ```
//!
//! # Complexity
//!
//! `2mn² - 2n³/3` flops; `Q` is applied implicitly, so [`Qr::apply_q`] costs
//! `4mn` rather than the `2m²n` of forming `Q` densely.

use crate::check_dims;
use ncly_core::{NclyError, Result};
use std::fmt;

/// A Householder QR factorisation.
#[derive(Debug, Clone)]
pub struct Qr {
    /// `R` is `rows × cols`, stored row-major with zeros below the diagonal.
    r: Vec<f64>,
    /// Reflector vectors, one per column, stored densely as `rows × cols`.
    reflectors: Vec<f64>,
    /// Scalar factor of each reflector.
    taus: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Qr {
    /// Factorise a row-major `m × n` matrix with `m ≥ n`.
    pub fn factor(data: &[f64], rows: usize, cols: usize) -> Result<Self> {
        check_dims("qr", data, rows, cols)?;
        if rows < cols {
            return Err(NclyError::invalid_shape(
                "qr",
                format!("need rows >= cols, got {rows}x{cols}"),
            ));
        }
        let mut r = data.to_vec();
        let mut reflectors = vec![0.0; rows * cols];
        let mut taus = vec![0.0; cols];

        for k in 0..cols {
            // Norm of the trailing column segment, computed with rescaling.
            let mut scale = 0.0f64;
            for i in k..rows {
                scale = scale.max(r[i * cols + k].abs());
            }
            if scale == 0.0 {
                // The column is already zero: the reflector is the identity.
                taus[k] = 0.0;
                reflectors[k * cols + k] = 1.0;
                continue;
            }
            let mut norm = 0.0f64;
            for i in k..rows {
                let scaled = r[i * cols + k] / scale;
                norm += scaled * scaled;
            }
            let norm = scale * norm.sqrt();

            let alpha = r[k * cols + k];
            let beta = if alpha >= 0.0 { -norm } else { norm };
            let tau = (beta - alpha) / beta;

            // v = [1, x_{k+1}/ (alpha - beta), ...]
            let denominator = alpha - beta;
            reflectors[k * cols + k] = 1.0;
            for i in k + 1..rows {
                reflectors[i * cols + k] = r[i * cols + k] / denominator;
            }
            taus[k] = tau;

            // Apply H = I - tau v vᵀ to the trailing submatrix.
            for j in k..cols {
                let mut dot = r[k * cols + j];
                for i in k + 1..rows {
                    dot += reflectors[i * cols + k] * r[i * cols + j];
                }
                dot *= tau;
                r[k * cols + j] -= dot;
                for i in k + 1..rows {
                    r[i * cols + j] -= dot * reflectors[i * cols + k];
                }
            }
            // Force the exact triangular structure and store beta on the
            // diagonal so that orthogonality holds to machine precision.
            r[k * cols + k] = beta;
            for i in k + 1..rows {
                r[i * cols + k] = 0.0;
            }
        }

        Ok(Self {
            r,
            reflectors,
            taus,
            rows,
            cols,
        })
    }

    /// Number of rows of the original matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns of the original matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The upper triangular factor as a dense `m × n` row-major matrix.
    pub fn r(&self) -> &[f64] {
        &self.r
    }

    /// The thin `R` (`n × n`), which is the useful part for least squares.
    pub fn r_thin(&self) -> Vec<f64> {
        let n = self.cols;
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = self.r[i * self.cols + j];
            }
        }
        out
    }

    /// Form the `m × n` thin `Q` densely (columns of `Q` up to `n`).
    pub fn q(&self) -> Vec<f64> {
        let n = self.cols;
        let mut q = vec![0.0; self.rows * n];
        for column in 0..n {
            // Start from e_column restricted to the full row space.
            let mut vector = vec![0.0; self.rows];
            vector[column] = 1.0;
            // Apply reflectors in reverse order: Q = H_0 H_1 ... H_{n-1}.
            for k in (0..n).rev() {
                if self.taus[k] == 0.0 {
                    continue;
                }
                let mut dot = vector[k];
                for i in k + 1..self.rows {
                    dot += self.reflectors[i * self.cols + k] * vector[i];
                }
                dot *= self.taus[k];
                vector[k] -= dot;
                for i in k + 1..self.rows {
                    vector[i] -= dot * self.reflectors[i * self.cols + k];
                }
            }
            for row in 0..self.rows {
                q[row * n + column] = vector[row];
            }
        }
        q
    }

    /// Apply `Q` to a length-`m` vector: returns `Q·x`.
    pub fn apply_q(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.rows {
            return Err(NclyError::invalid_shape(
                "qr::apply_q",
                format!("vector has {} entries, expected {}", x.len(), self.rows),
            ));
        }
        let mut out = x.to_vec();
        for k in 0..self.cols {
            if self.taus[k] == 0.0 {
                continue;
            }
            let mut dot = out[k];
            for i in k + 1..self.rows {
                dot += self.reflectors[i * self.cols + k] * out[i];
            }
            dot *= self.taus[k];
            out[k] -= dot;
            for i in k + 1..self.rows {
                out[i] -= dot * self.reflectors[i * self.cols + k];
            }
        }
        Ok(out)
    }

    /// Apply `Qᵀ` to a length-`m` vector.
    pub fn apply_qt(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.rows {
            return Err(NclyError::invalid_shape(
                "qr::apply_qt",
                format!("vector has {} entries, expected {}", x.len(), self.rows),
            ));
        }
        let mut out = x.to_vec();
        for k in (0..self.cols).rev() {
            if self.taus[k] == 0.0 {
                continue;
            }
            let mut dot = out[k];
            for i in k + 1..self.rows {
                dot += self.reflectors[i * self.cols + k] * out[i];
            }
            dot *= self.taus[k];
            out[k] -= dot;
            for i in k + 1..self.rows {
                out[i] -= dot * self.reflectors[i * self.cols + k];
            }
        }
        Ok(out)
    }
}

impl fmt::Display for Qr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Qr({}x{})", self.rows, self.cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn reconstructs_square_matrices() {
        for n in 1..7 {
            let a: Vec<f64> = (0..n * n)
                .map(|i| ((i * 11 + 7) % 13) as f64 - 6.0)
                .collect();
            let qr = Qr::factor(&a, n, n).unwrap();
            let q = qr.q();
            let r = qr.r_thin();
            // Q·R
            let mut product = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut total = 0.0;
                    for k in 0..n {
                        total += q[i * n + k] * r[k * n + j];
                    }
                    product[i * n + j] = total;
                }
            }
            assert!(max_diff(&product, &a) < 1e-10, "n={n}");
        }
    }

    #[test]
    fn q_is_orthonormal() {
        let (m, n) = (7, 4);
        let a: Vec<f64> = (0..m * n).map(|i| ((i * 7 + 1) % 17) as f64 - 8.0).collect();
        let qr = Qr::factor(&a, m, n).unwrap();
        let q = qr.q();
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..m {
                    dot += q[k * n + i] * q[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12, "QᵀQ[{i}][{j}] = {dot}");
            }
        }
    }

    #[test]
    fn apply_qt_zeroes_the_tail() {
        let (m, n) = (5, 3);
        let a: Vec<f64> = (0..m * n).map(|i| (i as f64).sin()).collect();
        let qr = Qr::factor(&a, m, n).unwrap();
        let first_column: Vec<f64> = (0..m).map(|i| a[i * n]).collect();
        let rotated = qr.apply_qt(&first_column).unwrap();
        for entry in &rotated[1..] {
            assert!(entry.abs() < 1e-12, "tail entry {entry}");
        }
    }

    #[test]
    fn thin_factorisation_works() {
        // Least squares style shape.
        let (m, n) = (6, 2);
        let a: Vec<f64> = (0..m * n).map(|i| ((i % 5) as f64) + 1.0).collect();
        let qr = Qr::factor(&a, m, n).unwrap();
        assert_eq!(qr.r_thin().len(), n * n);
        assert_eq!(qr.q().len(), m * n);
    }

    #[test]
    fn zero_and_rank_deficient_columns_are_handled() {
        let a = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let qr = Qr::factor(&a, 3, 2).unwrap();
        assert_eq!(qr.r()[0], 0.0);
    }

    #[test]
    fn rejects_invalid_shapes() {
        assert!(Qr::factor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).is_err());
        assert!(Qr::factor(&[1.0, 2.0], 3, 3).is_err());
    }
}
