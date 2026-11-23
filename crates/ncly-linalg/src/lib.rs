//! Dense linear algebra: LU, QR, solvers, inverses and eigenvalues.
//!
//! Every routine operates on row-major `f64` slices through a small [`Matrix`]
//! view type, so `ncly-linalg` does not depend on the tensor engine and can be
//! used stand-alone or reused by the sparse and Python layers.
//!
//! # Numerical contract
//!
//! * **Partial pivoting** is used throughout; a pivot is rejected only when it
//!   is negligible *relative to the matrix scale* (see
//!   [`ncly_core::scalar::is_negligible`]), so a matrix scaled by `1e-300` is
//!   not mistaken for a singular one.
//! * **Stability.** LU with partial pivoting has growth factor `≤ 2^{n-1}` in
//!   the worst case; QR via Householder reflectors is unconditionally stable
//!   and is used for least squares and for the eigenvalue iteration.
//! * **Determinants** are accumulated in log space to avoid overflow.
//! * **Residuals** are reported alongside failures so callers can distinguish
//!   "singular" from "ill-conditioned".
//!
//! # Example
//!
//! ```
//! use ncly_linalg::lu::Lu;
//!
//! // A = [[4, 3], [6, 3]]
//! let a = [4.0, 3.0, 6.0, 3.0];
//! let lu = Lu::factor(&a, 2).unwrap();
//! let x = lu.solve(&[10.0, 12.0]).unwrap();
//! // 4x + 3y = 10 ; 6x + 3y = 12  →  x = 1, y = 2
//! assert!((x[0] - 1.0).abs() < 1e-12);
//! assert!((x[1] - 2.0).abs() < 1e-12);
//! ```

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod cholesky;
pub mod eigen;
pub mod lu;
pub mod matrix;
pub mod qr;
pub mod solve;

pub use cholesky::Cholesky;
pub use eigen::{eigenvalues, Eigen};
pub use lib_norm::matrix_norm_inf;
pub use lu::Lu;
pub use matrix::Matrix;
pub use qr::Qr;
pub use solve::{inverse, lstsq, residual, solve};

use ncly_core::{NclyError, Result};

mod lib_norm;

/// Validate that `data` can describe an `rows × cols` matrix.
pub(crate) fn check_dims(op: &'static str, data: &[f64], rows: usize, cols: usize) -> Result<()> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| NclyError::invalid_shape(op, "row/column count overflows usize"))?;
    if data.len() != expected {
        return Err(NclyError::invalid_shape(
            op,
            format!(
                "expected {expected} elements for {rows}x{cols}, got {}",
                data.len()
            ),
        ));
    }
    Ok(())
}
