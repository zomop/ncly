//! Sparse matrices in compressed sparse row (CSR) and column (CSC) formats.
//!
//! # Storage
//!
//! [`CsrMatrix`] stores an `m × n` matrix as three arrays:
//!
//! ```text
//! row_ptr:    [0, 2, 3, 5]                 — m + 1 offsets into col_idx
//! col_idx:    [0, 2, 1, 0, 3]              — column of each stored value
//! values:     [1.5, 2.0, -1.0, 3.0, 4.0]   — the stored values
//! ```
//!
//! Row `i` owns the slice `col_idx[row_ptr[i]..row_ptr[i + 1]]`, so
//! sparse matrix–vector multiplication and row slicing are sequential reads.
//! [`CscMatrix`] is the transpose-oriented mirror image and is what column
//! access, `Aᵀx` and some preconditioners need.
//!
//! Duplicate entries are **summed** at construction time, which matches the
//! "coordinate input" convention of sparse toolkits. Explicit zeros are
//! preserved: they are structurally meaningful for some applications, and
//! dropping them silently would change the sparsity pattern users asked for.
//!
//! # Complexity
//!
//! | Operation | Cost |
//! | --- | --- |
//! | `from_triplets` | `O(nnz log nnz)` |
//! | `spmv` | `O(nnz)` |
//! | `spmm` (A·B) | `O(nnz(A) · avg row length of B)` |
//! | `transpose` | `O(nnz)` |
//! | `to_dense` | `O(mn)` |
//!
//! # Example
//!
//! ```
//! use ncly_sparse::CsrMatrix;
//!
//! // [[2, 0, 1], [0, 3, 0]]
//! let a = CsrMatrix::from_triplets(2, 3, &[0, 0, 1], &[0, 2, 1], &[2.0, 1.0, 3.0])?;
//! assert_eq!(a.nnz(), 3);
//! assert_eq!(a.spmv(&[1.0, 1.0, 1.0]), vec![3.0, 3.0]);
//! # Ok::<(), ncly_core::NclyError>(())
//! ```

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

mod csc;
mod csr;

pub use csc::CscMatrix;
pub use csr::CsrMatrix;

/// Row-major length check shared by the constructors.
pub(crate) fn check_dims(
    op: &'static str,
    data: &[f64],
    rows: usize,
    cols: usize,
) -> ncly_core::Result<()> {
    match rows.checked_mul(cols) {
        Some(expected) if expected == data.len() => Ok(()),
        Some(expected) => Err(ncly_core::NclyError::invalid_shape(
            op,
            format!(
                "expected {expected} elements for {rows}x{cols}, got {}",
                data.len()
            ),
        )),
        None => Err(ncly_core::NclyError::invalid_shape(
            op,
            "row/column count overflows usize",
        )),
    }
}
