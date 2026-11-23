//! Compressed sparse column matrices.

use crate::csr::CsrMatrix;
use core::fmt;
use ncly_core::{NclyError, Result};

/// A sparse matrix in compressed sparse column format.
///
/// CSC is the natural layout when the *columns* are the unit of work: `Aᵀx`
/// products, extracting columns for the QR of a sparse least squares problem,
/// and column-oriented preconditioners.
#[derive(Clone, PartialEq)]
pub struct CscMatrix {
    rows: usize,
    cols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
}

impl CscMatrix {
    /// Build from coordinate (triplet) input; duplicates are summed and each
    /// column is sorted by row index.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        row: &[usize],
        col: &[usize],
        values: &[f64],
    ) -> Result<Self> {
        let csr = CsrMatrix::from_triplets(rows, cols, row, col, values)?;
        Ok(csr.to_csc())
    }

    /// Build from a dense row-major matrix, dropping exact zeros.
    pub fn from_dense(data: &[f64], rows: usize, cols: usize) -> Result<Self> {
        Ok(CsrMatrix::from_dense(data, rows, cols)?.to_csc())
    }

    pub(crate) fn from_parts_unchecked(
        rows: usize,
        cols: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(col_ptr.len(), cols + 1);
        debug_assert_eq!(col_ptr[0], 0);
        debug_assert_eq!(*col_ptr.last().unwrap_or(&0), row_idx.len());
        debug_assert_eq!(row_idx.len(), values.len());
        Self {
            rows,
            cols,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Shape as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Column pointer array.
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    /// Row index array.
    pub fn row_idx(&self) -> &[usize] {
        &self.row_idx
    }

    /// Stored values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// `(rows, values)` of a single column.
    pub fn column(&self, index: usize) -> Result<(&[usize], &[f64])> {
        if index >= self.cols {
            return Err(NclyError::IndexOutOfBounds {
                index: vec![index],
                shape: format!("({}, {})", self.rows, self.cols),
            });
        }
        let start = self.col_ptr[index];
        let end = self.col_ptr[index + 1];
        Ok((&self.row_idx[start..end], &self.values[start..end]))
    }

    /// Sparse matrix–vector product `y = A·x` (column-oriented).
    ///
    /// Each column contributes an AXPY onto `y`, so the loop reads the
    /// compressed arrays sequentially.
    pub fn spmv(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.cols {
            return Err(NclyError::invalid_shape(
                "csc::spmv",
                format!("vector has {} entries, expected {}", x.len(), self.cols),
            ));
        }
        let mut y = vec![0.0; self.rows];
        for column in 0..self.cols {
            let scale = x[column];
            if scale == 0.0 {
                continue;
            }
            for position in self.col_ptr[column]..self.col_ptr[column + 1] {
                y[self.row_idx[position]] += scale * self.values[position];
            }
        }
        Ok(y)
    }

    /// `y = Aᵀ·x` (row-oriented over CSC, i.e. the transpose product).
    pub fn spmv_transpose(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.rows {
            return Err(NclyError::invalid_shape(
                "csc::spmv_transpose",
                format!("vector has {} entries, expected {}", x.len(), self.rows),
            ));
        }
        let mut y = vec![0.0; self.cols];
        for column in 0..self.cols {
            let mut total = 0.0;
            for position in self.col_ptr[column]..self.col_ptr[column + 1] {
                total += self.values[position] * x[self.row_idx[position]];
            }
            y[column] = total;
        }
        Ok(y)
    }

    /// Convert back to CSR.
    pub fn to_csr(&self) -> CsrMatrix {
        let mut row_ptr = vec![0usize; self.rows + 1];
        for &row in &self.row_idx {
            row_ptr[row + 1] += 1;
        }
        for i in 0..self.rows {
            row_ptr[i + 1] += row_ptr[i];
        }
        let mut col_idx = vec![0usize; self.nnz()];
        let mut values = vec![0.0f64; self.nnz()];
        let mut cursor = row_ptr.clone();
        for column in 0..self.cols {
            for position in self.col_ptr[column]..self.col_ptr[column + 1] {
                let row = self.row_idx[position];
                let target = cursor[row];
                col_idx[target] = column;
                values[target] = self.values[position];
                cursor[row] += 1;
            }
        }
        CsrMatrix::from_parts_unchecked(self.rows, self.cols, row_ptr, col_idx, values)
    }

    /// Dense row-major copy.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.rows * self.cols];
        for column in 0..self.cols {
            for position in self.col_ptr[column]..self.col_ptr[column + 1] {
                out[self.row_idx[position] * self.cols + column] = self.values[position];
            }
        }
        out
    }
}

impl fmt::Debug for CscMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CscMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("nnz", &self.nnz())
            .finish()
    }
}

impl fmt::Display for CscMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CscMatrix({}x{}, nnz={})",
            self.rows,
            self.cols,
            self.nnz()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CscMatrix {
        // [[1, 0], [0, 2], [3, 0]]
        CscMatrix::from_triplets(3, 2, &[0, 1, 2], &[0, 1, 0], &[1.0, 2.0, 3.0]).unwrap()
    }

    #[test]
    fn structure_is_column_major() {
        let a = sample();
        assert_eq!(a.col_ptr(), &[0, 2, 3]);
        assert_eq!(a.row_idx(), &[0, 2, 1]);
        assert_eq!(a.nnz(), 3);
    }

    #[test]
    fn spmv_matches_dense() {
        let a = sample();
        assert_eq!(a.spmv(&[2.0, 5.0]).unwrap(), vec![2.0, 10.0, 6.0]);
        let b = CscMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(b.spmv(&[1.0, 1.0]).unwrap(), vec![3.0, 7.0]);
    }

    #[test]
    fn transpose_product() {
        let a = sample();
        let y = a.spmv_transpose(&[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(y, vec![4.0, 2.0]);
    }

    #[test]
    fn conversion_round_trips() {
        let a = sample();
        let round_trip = a.to_csr().to_csc();
        assert_eq!(round_trip, a);
        assert_eq!(a.to_csr().to_dense(), a.to_dense());
    }

    #[test]
    fn columns_are_addressable() {
        let a = sample();
        let (rows, values) = a.column(0).unwrap();
        assert_eq!(rows, &[0, 2]);
        assert_eq!(values, &[1.0, 3.0]);
        assert!(a.column(5).is_err());
    }

    #[test]
    fn shape_errors() {
        let a = sample();
        assert!(a.spmv(&[1.0]).is_err());
        assert!(a.spmv_transpose(&[1.0]).is_err());
    }
}
