//! Compressed sparse row matrices.

use crate::csc::CscMatrix;
use core::fmt;
use ncly_core::{NclyError, Result};

/// A sparse matrix in compressed sparse row format.
#[derive(Clone, PartialEq)]
pub struct CsrMatrix {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// Build from coordinate (triplet) input.
    ///
    /// Duplicate `(row, col)` pairs are summed. Entries appear in each row in
    /// ascending column order, which is required for the merge-based kernels.
    ///
    /// # Errors
    ///
    /// Returns [`NclyError::MalformedData`] when the three slices have different
    /// lengths, or [`NclyError::IndexOutOfBounds`] when a coordinate is outside
    /// the declared shape.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        row: &[usize],
        col: &[usize],
        values: &[f64],
    ) -> Result<Self> {
        if row.len() != col.len() || row.len() != values.len() {
            return Err(NclyError::malformed(
                "csr::from_triplets",
                format!(
                    "row ({}), col ({}) and values ({}) must have equal lengths",
                    row.len(),
                    col.len(),
                    values.len()
                ),
            ));
        }
        let mut counts = vec![0usize; rows + 1];
        for &r in row {
            if r >= rows {
                return Err(NclyError::IndexOutOfBounds {
                    index: vec![r],
                    shape: format!("({rows}, {cols})"),
                });
            }
            counts[r + 1] += 1;
        }
        for &c in col {
            if c >= cols {
                return Err(NclyError::IndexOutOfBounds {
                    index: vec![c],
                    shape: format!("({rows}, {cols})"),
                });
            }
        }
        for i in 0..rows {
            counts[i + 1] += counts[i];
        }
        let nnz = row.len();
        let mut col_idx = vec![0usize; nnz];
        let mut data = vec![0.0f64; nnz];
        let mut cursor = counts.clone();
        for index in 0..nnz {
            let r = row[index];
            let position = cursor[r];
            col_idx[position] = col[index];
            data[position] = values[index];
            cursor[r] += 1;
        }

        // Sort each row and merge duplicates.
        let mut merge_rows = Vec::with_capacity(rows + 1);
        merge_rows.push(0usize);
        let mut merged_cols = Vec::with_capacity(nnz);
        let mut merged_values = Vec::with_capacity(nnz);
        for r in 0..rows {
            let start = counts[r];
            let end = counts[r + 1];
            let mut order: Vec<usize> = (start..end).collect();
            order.sort_by_key(|&i| col_idx[i]);
            let mut index = 0;
            while index < order.len() {
                let column = col_idx[order[index]];
                let mut total = 0.0;
                while index < order.len() && col_idx[order[index]] == column {
                    total += data[order[index]];
                    index += 1;
                }
                merged_cols.push(column);
                merged_values.push(total);
            }
            merge_rows.push(merged_cols.len());
        }

        Ok(Self {
            rows,
            cols,
            row_ptr: merge_rows,
            col_idx: merged_cols,
            values: merged_values,
        })
    }

    /// Build from a dense row-major matrix, dropping exact zeros.
    pub fn from_dense(data: &[f64], rows: usize, cols: usize) -> Result<Self> {
        crate::check_dims("csr::from_dense", data, rows, cols)?;
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut values = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                let value = data[i * cols + j];
                if value != 0.0 {
                    row.push(i);
                    col.push(j);
                    values.push(value);
                }
            }
        }
        Self::from_triplets(rows, cols, &row, &col, &values)
    }

    /// Build from validated parts without checking invariants.
    ///
    /// # Panics
    ///
    /// Panics (in debug builds) when the internals are inconsistent. Used by
    /// format conversions that are already known to be valid.
    pub(crate) fn from_parts_unchecked(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), rows + 1);
        debug_assert_eq!(row_ptr[0], 0);
        debug_assert_eq!(*row_ptr.last().unwrap_or(&0), col_idx.len());
        debug_assert_eq!(col_idx.len(), values.len());
        Self {
            rows,
            cols,
            row_ptr,
            col_idx,
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

    /// Row pointer array.
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Column index array.
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// Stored values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutable access to the stored values (structure must not change).
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// `(columns, values)` of a single row.
    pub fn row(&self, index: usize) -> Result<(&[usize], &[f64])> {
        if index >= self.rows {
            return Err(NclyError::IndexOutOfBounds {
                index: vec![index],
                shape: format!("({}, {})", self.rows, self.cols),
            });
        }
        let start = self.row_ptr[index];
        let end = self.row_ptr[index + 1];
        Ok((&self.col_idx[start..end], &self.values[start..end]))
    }

    /// Look up one element, `0.0` when structurally absent.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.rows || col >= self.cols {
            return 0.0;
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        let columns = &self.col_idx[start..end];
        match columns.binary_search(&col) {
            Ok(position) => self.values[start + position],
            Err(_) => 0.0,
        }
    }

    /// Sparse matrix–vector product `y = A·x`.
    ///
    /// # Panics
    ///
    /// Panics when `x` does not have `self.cols()` entries; call
    /// [`CsrMatrix::try_spmv`] for a checked version.
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        self.try_spmv(x).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Checked sparse matrix–vector product.
    pub fn try_spmv(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.cols {
            return Err(NclyError::invalid_shape(
                "csr::spmv",
                format!("vector has {} entries, expected {}", x.len(), self.cols),
            ));
        }
        let mut y = vec![0.0; self.rows];
        for (row, result) in y.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            let mut total = 0.0f64;
            let mut compensation = 0.0f64;
            for position in start..end {
                let term = self.values[position] * x[self.col_idx[position]];
                let t = total + term;
                if total.abs() >= term.abs() {
                    compensation += (total - t) + term;
                } else {
                    compensation += (term - t) + total;
                }
                total = t;
            }
            *result = total + compensation;
        }
        Ok(y)
    }

    /// `y = Aᵀ·x`, accumulated with the same compensated scheme as [`spmv`].
    ///
    /// [`spmv`]: CsrMatrix::spmv
    pub fn spmv_transpose(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.rows {
            return Err(NclyError::invalid_shape(
                "csr::spmv_transpose",
                format!("vector has {} entries, expected {}", x.len(), self.rows),
            ));
        }
        let mut y = vec![0.0; self.cols];
        for row in 0..self.rows {
            let scale = x[row];
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for position in start..end {
                y[self.col_idx[position]] += scale * self.values[position];
            }
        }
        Ok(y)
    }

    /// Sparse matrix–matrix product `C = A·B`.
    ///
    /// Uses the "row of C = A's row times B" formulation with an accumulator
    /// and a touched-column list, giving `O(nnz(A) · rowlen(B))` work and
    /// `O(nnz(C))` extra memory.
    pub fn spmm(&self, other: &CsrMatrix) -> Result<CsrMatrix> {
        if self.cols != other.rows {
            return Err(NclyError::shape_mismatch(
                "csr::spmm",
                format!("({}, {})", self.rows, self.cols),
                format!("({}, {})", other.rows, other.cols),
            ));
        }
        let mut row_ptr = Vec::with_capacity(self.rows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        let mut accumulator = vec![0.0f64; other.cols];
        let mut touched: Vec<usize> = Vec::new();
        row_ptr.push(0);

        for row in 0..self.rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for position in start..end {
                let scale = self.values[position];
                let inner = self.col_idx[position];
                let b_start = other.row_ptr[inner];
                let b_end = other.row_ptr[inner + 1];
                for offset in b_start..b_end {
                    let column = other.col_idx[offset];
                    if accumulator[column] == 0.0 {
                        touched.push(column);
                    }
                    accumulator[column] += scale * other.values[offset];
                }
            }
            touched.sort_unstable();
            for &column in &touched {
                let value = accumulator[column];
                if value != 0.0 {
                    col_idx.push(column);
                    values.push(value);
                }
                accumulator[column] = 0.0;
            }
            touched.clear();
            row_ptr.push(col_idx.len());
        }

        Ok(CsrMatrix::from_parts_unchecked(
            self.rows, other.cols, row_ptr, col_idx, values,
        ))
    }

    /// Transpose into CSC format (a structural operation, `O(nnz)`).
    pub fn to_csc(&self) -> CscMatrix {
        // Counting sort by column.
        let mut col_ptr = vec![0usize; self.cols + 1];
        for &column in &self.col_idx {
            col_ptr[column + 1] += 1;
        }
        for j in 0..self.cols {
            col_ptr[j + 1] += col_ptr[j];
        }
        let mut row_idx = vec![0usize; self.nnz()];
        let mut values = vec![0.0f64; self.nnz()];
        let mut cursor = col_ptr.clone();
        for row in 0..self.rows {
            for position in self.row_ptr[row]..self.row_ptr[row + 1] {
                let column = self.col_idx[position];
                let target = cursor[column];
                row_idx[target] = row;
                values[target] = self.values[position];
                cursor[column] += 1;
            }
        }
        CscMatrix::from_parts_unchecked(self.rows, self.cols, col_ptr, row_idx, values)
    }

    /// Dense row-major copy.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.rows * self.cols];
        for row in 0..self.rows {
            for position in self.row_ptr[row]..self.row_ptr[row + 1] {
                out[row * self.cols + self.col_idx[position]] = self.values[position];
            }
        }
        out
    }

    /// `A + B` element-wise; the result has the union sparsity pattern.
    pub fn add(&self, other: &CsrMatrix) -> Result<CsrMatrix> {
        if self.shape() != other.shape() {
            return Err(NclyError::shape_mismatch(
                "csr::add",
                format!("({}, {})", self.rows, self.cols),
                format!("({}, {})", other.rows, other.cols),
            ));
        }
        let mut row_ptr = Vec::with_capacity(self.rows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for row in 0..self.rows {
            let (mut i, i_end) = (self.row_ptr[row], self.row_ptr[row + 1]);
            let (mut j, j_end) = (other.row_ptr[row], other.row_ptr[row + 1]);
            while i < i_end || j < j_end {
                let left_col = if i < i_end {
                    self.col_idx[i]
                } else {
                    usize::MAX
                };
                let right_col = if j < j_end {
                    other.col_idx[j]
                } else {
                    usize::MAX
                };
                let (column, value) = match left_col.cmp(&right_col) {
                    core::cmp::Ordering::Less => {
                        let value = self.values[i];
                        i += 1;
                        (left_col, value)
                    }
                    core::cmp::Ordering::Greater => {
                        let value = other.values[j];
                        j += 1;
                        (right_col, value)
                    }
                    core::cmp::Ordering::Equal => {
                        let value = self.values[i] + other.values[j];
                        i += 1;
                        j += 1;
                        (left_col, value)
                    }
                };
                if value != 0.0 {
                    col_idx.push(column);
                    values.push(value);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Ok(CsrMatrix::from_parts_unchecked(
            self.rows, self.cols, row_ptr, col_idx, values,
        ))
    }

    /// Multiply every stored value by `scalar`.
    pub fn scale(&self, scalar: f64) -> CsrMatrix {
        CsrMatrix::from_parts_unchecked(
            self.rows,
            self.cols,
            self.row_ptr.clone(),
            self.col_idx.clone(),
            self.values.iter().map(|v| v * scalar).collect(),
        )
    }

    /// Apply `f` to every stored value.
    pub fn map_values(&self, f: impl Fn(f64) -> f64) -> CsrMatrix {
        CsrMatrix::from_parts_unchecked(
            self.rows,
            self.cols,
            self.row_ptr.clone(),
            self.col_idx.clone(),
            self.values.iter().map(|&v| f(v)).collect(),
        )
    }

    /// Sum of the diagonal.
    pub fn trace(&self) -> f64 {
        let limit = self.rows.min(self.cols);
        (0..limit).map(|i| self.get(i, i)).sum()
    }

    /// Whether the pattern is square and every off-diagonal entry has a
    /// mirrored counterpart.
    pub fn is_symmetric_pattern(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for row in 0..self.rows {
            for position in self.row_ptr[row]..self.row_ptr[row + 1] {
                let column = self.col_idx[position];
                let start = self.row_ptr[column];
                let end = self.row_ptr[column + 1];
                if self.col_idx[start..end].binary_search(&row).is_err() {
                    return false;
                }
            }
        }
        true
    }

    /// Largest absolute stored value.
    pub fn max_abs(&self) -> f64 {
        self.values.iter().fold(0.0f64, |acc, v| acc.max(v.abs()))
    }

    /// Frobenius norm, computed with rescaling.
    pub fn norm(&self) -> f64 {
        let scale = self.max_abs();
        if scale == 0.0 {
            return 0.0;
        }
        let total: f64 = self.values.iter().map(|v| (v / scale) * (v / scale)).sum();
        scale * total.sqrt()
    }

    /// Convert to a dense representation.
    pub fn dense(&self) -> Vec<f64> {
        self.to_dense()
    }
}

impl fmt::Debug for CsrMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CsrMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("nnz", &self.nnz())
            .field("density", &self.density())
            .finish()
    }
}

impl fmt::Display for CsrMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CsrMatrix({}x{}, nnz={}, density={:.3})",
            self.rows,
            self.cols,
            self.nnz(),
            self.density()
        )
    }
}

impl CsrMatrix {
    /// Fraction of stored entries, in `[0, 1]`.
    pub fn density(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CsrMatrix {
        // [[10, 0, 20], [0, 30, 0], [40, 0, 50]]
        CsrMatrix::from_triplets(
            3,
            3,
            &[0, 0, 1, 2, 2],
            &[0, 2, 1, 0, 2],
            &[10.0, 20.0, 30.0, 40.0, 50.0],
        )
        .unwrap()
    }

    #[test]
    fn structure_is_correct() {
        let a = sample();
        assert_eq!(a.row_ptr(), &[0, 2, 3, 5]);
        assert_eq!(a.col_idx(), &[0, 2, 1, 0, 2]);
        assert_eq!(a.nnz(), 5);
        assert_eq!(a.get(0, 0), 10.0);
        assert_eq!(a.get(0, 1), 0.0);
        assert_eq!(a.get(2, 2), 50.0);
    }

    #[test]
    fn triplets_are_sorted_and_summed() {
        let a = CsrMatrix::from_triplets(2, 3, &[0, 0, 0, 1], &[2, 0, 2, 1], &[1.0, 5.0, 2.0, 7.0])
            .unwrap();
        assert_eq!(a.col_idx(), &[0, 2, 1]);
        assert_eq!(a.values(), &[5.0, 3.0, 7.0]);
    }

    #[test]
    fn spmv_matches_dense() {
        let a = sample();
        let x = [1.0, 2.0, 3.0];
        assert_eq!(a.spmv(&x), vec![70.0, 60.0, 190.0]);
        // Compare against a dense loop.
        let dense = a.to_dense();
        for i in 0..3 {
            let mut expected = 0.0;
            for j in 0..3 {
                expected += dense[i * 3 + j] * x[j];
            }
            assert!((a.spmv(&x)[i] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn spmv_transpose_matches_dense_transpose() {
        let a = sample();
        let x = [1.0, 2.0, 3.0];
        let y = a.spmv_transpose(&x).unwrap();
        assert_eq!(y, vec![130.0, 60.0, 170.0]);
        assert_eq!(a.to_csc().spmv(&x).unwrap(), a.spmv(&x));
    }

    #[test]
    fn spmm_matches_dense_product() {
        let a = sample();
        let b =
            CsrMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();
        let c = a.spmm(&b).unwrap();
        let dense_a = a.to_dense();
        let dense_b = b.to_dense();
        let mut expected = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    expected[i * 3 + j] += dense_a[i * 3 + k] * dense_b[k * 3 + j];
                }
            }
        }
        let got = c.to_dense();
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-12, "{g} vs {e}");
        }
    }

    #[test]
    fn transpose_round_trips() {
        let a = sample();
        let round_trip = a.to_csc().to_csr();
        assert_eq!(round_trip, a);
    }

    #[test]
    fn add_unions_patterns() {
        let a = CsrMatrix::from_triplets(2, 2, &[0, 1], &[0, 1], &[1.0, 2.0]).unwrap();
        let b = CsrMatrix::from_triplets(2, 2, &[0, 1], &[1, 0], &[3.0, 4.0]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_dense(), vec![1.0, 3.0, 4.0, 2.0]);
    }

    #[test]
    fn cancelling_entries_are_dropped() {
        let a = CsrMatrix::from_triplets(1, 1, &[0], &[0], &[1.0]).unwrap();
        let b = CsrMatrix::from_triplets(1, 1, &[0], &[0], &[-1.0]).unwrap();
        assert_eq!(a.add(&b).unwrap().nnz(), 0);
    }

    #[test]
    fn symmetry_detection() {
        let symmetric =
            CsrMatrix::from_triplets(2, 2, &[0, 1, 0, 1], &[0, 1, 1, 0], &[1.0, 1.0, 2.0, 2.0])
                .unwrap();
        assert!(symmetric.is_symmetric_pattern());
        let a = sample();
        assert!(!a.is_symmetric_pattern());
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(CsrMatrix::from_triplets(2, 2, &[0], &[0, 1], &[1.0]).is_err());
        assert!(CsrMatrix::from_triplets(2, 2, &[5], &[0], &[1.0]).is_err());
        assert!(CsrMatrix::from_triplets(2, 2, &[0], &[5], &[1.0]).is_err());
        assert!(sample().spmv_transpose(&[1.0, 2.0]).is_err());
        assert!(sample().try_spmv(&[1.0]).is_err());
        assert!(sample().spmm(&sample()).is_ok());
    }

    #[test]
    fn empty_matrices() {
        let a = CsrMatrix::from_triplets(0, 0, &[], &[], &[]).unwrap();
        assert_eq!(a.nnz(), 0);
        assert_eq!(a.density(), 0.0);
        assert_eq!(a.spmv(&[]), Vec::<f64>::new());
        let b = CsrMatrix::from_triplets(3, 2, &[], &[], &[]).unwrap();
        assert_eq!(b.to_dense(), vec![0.0; 6]);
    }

    #[test]
    fn norms_and_trace() {
        let a = sample();
        assert_eq!(a.trace(), 90.0);
        assert_eq!(a.max_abs(), 50.0);
        assert!((a.norm() - (100.0 + 400.0 + 900.0 + 1600.0 + 2500.0f64).sqrt()).abs() < 1e-9);
    }
}
