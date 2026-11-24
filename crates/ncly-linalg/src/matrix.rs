//! Minimal row-major matrix view.

use core::fmt;
use ncly_core::{NclyError, Result};

/// An immutable row-major `rows × cols` view over `f64` data.
#[derive(Clone, Copy)]
pub struct Matrix<'a> {
    data: &'a [f64],
    rows: usize,
    cols: usize,
}

impl<'a> Matrix<'a> {
    /// Wrap a slice, checking that its length matches `rows * cols`.
    pub fn new(data: &'a [f64], rows: usize, cols: usize) -> Result<Self> {
        crate::check_dims("Matrix::new", data, rows, cols)?;
        Ok(Self { data, rows, cols })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major element access.
    ///
    /// # Panics
    ///
    /// Panics when the index is out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// Raw row-major data.
    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }

    /// Require a square matrix.
    pub fn square(&self) -> Result<()> {
        if self.rows != self.cols {
            return Err(NclyError::invalid_shape(
                "square",
                format!("matrix is {}x{}", self.rows, self.cols),
            ));
        }
        Ok(())
    }

    /// `Aᵀ` in a new buffer.
    pub fn transpose(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        out
    }
}

impl fmt::Debug for Matrix<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish_non_exhaustive()
    }
}
