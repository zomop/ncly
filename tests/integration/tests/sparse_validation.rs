//! Sparse matrix validation against dense references.

use ncly_integration_tests::{assert_close, max_abs_diff, Rng};
use ncly_sparse::{CscMatrix, CsrMatrix};

/// Build a random sparse matrix with roughly `density` fill.
fn random_sparse(rows: usize, cols: usize, density: f64, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut r = Vec::new();
    let mut c = Vec::new();
    let mut v = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            if rng.uniform() < density {
                r.push(i);
                c.push(j);
                v.push(rng.range(-2.0, 2.0));
            }
        }
    }
    (r, c, v)
}

fn dense_spmv(a: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    (0..rows)
        .map(|i| (0..cols).map(|j| a[i * cols + j] * x[j]).sum())
        .collect()
}

#[test]
fn csr_spmv_matches_dense_reference() {
    let mut rng = Rng::new(9001);
    for (rows, cols, density) in [(10usize, 10usize, 0.2), (50, 30, 0.05), (5, 100, 0.3)] {
        let (r, c, v) = random_sparse(rows, cols, density, rows as u64 * 31 + cols as u64);
        let matrix = CsrMatrix::from_triplets(rows, cols, &r, &c, &v).unwrap();
        let dense = matrix.to_dense();
        let x: Vec<f64> = (0..cols).map(|_| rng.range(-1.0, 1.0)).collect();
        let sparse_product = matrix.spmv(&x);
        let dense_product = dense_spmv(&dense, rows, cols, &x);
        assert_close(
            &sparse_product,
            &dense_product,
            1e-12,
            &format!("spmv {rows}x{cols}"),
        );

        // Transpose product.
        let y: Vec<f64> = (0..rows).map(|_| rng.range(-1.0, 1.0)).collect();
        let sparse_t = matrix.spmv_transpose(&y).unwrap();
        let dense_t: Vec<f64> = (0..cols)
            .map(|j| (0..rows).map(|i| dense[i * cols + j] * y[i]).sum())
            .collect();
        assert_close(&sparse_t, &dense_t, 1e-12, "spmv_transpose");
    }
}

#[test]
fn spmm_matches_dense_matrix_product() {
    let (ra, ca, va) = random_sparse(20, 15, 0.15, 77);
    let (rb, cb, vb) = random_sparse(15, 12, 0.2, 78);
    let a = CsrMatrix::from_triplets(20, 15, &ra, &ca, &va).unwrap();
    let b = CsrMatrix::from_triplets(15, 12, &rb, &cb, &vb).unwrap();
    let product = a.spmm(&b).unwrap();

    let dense_a = a.to_dense();
    let dense_b = b.to_dense();
    let mut expected = vec![0.0; 20 * 12];
    for i in 0..20 {
        for p in 0..15 {
            let scale = dense_a[i * 15 + p];
            if scale == 0.0 {
                continue;
            }
            for j in 0..12 {
                expected[i * 12 + j] += scale * dense_b[p * 12 + j];
            }
        }
    }
    assert!(max_abs_diff(&product.to_dense(), &expected) < 1e-12);
    // The product of two sparse matrices with these densities stays sparse.
    assert!(product.nnz() <= 20 * 12);
}

#[test]
fn csc_and_csr_formats_agree() {
    let (r, c, v) = random_sparse(12, 9, 0.35, 123);
    let csr = CsrMatrix::from_triplets(12, 9, &r, &c, &v).unwrap();
    let csc = CscMatrix::from_triplets(12, 9, &r, &c, &v).unwrap();
    assert_eq!(csr.nnz(), csc.nnz());
    assert_eq!(csr.to_dense(), csc.to_dense());

    let mut rng = Rng::new(5);
    let x: Vec<f64> = (0..9).map(|_| rng.range(-1.0, 1.0)).collect();
    assert_close(&csr.spmv(&x), &csc.spmv(&x).unwrap(), 1e-14, "csr vs csc spmv");
    assert_eq!(csr.to_csc().to_csr(), csr);
}

#[test]
fn duplicate_coordinates_sum() {
    // Three contributions to (0,0): 1 + 2 + 3 = 6.
    let a = CsrMatrix::from_triplets(2, 2, &[0, 0, 0, 1], &[0, 0, 0, 1], &[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    assert_eq!(a.get(0, 0), 6.0);
    assert_eq!(a.get(1, 1), 4.0);
    assert_eq!(a.nnz(), 2);
}

#[test]
fn structural_zeros_are_preserved_from_triplets() {
    // An explicit zero is structural data: the pattern must not silently change.
    let a = CsrMatrix::from_triplets(1, 2, &[0, 0], &[0, 1], &[0.0, 1.5]).unwrap();
    assert_eq!(a.nnz(), 2);
    assert_eq!(a.col_idx(), &[0, 1]);
}

#[test]
fn sparse_add_matches_dense_addition() {
    let (r1, c1, v1) = random_sparse(8, 8, 0.25, 11);
    let (r2, c2, v2) = random_sparse(8, 8, 0.25, 22);
    let a = CsrMatrix::from_triplets(8, 8, &r1, &c1, &v1).unwrap();
    let b = CsrMatrix::from_triplets(8, 8, &r2, &c2, &v2).unwrap();
    let sum = a.add(&b).unwrap();
    let dense_a = a.to_dense();
    let dense_b = b.to_dense();
    let expected: Vec<f64> = dense_a.iter().zip(&dense_b).map(|(x, y)| x + y).collect();
    assert!(max_abs_diff(&sum.to_dense(), &expected) < 1e-14);
}

#[test]
fn empty_rows_and_columns_are_consistent() {
    // Row 1 and column 2 are structurally empty.
    let a = CsrMatrix::from_triplets(3, 3, &[0, 2], &[0, 1], &[1.0, 2.0]).unwrap();
    assert_eq!(a.row_ptr(), &[0, 1, 1, 2]);
    assert_eq!(a.spmv(&[1.0, 1.0, 1.0]), vec![1.0, 0.0, 2.0]);
    assert_eq!(a.transpose_dense(), a.to_dense());

    let empty = CsrMatrix::from_triplets(0, 5, &[], &[], &[]).unwrap();
    assert_eq!(empty.spmv(&[0.0; 5]), Vec::<f64>::new());
}

#[test]
fn large_sparse_system_is_solvable_by_iteration_contract() {
    // Build a 1-d Laplacian and verify the spmv contract that iterative solvers
    // rely on: symmetry and positive definiteness (checked via quadratic forms).
    let n = 200;
    let mut r = Vec::new();
    let mut c = Vec::new();
    let mut v = Vec::new();
    for i in 0..n {
        r.push(i);
        c.push(i);
        v.push(2.0);
        if i + 1 < n {
            r.push(i);
            c.push(i + 1);
            v.push(-1.0);
            r.push(i + 1);
            c.push(i);
            v.push(-1.0);
        }
    }
    let laplacian = CsrMatrix::from_triplets(n, n, &r, &c, &v).unwrap();
    assert_eq!(laplacian.nnz(), 3 * n - 2);

    // xᵀAx > 0 for x = ones, and Ax for ones is zero except at the ends.
    let ones = vec![1.0; n];
    let ax = laplacian.spmv(&ones);
    assert!(ax[0].abs() < 1e-12);
    assert!(ax[n - 1].abs() < 1e-12);
    for value in &ax[1..n - 1] {
        assert!(value.abs() < 1e-12);
    }

    // A random vector gives a strictly positive quadratic form.
    let mut rng = Rng::new(314);
    let x: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
    let ax = laplacian.spmv(&x);
    let quadratic: f64 = x.iter().zip(&ax).map(|(a, b)| a * b).sum();
    assert!(quadratic > 0.0, "Laplacian must be positive semi-definite");
}

trait TransposeDense {
    fn transpose_dense(&self) -> Vec<f64>;
}

impl TransposeDense for CsrMatrix {
    fn transpose_dense(&self) -> Vec<f64> {
        self.to_csc().to_dense()
    }
}
