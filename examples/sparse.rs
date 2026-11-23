//! Sparse CSR assembly and matrix-vector products.
//!
//! Builds the 1-d Laplacian and multiplies it by a few vectors.
//!
//! Run with: `cargo run --release --example sparse`

use ncly::sparse::CsrMatrix;
use ncly::NclyError;

fn main() -> Result<(), NclyError> {
    let n = 10usize;
    let (mut rows, mut cols, mut values) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        values.push(2.0);
        if i + 1 < n {
            rows.push(i);
            cols.push(i + 1);
            values.push(-1.0);
            rows.push(i + 1);
            cols.push(i);
            values.push(-1.0);
        }
    }

    let laplacian = CsrMatrix::from_triplets(n, n, &rows, &cols, &values)?;
    println!(
        "shape {}x{}, nnz = {}",
        laplacian.rows(),
        laplacian.cols(),
        laplacian.nnz()
    );

    // A·1 = e_0 + e_{n-1} for this stencil.
    let ax = laplacian.spmv(&vec![1.0; n]);
    println!("A·1              {:?}…", &ax[..3]);
    assert!((ax[0] - 1.0).abs() < 1e-12);
    assert!((ax[n - 1] - 1.0).abs() < 1e-12);
    assert!(ax[1..n - 1].iter().all(|v| v.abs() < 1e-12));

    // Quadratic form xᵀAx > 0 for non-constant x (positive definite).
    let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let quadratic: f64 = x.iter().zip(laplacian.spmv(&x)).map(|(a, b)| a * b).sum();
    println!("xᵀAx (x = 0..0.9) {quadratic:.6}  (must be > 0)");
    assert!(quadratic > 0.0);

    // Sparse × sparse and format conversions.
    let squared = laplacian.spmm(&laplacian)?;
    println!("A² nnz           {}", squared.nnz());
    let csc = laplacian.to_csc();
    let dense = laplacian.to_dense();
    assert_eq!(dense.len(), n * n);
    assert_eq!(csc.nnz(), laplacian.nnz());

    // The dense reference must agree with the sparse kernel.
    let x = vec![1.0; n];
    let dense_rows = linalg_matvec(&squared.to_dense(), &x, n);
    for (i, (sparse, dense)) in squared.spmv(&x).iter().zip(&dense_rows).enumerate() {
        assert!(
            (sparse - dense).abs() < 1e-12,
            "row {i}: {sparse} vs {dense}"
        );
    }
    println!("sparse and dense agree.");
    Ok(())
}

fn linalg_matvec(matrix: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| matrix[i * n + j] * x[j]).sum())
        .collect()
}
