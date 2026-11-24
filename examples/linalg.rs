//! Dense linear algebra: LU, QR least squares, Cholesky, eigenvalues.
//!
//! Run with: `cargo run --release --example linalg`

use ncly::linalg::{self, cholesky::Cholesky, eigen, qr::Qr, Matrix};
use ncly::NclyError;

fn main() -> Result<(), NclyError> {
    // --- LU solve: [4 3; 6 3] x = [10, 12]  →  x = [1, 2] ------------------
    let a = [4.0, 3.0, 6.0, 3.0];
    let b = [10.0, 12.0];
    let x = linalg::solve(&a, &b, 2)?;
    println!("LU solve         x = [{:.6}, {:.6}]", x[0], x[1]);
    println!(
        "residual         ‖Ax-b‖ = {:.3e}",
        linalg::residual(&a, &x, &b, 2, 2)?
    );

    // --- QR least squares for an overdetermined system ---------------------
    // Points (1,1), (1,3), (1,5) with the design matrix [1 0; 1 1; 1 2].
    let design = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0];
    let targets = [1.0, 3.0, 5.0];
    let fit = linalg::lstsq(&design, &targets, 3, 2)?;
    println!(
        "least squares    intercept = {:.6}, slope = {:.6}",
        fit[0], fit[1]
    );

    let factorisation = Qr::factor(&design, 3, 2)?;
    let q = factorisation.q();
    println!("‖QᵀQ - I‖_∞      = {:.3e}", orthogonality(&q, 3, 2));

    // --- Cholesky on a symmetric positive definite matrix ------------------
    // A = [[4, 2], [2, 3]]  →  L = [[2, 0], [1, √2]]
    let spd = [4.0, 2.0, 2.0, 3.0];
    let cholesky = Cholesky::factor(&spd, 2)?;
    println!("Cholesky solve   x = {:?}", cholesky.solve(&[6.0, 5.0])?);

    // --- Symmetric eigen-decomposition -------------------------------------
    let symmetric = [2.0, 1.0, 1.0, 2.0];
    let eigen::Eigen { values, vectors } = eigen::symmetric(&symmetric, 2)?;
    println!("eigenvalues      λ = {values:?}");
    println!("eigenvectors     first = {:?}", &vectors[0][..]);

    // --- A singular matrix is reported, not hidden -------------------------
    let singular = Matrix::new(&[1.0, 2.0, 2.0, 4.0], 2, 2)?;
    match linalg::solve(singular.as_slice(), &[1.0, 2.0], 2) {
        Err(error) => println!("singular input   {error}"),
        Ok(solution) => panic!("expected a singular error, got {solution:?}"),
    }
    Ok(())
}

/// `‖QᵀQ - I‖_∞`, with `Q` stored column-major as returned by `Qr::q`.
fn orthogonality(q: &[f64], rows: usize, cols: usize) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..cols {
        for j in 0..cols {
            let mut dot = 0.0;
            for k in 0..rows {
                dot += q[i * rows + k] * q[j * rows + k];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - expected).abs());
        }
    }
    worst
}
