//! Linear algebra validation: residuals, orthogonality, conditioning.
//!
//! These tests check *properties* that must hold for any correct
//! implementation, rather than hard-coded expected matrices:
//!
//! * `‖A - LU‖` small (factorisation is exact up to rounding),
//! * `‖A - QR‖` small and `‖QᵀQ - I‖` small,
//! * `‖Ax - b‖ / ‖b‖` at the level implied by the condition number,
//! * eigenvalues satisfy `‖Av - λv‖` small.

use ncly_integration_tests::{assert_close, max_abs_diff, Rng};
use ncly_linalg::{self as linalg, cholesky::Cholesky, eigen, lu::Lu, qr::Qr};

fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

fn multiply(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let scale = a[i * k + p];
            if scale == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += scale * b[p * n + j];
            }
        }
    }
    out
}

fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

/// Diagonally dominant random matrix, which is safely non-singular.
fn well_conditioned(n: usize, diagonal: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut a: Vec<f64> = (0..n * n).map(|_| rng.range(-1.0, 1.0)).collect();
    for i in 0..n {
        a[i * n + i] = diagonal;
    }
    a
}

#[test]
fn lu_factorisation_residual_and_solve_residual() {
    for n in [2usize, 3, 5, 8, 13] {
        let a = well_conditioned(n, n as f64 + 2.0, 100 + n as u64);
        let lu = Lu::factor(&a, n).unwrap();

        // P·A = L·U
        let product = multiply(&lu.lower(), &lu.upper(), n, n, n);
        let pa = multiply(&lu.permutation(), &a, n, n, n);
        assert!(
            max_abs_diff(&product, &pa) < 1e-10,
            "n={n}: PA - LU residual {}",
            max_abs_diff(&product, &pa)
        );

        // A·x = b with a known solution.
        let mut rng = Rng::new(7 + n as u64);
        let x_true: Vec<f64> = (0..n).map(|_| rng.range(-2.0, 2.0)).collect();
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i * n + j] * x_true[j];
            }
        }
        let x = lu.solve(&b).unwrap();
        let residual = {
            let ax = multiply(&a, &x, n, n, 1);
            max_abs_diff(&ax, &b)
        };
        assert!(residual < 1e-10, "n={n}: solve residual {residual:e}");
        assert!(max_abs_diff(&x, &x_true) < 1e-9, "n={n}: solution error");
    }
}

#[test]
fn qr_is_orthogonal_and_reconstructs() {
    for (m, n) in [(4usize, 4usize), (7, 4), (10, 3), (5, 5)] {
        let a = well_conditioned(m.max(n), 5.0, 200 + m as u64);
        let a: Vec<f64> = if m >= n {
            // Take the first n columns of an m×m-ish matrix.
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    out[i * n + j] = a[i * m.max(n) + j];
                }
            }
            out
        } else {
            a[..m * n].to_vec()
        };
        let qr = Qr::factor(&a, m, n).unwrap();
        let q = qr.q();
        let r = qr.r_thin();

        // QᵀQ = I_n
        let qt = transpose(&q, m, n);
        let qtq = multiply(&qt, &q, n, m, n);
        assert!(
            max_abs_diff(&qtq, &identity(n)) < 1e-12,
            "({m},{n}): orthogonality residual {}",
            max_abs_diff(&qtq, &identity(n))
        );

        // Q·R = A
        let reconstructed = multiply(&q, &r, m, n, n);
        assert!(
            max_abs_diff(&reconstructed, &a) < 1e-12,
            "({m},{n}): reconstruction residual {}",
            max_abs_diff(&reconstructed, &a)
        );

        // R must be upper triangular.
        for i in 0..n {
            for j in 0..i {
                assert_eq!(r[i * n + j], 0.0, "R[{i}][{j}] is not zero");
            }
        }
    }
}

#[test]
fn least_squares_matches_normal_equations() {
    let mut rng = Rng::new(303);
    let (m, n) = (30usize, 4usize);
    let a: Vec<f64> = (0..m * n).map(|_| rng.range(-1.0, 1.0)).collect();
    let true_coefficients = [0.5, -1.0, 2.0, 0.25];
    let b: Vec<f64> = (0..m)
        .map(|i| {
            let mut value = 0.0;
            for j in 0..n {
                value += a[i * n + j] * true_coefficients[j];
            }
            value + rng.range(-0.05, 0.05)
        })
        .collect();

    let solution = linalg::lstsq(&a, &b, m, n).unwrap();

    // Independent reference: normal equations solved with NCLY's LU (different
    // code path from the QR-based lstsq).
    let at = transpose(&a, m, n);
    let ata = multiply(&at, &a, n, m, n);
    let mut atb = vec![0.0; n];
    for j in 0..n {
        for i in 0..m {
            atb[j] += at[j * m + i] * b[i];
        }
    }
    let reference = Lu::factor(&ata, n).unwrap().solve(&atb).unwrap();
    assert_close(&solution, &reference, 1e-9, "lstsq vs normal equations");

    // The solution must not be worse than the true coefficients.
    let residual_solution = linalg::residual(&a, &solution, &b, m, n).unwrap();
    let residual_true = linalg::residual(&a, &true_coefficients, &b, m, n).unwrap();
    assert!(
        residual_solution <= residual_true + 1e-12,
        "solution residual {residual_solution} vs truth {residual_true}"
    );
}

#[test]
fn cholesky_matches_lu_on_positive_definite_matrices() {
    let n = 6;
    // A = MᵀM + n·I is symmetric positive definite by construction.
    let mut rng = Rng::new(404);
    let m_values: Vec<f64> = (0..n * n).map(|_| rng.range(-1.0, 1.0)).collect();
    let mt = transpose(&m_values, n, n);
    let mut a = multiply(&mt, &m_values, n, n, n);
    for i in 0..n {
        a[i * n + i] += n as f64;
    }

    let chol = Cholesky::factor(&a, n).unwrap();
    let l = chol.lower();
    let lt = transpose(l, n, n);
    let reconstructed = multiply(l, &lt, n, n, n);
    assert!(max_abs_diff(&reconstructed, &a) < 1e-10);

    // Solve with Cholesky and with LU; both must satisfy A·x = b.
    let b: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
    let x_chol = chol.solve(&b).unwrap();
    let x_lu = Lu::factor(&a, n).unwrap().solve(&b).unwrap();
    assert_close(&x_chol, &x_lu, 1e-9, "Cholesky vs LU solution");
    assert!(max_abs_diff(&multiply(&a, &x_chol, n, n, 1), &b) < 1e-10);
}

#[test]
fn symmetric_eigen_decomposition_is_accurate() {
    let n = 6;
    let mut rng = Rng::new(505);
    // Build a symmetric matrix from a random one: A = (M + Mᵀ)/2.
    let m_values: Vec<f64> = (0..n * n).map(|_| rng.range(-2.0, 2.0)).collect();
    let mt = transpose(&m_values, n, n);
    let a: Vec<f64> = m_values
        .iter()
        .zip(&mt)
        .map(|(x, y)| 0.5 * (x + y))
        .collect();

    let eigen::Eigen { values, vectors } = eigen::symmetric(&a, n).unwrap();
    assert_eq!(values.len(), n);

    // Every pair must satisfy A·v = λ·v and every v must be a unit vector.
    for (lambda, vector) in values.iter().zip(&vectors) {
        let av = multiply(&a, vector, n, n, 1);
        for i in 0..n {
            let expected = lambda * vector[i];
            assert!(
                (av[i] - expected).abs() < 1e-9,
                "eigenpair residual at {i}: {} vs {expected}",
                av[i]
            );
        }
        let norm: f64 = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12, "eigenvector norm {norm}");
    }

    // Eigenvectors of a symmetric matrix are orthogonal.
    for i in 0..n {
        for j in i + 1..n {
            let dot: f64 = vectors[i].iter().zip(&vectors[j]).map(|(a, b)| a * b).sum();
            assert!(
                dot.abs() < 1e-9,
                "eigenvectors {i}, {j} not orthogonal: {dot}"
            );
        }
    }

    // The trace equals the sum of eigenvalues.
    let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
    let sum: f64 = values.iter().sum();
    assert!((trace - sum).abs() < 1e-9, "trace {trace} vs sum {sum}");
}

#[test]
fn singular_and_ill_conditioned_inputs_are_reported_not_hidden() {
    // Exactly singular.
    let singular = [1.0, 2.0, 2.0, 4.0];
    assert!(Lu::factor(&singular, 2).is_err());

    // Badly conditioned but not singular: Hilbert matrix. NCLY must still
    // factor it (pivots are non-zero) but the residual grows with the
    // condition number, which is the honest behaviour.
    let n = 6;
    let hilbert: Vec<f64> = (0..n)
        .flat_map(|i| (0..n).map(move |j| 1.0 / (i + j + 1) as f64))
        .collect();
    let lu = Lu::factor(&hilbert, n).unwrap();
    let product = multiply(&lu.lower(), &lu.upper(), n, n, n);
    let pa = multiply(&lu.permutation(), &hilbert, n, n, n);
    // κ(H_6) ≈ 1.5e7, so a relative residual near 1e-16·κ ≈ 1e-9 is expected.
    let residual = max_abs_diff(&product, &pa);
    assert!(residual < 1e-8, "Hilbert residual {residual:e}");
}

#[test]
fn determinant_signs_and_magnitudes() {
    // det(diag(2,3,4)) = 24
    let diagonal = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
    let lu = Lu::factor(&diagonal, 3).unwrap();
    assert!((lu.determinant() - 24.0).abs() < 1e-12);

    // A single row swap flips the sign.
    let swapped = [0.0, 1.0, 1.0, 0.0];
    let lu = Lu::factor(&swapped, 2).unwrap();
    assert!((lu.determinant() + 1.0).abs() < 1e-12);

    // Large determinants must not overflow.
    let big = [1e200, 0.0, 0.0, 1e200];
    let lu = Lu::factor(&big, 2).unwrap();
    assert!(
        lu.determinant().is_infinite(),
        "1e400 overflows to infinity"
    );
    let (log_abs, sign) = lu.log_determinant();
    assert!((log_abs - 2.0 * 1e200f64.ln()).abs() < 1e-9);
    assert_eq!(sign, 1.0);
}
