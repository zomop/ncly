//! Linear system solvers and least squares.

use crate::check_dims;
use crate::lu::Lu;
use crate::qr::Qr;
use ncly_core::Result;

/// Solve `A·x = b` for a dense `n × n` row-major `A`.
///
/// This is a convenience wrapper over [`Lu::factor`] + [`Lu::solve`]; call those
/// directly when solving several right-hand sides against the same matrix.
pub fn solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    Lu::factor(a, n)?.solve(b)
}

/// Invert a dense `n × n` row-major matrix.
pub fn inverse(a: &[f64], n: usize) -> Result<Vec<f64>> {
    Lu::factor(a, n)?.inverse()
}

/// Solve the linear least squares problem `min ‖Ax - b‖₂` for `m ≥ n`.
///
/// Uses the QR factorisation: with `A = QR`, the solution solves `R x = Qᵀb`
/// restricted to the first `n` rows. This is the numerically stable choice for
/// overdetermined systems; forming the normal equations would square the
/// condition number.
///
/// # Errors
///
/// * [`ncly_core::NclyError::InvalidShape`] when `a` is not `m × n` or `b` is
///   not length `m`.
pub fn lstsq(a: &[f64], b: &[f64], m: usize, n: usize) -> Result<Vec<f64>> {
    check_dims("lstsq", a, m, n)?;
    if b.len() != m {
        return Err(ncly_core::NclyError::invalid_shape(
            "lstsq",
            format!("right-hand side has {} entries, expected {m}", b.len()),
        ));
    }
    let qr = Qr::factor(a, m, n)?;
    let rotated = qr.apply_qt(b)?;
    // Back substitution on the triangular R (upper n × n block).
    let r = qr.r_thin();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut total = rotated[i];
        for j in i + 1..n {
            total -= r[i * n + j] * x[j];
        }
        let diagonal = r[i * n + i];
        if diagonal == 0.0 {
            return Err(ncly_core::NclyError::Singular {
                op: "lstsq",
                pivot: i,
            });
        }
        x[i] = total / diagonal;
    }
    Ok(x)
}

/// Residual `‖Ax - b‖₂` of a candidate solution.
pub fn residual(a: &[f64], x: &[f64], b: &[f64], m: usize, n: usize) -> Result<f64> {
    check_dims("residual", a, m, n)?;
    if x.len() != n || b.len() != m {
        return Err(ncly_core::NclyError::invalid_shape(
            "residual",
            "vector lengths do not match the matrix",
        ));
    }
    let mut total = 0.0;
    for i in 0..m {
        let mut row = 0.0;
        for j in 0..n {
            row += a[i * n + j] * x[j];
        }
        let diff = row - b[i];
        total += diff * diff;
    }
    Ok(total.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_recovers_exact_solution() {
        let a = [2.0, 1.0, -1.0, 1.0, 3.0, 2.0, 1.0, -1.0, 4.0];
        let x_true = [1.0, 2.0, 3.0];
        let mut b = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                b[i] += a[i * 3 + j] * x_true[j];
            }
        }
        let x = solve(&a, &b, 3).unwrap();
        for (got, want) in x.iter().zip(&x_true) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn lstsq_fits_a_line_exactly() {
        // Points (0,1), (1,3), (2,5), (3,7) lie on y = 2x + 1.
        let m = 4;
        let n = 2;
        let mut a = vec![0.0; m * n];
        for i in 0..m {
            a[i * n] = 1.0;
            a[i * n + 1] = i as f64;
        }
        let b = [1.0, 3.0, 5.0, 7.0];
        let x = lstsq(&a, &b, m, n).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12, "intercept {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-12, "slope {}", x[1]);
        assert!(residual(&a, &x, &b, m, n).unwrap() < 1e-12);
    }

    #[test]
    fn lstsq_minimises_residual() {
        // Noisy line fit: the normal-equation solution is the reference.
        let m = 6;
        let n = 2;
        let mut a = vec![0.0; m * n];
        for i in 0..m {
            a[i * n] = 1.0;
            a[i * n + 1] = i as f64;
        }
        let b = [1.1, 2.9, 5.2, 6.8, 9.1, 11.0];
        let x = lstsq(&a, &b, m, n).unwrap();

        // Normal equations: (AᵀA) x = Aᵀb, solved independently.
        let mut ata = vec![0.0; n * n];
        let mut atb = vec![0.0; n];
        for i in 0..m {
            for j in 0..n {
                atb[j] += a[i * n + j] * b[i];
                for k in 0..n {
                    ata[j * n + k] += a[i * n + j] * a[i * n + k];
                }
            }
        }
        let reference = Lu::factor(&ata, n).unwrap().solve(&atb).unwrap();
        for (got, want) in x.iter().zip(&reference) {
            assert!((got - want).abs() < 1e-10, "{got} vs {want}");
        }
    }

    #[test]
    fn inverse_roundtrip() {
        let a = [4.0, 7.0, 2.0, 6.0];
        let inv = inverse(&a, 2).unwrap();
        let product = [
            a[0] * inv[0] + a[1] * inv[2],
            a[0] * inv[1] + a[1] * inv[3],
            a[2] * inv[0] + a[3] * inv[2],
            a[2] * inv[1] + a[3] * inv[3],
        ];
        assert!((product[0] - 1.0).abs() < 1e-12);
        assert!(product[1].abs() < 1e-12);
        assert!(product[2].abs() < 1e-12);
        assert!((product[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn shape_errors_are_reported() {
        assert!(lstsq(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2, 2).is_err());
        assert!(lstsq(&[1.0, 0.0, 0.0, 1.0], &[1.0], 2, 2).is_err());
        assert!(solve(&[1.0, 2.0], &[1.0, 2.0], 2).is_err());
    }
}
