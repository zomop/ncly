//! Symmetric and general eigenvalue algorithms.
//!
//! Two entry points:
//!
//! * [`symmetric`] — eigenvalues *and* eigenvectors of a real symmetric matrix
//!   via the cyclic Jacobi method. Jacobi is slower than tridiagonalisation +
//!   implicit QL for large `n` (`O(n³)` with a larger constant) but it is
//!   unconditionally stable, needs no shift heuristics, and delivers very
//!   accurate eigenvectors for graded matrices — the property that makes it the
//!   right default for a first implementation.
//! * [`eigenvalues`] — eigenvalues of a general real matrix via Hessenberg
//!   reduction followed by the shifted QR iteration. Eigenvalues are returned as
//!   `(real, imaginary)` pairs, in no particular order.
//!
//! # Accuracy
//!
//! Jacobi reduces the off-diagonal Frobenius norm to below
//! `tol · ‖A‖_F`. For a symmetric matrix, the backward error of each computed
//! pair is `O(tol)`, so `A v - λ v` is small to the requested tolerance.
//! The QR iteration converges quadratically near the end; deflation happens as
//! soon as a subdiagonal entry is negligible relative to its neighbours.

use ncly_core::scalar::DEFAULT_PIVOT_TOLERANCE;
use ncly_core::{NclyError, Result};
use std::fmt;

/// The result of a symmetric eigen decomposition.
#[derive(Debug, Clone)]
pub struct Eigen {
    /// Eigenvalues, ascending.
    pub values: Vec<f64>,
    /// Eigenvectors, stored **column-major by eigenvector**: `vectors[j]` is the
    /// eigenvector belonging to `values[j]`, as a length-`n` slice.
    pub vectors: Vec<Vec<f64>>,
}

impl Eigen {
    /// Dimension.
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// Reconstruct `A = V Λ Vᵀ` in row-major form (useful for tests).
    pub fn reconstruct(&self) -> Vec<f64> {
        let n = self.size();
        let mut out = vec![0.0; n * n];
        for j in 0..n {
            let lambda = self.values[j];
            let vector = &self.vectors[j];
            for i in 0..n {
                for k in 0..n {
                    out[i * n + k] += lambda * vector[i] * vector[k];
                }
            }
        }
        out
    }
}

/// Relative symmetry check for a dense row-major matrix.
///
/// The Jacobi iteration assumes symmetry; without this check an asymmetric input
/// would silently converge to the eigenvalues of its symmetric part.
fn check_symmetric(a: &[f64], n: usize) -> Result<()> {
    for i in 0..n {
        for j in i + 1..n {
            let lhs = a[i * n + j];
            let rhs = a[j * n + i];
            let scale = lhs.abs().max(rhs.abs()).max(1.0);
            if (lhs - rhs).abs() > 1e-12 * scale {
                return Err(NclyError::invalid_shape(
                    "eigen::symmetric",
                    format!("matrix is not symmetric at ({i}, {j})"),
                ));
            }
        }
    }
    Ok(())
}

/// Frobenius norm of the strictly upper triangle: the quantity a Jacobi sweep
/// drives to zero.
fn off_diagonal_norm(m: &[f64], n: usize) -> f64 {
    let mut total = 0.0f64;
    for i in 0..n {
        for j in i + 1..n {
            total += m[i * n + j] * m[i * n + j];
        }
    }
    total.sqrt()
}

/// Apply the plane rotation `G(p, q, θ)` to `A` and accumulate it into `V`.
///
/// The row pass computes `A·G` and the column pass `Gᵀ·(A·G)`; `V` accumulates
/// `V·G` so that, at convergence, `A = V Λ Vᵀ`.
fn apply_rotation(m: &mut [f64], v: &mut [f64], n: usize, p: usize, q: usize, c: f64, s: f64) {
    for k in 0..n {
        let mkp = m[k * n + p];
        let mkq = m[k * n + q];
        m[k * n + p] = c * mkp - s * mkq;
        m[k * n + q] = s * mkp + c * mkq;
    }
    for k in 0..n {
        let mpk = m[p * n + k];
        let mqk = m[q * n + k];
        m[p * n + k] = c * mpk - s * mqk;
        m[q * n + k] = s * mpk + c * mqk;
    }
    for k in 0..n {
        let vkp = v[k * n + p];
        let vkq = v[k * n + q];
        v[k * n + p] = c * vkp - s * vkq;
        v[k * n + q] = s * vkp + c * vkq;
    }
}

/// Cyclic Jacobi eigen decomposition of a real symmetric matrix.
///
/// # Errors
///
/// Returns [`NclyError::Convergence`] when the sweep budget is exhausted, and
/// [`NclyError::InvalidShape`] when `a` is not `n × n`.
pub fn symmetric(a: &[f64], n: usize) -> Result<Eigen> {
    symmetric_with_tolerance(a, n, DEFAULT_PIVOT_TOLERANCE)
}

/// [`symmetric`] with an explicit relative tolerance.
pub fn symmetric_with_tolerance(a: &[f64], n: usize, tolerance: f64) -> Result<Eigen> {
    crate::check_dims("eigen::symmetric", a, n, n)?;
    if n == 0 {
        return Ok(Eigen {
            values: Vec::new(),
            vectors: Vec::new(),
        });
    }
    check_symmetric(a, n)?;

    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let frobenius = {
        let mut total = 0.0;
        for value in &m {
            total += value * value;
        }
        total.sqrt()
    };
    let target = tolerance * frobenius.max(f64::MIN_POSITIVE);

    let max_sweeps = 100 * n.max(1);
    let mut converged = false;
    for _sweep in 0..max_sweeps {
        let off = off_diagonal_norm(&m, n);
        if off <= target {
            converged = true;
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                // Rotation that zeroes the (p, q) entry. With
                //     G = [[c, s], [-s, c]]
                // the updated entry is
                //     a'_pq = a_pq (c² - s²) + (a_pp - a_qq) c s,
                // which vanishes for
                //     θ = ½ atan2(2 a_pq, a_qq - a_pp).
                // The sign of the second argument matters: using a_pp - a_qq
                // rotates the wrong way and the sweep does not converge.
                let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
                let (s, c) = theta.sin_cos();

                apply_rotation(&mut m, &mut v, n, p, q, c, s);
            }
        }
    }
    if !converged {
        // A single extra measurement is cheap and makes the error actionable.
        let mut off = 0.0f64;
        for i in 0..n {
            for j in i + 1..n {
                off += m[i * n + j] * m[i * n + j];
            }
        }
        return Err(NclyError::Convergence {
            op: "eigen::symmetric",
            iterations: max_sweeps,
            residual: Some(off.sqrt()),
        });
    }

    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|j| {
            let value = m[j * n + j];
            let mut vector = Vec::with_capacity(n);
            for i in 0..n {
                vector.push(v[i * n + j]);
            }
            // Fix the sign so results are deterministic across runs.
            if let Some(&first) = vector.iter().find(|&&x| x != 0.0) {
                if first < 0.0 {
                    for entry in &mut vector {
                        *entry = -*entry;
                    }
                }
            }
            (value, vector)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let values = pairs.iter().map(|(value, _)| *value).collect();
    let vectors = pairs.into_iter().map(|(_, vector)| vector).collect();
    Ok(Eigen { values, vectors })
}

impl fmt::Display for Eigen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eigen(values={:?})", self.values)
    }
}

/// Eigenvalues of a general real matrix as `(real, imaginary)` pairs.
///
/// The matrix is reduced to upper Hessenberg form by Householder similarity
/// transformations, then the implicitly shifted QR iteration deflates it. The
/// shift is Wilkinson's (the trailing 2×2 eigenvalue closest to the corner), which
/// gives local cubic convergence for real eigenvalues.
///
/// Real matrices may have complex conjugate pairs; those are reported with
/// matching imaginary parts of opposite sign. A complex pair is detected when the
/// trailing 2×2 block is irreducible, and the block is then solved directly rather
/// than iterated on — a real shift cannot separate a complex pair, so iterating
/// would not converge.
///
/// # Errors
///
/// Returns [`NclyError::Convergence`] when the iteration budget is exhausted.
pub fn eigenvalues(a: &[f64], n: usize) -> Result<Vec<(f64, f64)>> {
    crate::check_dims("eigen::eigenvalues", a, n, n)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut h = hessenberg(a, n);
    let mut eigenvalues = Vec::with_capacity(n);
    let max_iterations = 1000 * n.max(1);

    // Active window is [low, high]. Rows/columns above `high` have already
    // converged and are never touched again.
    let mut high = n as isize - 1;
    let mut iterations = 0usize;
    let scale = a.iter().fold(0.0f64, |acc, v| acc.max(v.abs())).max(1.0);

    while high >= 1 {
        let high_index = high as usize;
        // Look for a negligible subdiagonal entry to split the problem. The test
        // is on `h[i][i-1]` — the subdiagonal — because a Hessenberg matrix has
        // zeros everywhere below it.
        let mut low = high_index;
        while low > 0 {
            let sub = h[low * n + (low - 1)].abs();
            let diagonal = h[(low - 1) * n + (low - 1)].abs() + h[low * n + low].abs();
            if sub <= 1e-15 * diagonal.max(scale) {
                h[low * n + (low - 1)] = 0.0;
                break;
            }
            low -= 1;
        }

        if low == high_index {
            // 1×1 block: a real eigenvalue.
            eigenvalues.push((h[high_index * n + high_index], 0.0));
            high -= 1;
            continue;
        }
        if low == high_index - 1 {
            // 2×2 block: solve the quadratic directly, which is both faster and
            // more accurate than iterating on it.
            let (r1, i1, r2, i2) = two_by_two(&h, n, low);
            eigenvalues.push((r1, i1));
            eigenvalues.push((r2, i2));
            high -= 2;
            continue;
        }

        iterations += 1;
        if iterations > max_iterations {
            return Err(NclyError::Convergence {
                op: "eigen::eigenvalues",
                iterations,
                residual: None,
            });
        }
        // Wilkinson shift: the eigenvalue of the trailing 2×2 block closer to the
        // corner. For a complex pair there is no good real shift; the corner entry
        // is used, which keeps the iteration progressing (the 2×2 block then
        // decouples and is solved directly above).
        let (r1, i1, r2, _i2) = two_by_two(&h, n, high_index - 1);
        // `i1 != 0.0` means the trailing block has a complex conjugate pair.
        let corner = h[high_index * n + high_index];
        let shift = if i1 == 0.0 {
            if (r1 - corner).abs() < (r2 - corner).abs() {
                r1
            } else {
                r2
            }
        } else {
            corner
        };
        implicit_qr_step(&mut h, n, low, high_index, shift);
    }
    if high == 0 {
        eigenvalues.push((h[0], 0.0));
    }
    Ok(eigenvalues)
}

/// Householder reduction to upper Hessenberg form (no eigenvector accumulation).
fn hessenberg(a: &[f64], n: usize) -> Vec<f64> {
    let mut h = a.to_vec();
    for k in 0..n.saturating_sub(2) {
        let mut scale = 0.0f64;
        for i in k + 1..n {
            scale = scale.max(h[i * n + k].abs());
        }
        if scale == 0.0 {
            continue;
        }
        let mut norm = 0.0;
        for i in k + 1..n {
            let value = h[i * n + k] / scale;
            norm += value * value;
        }
        let mut vector: Vec<f64> = (k + 1..n).map(|i| h[i * n + k] / scale).collect();
        let norm = norm.sqrt();
        let alpha = vector[0];
        let beta = if alpha >= 0.0 { -norm } else { norm };
        vector[0] -= beta;
        let denominator: f64 = vector.iter().map(|v| v * v).sum();
        if denominator == 0.0 {
            continue;
        }
        // Apply P = I - 2 v vᵀ / (vᵀv) from both sides to the trailing block.
        for j in 0..n {
            let mut dot = 0.0;
            for (offset, &value) in vector.iter().enumerate() {
                dot += value * h[(k + 1 + offset) * n + j];
            }
            let factor = 2.0 * dot / denominator;
            for (offset, &value) in vector.iter().enumerate() {
                h[(k + 1 + offset) * n + j] -= factor * value;
            }
        }
        for i in 0..n {
            let mut dot = 0.0;
            for (offset, &value) in vector.iter().enumerate() {
                dot += value * h[i * n + k + 1 + offset];
            }
            let factor = 2.0 * dot / denominator;
            for (offset, &value) in vector.iter().enumerate() {
                h[i * n + k + 1 + offset] -= factor * value;
            }
        }
    }
    h
}

/// Eigenvalues of the 2×2 block whose top-left corner is `start`.
fn two_by_two(h: &[f64], n: usize, start: usize) -> (f64, f64, f64, f64) {
    let a = h[start * n + start];
    let b = h[start * n + start + 1];
    let c = h[(start + 1) * n + start];
    let d = h[(start + 1) * n + start + 1];
    // λ² - (a+d)λ + (ad - bc) = 0
    let trace = a + d;
    let det = a * d - b * c;
    let discriminant = trace * trace - 4.0 * det;
    if discriminant >= 0.0 {
        let root = discriminant.sqrt();
        // Numerically stable form of the quadratic formula: adding the root with
        // the *sign of the trace* avoids the cancellation that subtracting it
        // would cause when `trace` and `root` are close.
        //
        // The subtraction that follows is deliberate (`clippy::suboptimal_flops`
        // would rewrite it as an addition and flip the sign of the root).
        #[allow(clippy::suboptimal_flops)]
        let q = 0.5 * (trace + trace.signum() * root);
        let l1 = if q == 0.0 { 0.0 } else { q };
        // Vieta: the product of the roots is the determinant.
        let l2 = if q == 0.0 { 0.0 } else { det / q };
        (l1, 0.0, l2, 0.0)
    } else {
        let real = 0.5 * trace;
        let imaginary = 0.5 * (-discriminant).sqrt();
        (real, imaginary, real, -imaginary)
    }
}

/// One implicit-shifted QR step (Francis single shift) on the active window
/// `[low, high]` of an upper Hessenberg matrix.
///
/// The first plane rotation is chosen to annihilate `(h[low][low] - shift,
/// h[low+1][low])`, which is exactly the first column of `H - shift·I`. The
/// remaining rotations chase the resulting bulge down the subdiagonal. Because
/// every rotation is applied from both sides, the result is `Qᵀ H Q`, so the
/// eigenvalues are unchanged and the Hessenberg form is restored.
///
/// # Why the shift must enter here
///
/// Rotating on `(h[low][low], h[low+1][low])` *without* subtracting the shift
/// makes the first rotation the identity whenever the window starts on a
/// Hessenberg matrix, so the step does nothing at all and the iteration never
/// deflates. The shift is not an optimisation: it is what gives the step a
/// non-trivial first rotation.
fn implicit_qr_step(h: &mut [f64], n: usize, low: usize, high: usize, shift: f64) {
    let mut x = h[low * n + low] - shift;
    let mut y = h[(low + 1) * n + low];
    for k in low..high {
        let (c, s) = rotation(x, y);
        // Left: rows `k`, `k+1`, across the whole active block. Columns before
        // `low` are already zero in a Hessenberg matrix and must stay zero.
        for j in low..n {
            let upper = h[k * n + j];
            let lower = h[(k + 1) * n + j];
            h[k * n + j] = c * upper + s * lower;
            h[(k + 1) * n + j] = -s * upper + c * lower;
        }
        // Right: columns `k`, `k+1`, over the rows the bulge can reach. Rows
        // below `k+2` are zero there already; touching them would push mass back
        // into the subdiagonal this rotation just cleared.
        for i in 0..(k + 2).min(high) + 1 {
            let left = h[i * n + k];
            let right = h[i * n + k + 1];
            h[i * n + k] = c * left + s * right;
            h[i * n + k + 1] = -s * left + c * right;
        }
        // Next rotation annihilates the bulge this one created.
        if k + 2 <= high {
            x = h[(k + 1) * n + k];
            y = h[(k + 2) * n + k];
        }
    }
}

/// Givens rotation `(c, s)` with `c·f + s·g = r` and `-s·f + c·g = 0`.
///
/// The magnitude is normalised so the result cannot overflow; the signs of
/// `(f, g)` are irrelevant because the rotation is used on both sides of a
/// similarity transform.
fn rotation(f: f64, g: f64) -> (f64, f64) {
    if g == 0.0 {
        return (1.0, 0.0);
    }
    if f == 0.0 {
        return (0.0, 1.0);
    }
    let r = f.hypot(g);
    (f / r, g / r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric_from(values: &[f64]) -> Vec<f64> {
        values.to_vec()
    }

    #[test]
    fn known_symmetric_system() {
        // A = [[2, 1], [1, 2]] → λ = 1, 3
        let a = symmetric_from(&[2.0, 1.0, 1.0, 2.0]);
        let eigen = symmetric(&a, 2).unwrap();
        assert!((eigen.values[0] - 1.0).abs() < 1e-12);
        assert!((eigen.values[1] - 3.0).abs() < 1e-12);
        // Eigenvector for λ=1 is (1, -1)/√2.
        let v = &eigen.vectors[0];
        assert!((v[0].abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!((v[0] + v[1]).abs() < 1e-12);
    }

    #[test]
    fn diagonal_matrix_returns_its_diagonal_sorted() {
        let a = [3.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0];
        let eigen = symmetric(&a, 3).unwrap();
        assert_eq!(eigen.values, vec![-1.0, 2.0, 3.0]);
    }

    #[test]
    fn reconstruction_is_accurate() {
        let a = [
            4.0, 1.0, -2.0, 2.0, //
            1.0, 2.0, 0.0, 1.0, //
            -2.0, 0.0, 3.0, -2.0, //
            2.0, 1.0, -2.0, -1.0,
        ];
        let eigen = symmetric(&a, 4).unwrap();
        let rebuilt = eigen.reconstruct();
        for (got, want) in rebuilt.iter().zip(&a) {
            assert!((got - want).abs() < 1e-10, "{got} vs {want}");
        }
    }

    #[test]
    fn eigenvectors_satisfy_the_eigen_equation() {
        let a = [1.0, 2.0, 2.0, 1.0];
        let n = 2;
        let eigen = symmetric(&a, n).unwrap();
        for (lambda, vector) in eigen.values.iter().zip(&eigen.vectors) {
            for i in 0..n {
                let mut product = 0.0;
                for j in 0..n {
                    product += a[i * n + j] * vector[j];
                }
                assert!((product - lambda * vector[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn rejects_asymmetric_input() {
        let a = [1.0, 2.0, 0.0, 1.0];
        assert!(symmetric(&a, 2).is_err());
    }

    #[test]
    fn general_eigenvalues_of_triangular_matrix() {
        // Upper triangular: eigenvalues are the diagonal.
        let a = [2.0, 5.0, 1.0, 0.0, -3.0, 7.0, 0.0, 0.0, 4.0];
        let mut values = eigenvalues(&a, 3).unwrap();
        values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert!((values[0].0 + 3.0).abs() < 1e-9, "{values:?}");
        assert!((values[1].0 - 2.0).abs() < 1e-9, "{values:?}");
        assert!((values[2].0 - 4.0).abs() < 1e-9, "{values:?}");
    }

    #[test]
    fn general_eigenvalues_of_a_non_symmetric_random_matrix() {
        // A deterministic 5x5 matrix whose eigenvalues were computed
        // independently (real parts, sorted):
        // (numpy.linalg.eigvals, real parts):
        //   -2.196856, -2.196856, -1.846219, 0.640739, 1.710028
        // with a complex pair present.
        #[rustfmt::skip]
        #[allow(clippy::unreadable_literal)] // a matrix literal reads better ungrouped
        let a = [
            -0.704669, -1.396603,  0.603738, -1.710255,  0.143528,
            -0.537244, -1.768004,  0.029743, -1.850017, -0.265417,
            -1.720578, -1.637148, -0.301923,  1.307408, -1.504792,
            -1.107044,  0.509733,  1.790836,  0.308412, -0.413278,
             1.905020, -1.813669,  1.433874, -0.841563, -1.422980,
        ];
        let values = eigenvalues(&a, 5).unwrap();
        assert_eq!(values.len(), 5);
        let mut real: Vec<f64> = values.iter().map(|v| v.0).collect();
        real.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let expected = [-2.196_856, -2.196_856, -1.846_219, 0.640_739, 1.710_028];
        for (found, want) in real.iter().zip(expected) {
            assert!(
                (found - want).abs() < 1e-4,
                "found {real:?}, expected {expected:?}"
            );
        }
        // The trace must be reproduced exactly: the sum of the eigenvalues.
        let trace: f64 = (0..5).map(|i| a[i * 5 + i]).sum();
        assert!((real.iter().sum::<f64>() - trace).abs() < 1e-9);
        // At least one conjugate pair.
        assert!(values.iter().any(|v| v.1.abs() > 0.1), "{values:?}");
        // Conjugate pairs are reported in matching pairs.
        let imaginary: f64 = values.iter().map(|v| v.1).sum();
        assert!(
            imaginary.abs() < 1e-9,
            "imaginary parts must cancel: {values:?}"
        );
    }

    #[test]
    fn general_eigenvalues_include_complex_pairs() {
        // Rotation by 90°: eigenvalues ±i.
        let a = [0.0, -1.0, 1.0, 0.0];
        let values = eigenvalues(&a, 2).unwrap();
        assert_eq!(
            values.len(),
            2,
            "must return exactly n eigenvalues: {values:?}"
        );
        let mut imaginary: Vec<f64> = values.iter().map(|v| v.1.abs()).collect();
        imaginary.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((imaginary[0] - 1.0).abs() < 1e-9, "{values:?}");
        assert!((imaginary[1] - 1.0).abs() < 1e-9, "{values:?}");
        for value in &values {
            assert!(value.0.abs() < 1e-9);
        }
    }
}
