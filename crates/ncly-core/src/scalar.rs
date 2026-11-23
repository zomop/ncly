//! Scalar helpers shared by every numerical algorithm.
//!
//! Numerical code needs a handful of decisions that are easy to get subtly
//! wrong — choosing a pivot tolerance, deciding when a residual is "good
//! enough", testing convergence. Centralising them keeps the behaviour of
//! `ncly-linalg` consistent and gives users a single place to tune.

/// Relative tolerance used when deciding whether a pivot is numerically zero.
pub const DEFAULT_PIVOT_TOLERANCE: f64 = 1e-12;

/// Relative tolerance used by iterative solvers.
pub const DEFAULT_CONVERGENCE_TOLERANCE: f64 = 1e-10;

/// Maximum iterations allowed by iterative solvers before reporting failure.
pub const DEFAULT_MAX_ITERATIONS: usize = 10_000;

/// Machine epsilon for `f64`.
#[must_use]
pub fn eps() -> f64 {
    f64::EPSILON
}

/// `|a|` when it is finite, otherwise `0.0`.
#[must_use]
pub fn finite_or_zero(a: f64) -> f64 {
    if a.is_finite() {
        a
    } else {
        0.0
    }
}

/// Scale-aware test for "this value is a zero pivot".
///
/// A pivot is rejected when `|pivot| <= tol * scale` where `scale` is the
/// magnitude of the largest entry seen so far. Using the matrix scale rather
/// than an absolute epsilon makes the test invariant to unit changes.
#[must_use]
pub fn is_negligible(pivot: f64, scale: f64, tol: f64) -> bool {
    if pivot == 0.0 {
        return true;
    }
    pivot.abs() <= tol * scale.max(f64::MIN_POSITIVE)
}

/// A stable, compensated sum (Neumaier variant).
///
/// Naive summation accumulates `O(n ε)` error; for long inner products this
/// matters for correctness tests against reference values.
#[must_use]
pub fn compensated_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;
    for value in values {
        let t = sum + value;
        if sum.abs() >= value.abs() {
            compensation += (sum - t) + value;
        } else {
            compensation += (value - t) + sum;
        }
        sum = t;
    }
    sum + compensation
}

/// Dot product with compensated accumulation.
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    compensated_sum(a.iter().zip(b).map(|(x, y)| x * y))
}

/// `(max_abs, index of max_abs)` over a slice, or `None` when empty.
#[must_use]
pub fn max_abs(values: &[f64]) -> Option<(f64, usize)> {
    let mut best: Option<(f64, usize)> = None;
    for (index, &value) in values.iter().enumerate() {
        let magnitude = value.abs();
        if magnitude.is_nan() {
            return Some((f64::NAN, index));
        }
        match best {
            Some((current, _)) if current >= magnitude => {}
            _ => best = Some((magnitude, index)),
        }
    }
    best
}

/// The `2`-norm of a slice.
#[must_use]
pub fn norm2(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

/// True when `sqrt` of the given sum of squares is stable without overflow.
///
/// Values whose squares would overflow are rescaled first; this keeps
/// [`norm2`] finite for inputs near `f64::MAX`.
#[must_use]
pub fn norm2_scaled(values: &[f64]) -> f64 {
    let scale = values.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let scaled = values.iter().map(|v| v / scale);
    scale * dot_iter(scaled)
}

fn dot_iter<I: Iterator<Item = f64>>(values: I) -> f64 {
    values.map(|v| v * v).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_sum_beats_naive() {
        let values = [1e100, 1.0, -1e100];
        let naive: f64 = values.iter().sum();
        assert_eq!(naive, 0.0);
        assert_eq!(compensated_sum(values), 1.0);
    }

    #[test]
    fn negligible_pivots_respect_scale() {
        assert!(is_negligible(0.0, 1.0, DEFAULT_PIVOT_TOLERANCE));
        assert!(is_negligible(1e-14, 1.0, 1e-12));
        assert!(!is_negligible(1e-6, 1.0, 1e-12));
        // Same relative magnitude, much larger scale.
        assert!(!is_negligible(1e-3, 1e9, 1e-12));
    }

    #[test]
    fn scaled_norm_handles_overflow() {
        assert!(norm2_scaled(&[1e300, 1e300]).is_finite());
        assert!((norm2_scaled(&[3.0, 4.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn max_abs_tracks_index() {
        assert_eq!(max_abs(&[1.0, -5.0, 3.0]), Some((5.0, 1)));
        assert_eq!(max_abs(&[]), None);
    }
}
