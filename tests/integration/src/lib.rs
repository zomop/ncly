//! Shared helpers for NCLY integration tests.
//!
//! The helpers here are intentionally dependency-free (no `rand`, no `approx`)
//! so that the validation suite measures NCLY against arithmetic it can audit,
//! not against a third-party reference implementation.

#![deny(missing_docs)]

/// Deterministic pseudo-random `f64` stream (SplitMix64 → uniform in `[0, 1)`).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[low, high)`.
    pub fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    /// Standard normal sample (Box–Muller, one value per call).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Maximum absolute difference between two equal-length slices.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "slices must have equal length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Maximum relative difference, treating values below 1 as absolute.
pub fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1.0))
        .fold(0.0, f64::max)
}

/// Step size for a central finite difference.
///
/// Total error for a central difference is `h^2 * |f3| / 6 + eps * |f| / h`; the
/// optimal balance therefore satisfies `h ~ (eps / |f3|)^(1/3) ~ 6e-6` for double
/// precision. Using the comparison tolerance in place of `eps` would give
/// `h ~ 0.05` for the usual `1e-4` tolerance, which leaves an `h^2` truncation
/// error *larger than the tolerance itself* -- a gradient check that fails for
/// the wrong reason.
pub fn fd_step(x: f64) -> f64 {
    const CBRT_EPS: f64 = 6.055_454_5e-6; // cube root of 2^-52, rounded up
    CBRT_EPS * x.abs().max(1.0)
}

/// Central finite difference of a scalar function.
///
/// The step is scaled to `|x|` so that both truncation and rounding error stay
/// proportional to the function scale.
pub fn central_difference(f: impl Fn(f64) -> f64, x: f64, _tolerance: f64) -> f64 {
    let h = fd_step(x);
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Assert `left ≈ right` element-wise within an absolute/relative tolerance.
pub fn assert_close(left: &[f64], right: &[f64], tolerance: f64, context: &str) {
    let diff = max_abs_diff(left, right);
    assert!(
        diff <= tolerance,
        "{context}: max |difference| = {diff:e} exceeds {tolerance:e}"
    );
}
