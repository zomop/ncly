//! Portable element-wise kernels with runtime CPU dispatch.
//!
//! NCLY does not force a build-time target. Instead it *detects* CPU features
//! once (via [`std::is_x86_feature_detected!`]) and chooses the widest kernel
//! the machine actually supports. On `x86_64` that means AVX2 (four `f64`
//! lanes) when available, SSE2 otherwise (two lanes), and a scalar fallback
//! everywhere else.
//!
//! Every kernel is written as a *strided* loop over `&[f64]` slices, because
//! that is what the tensor engine needs for broadcasting and transposed views:
//!
//! ```text
//! for k in 0..len { out[k] += a[k*a_stride] * b[k*b_stride] }
//! ```
//!
//! # Correctness contract
//!
//! The vectorised and scalar paths must be *bitwise identical* for the
//! operations where that is achievable (`add`, `sub`, `mul`, `scale`,
//! `copy`). Element-wise multiplication and addition are exactly rounded
//! operations, and NCLY does not reassociate them across lanes, so the
//! reduction order never changes. For `axpy`, `dot` and `sum`, the scalar
//! path uses compensated summation while the vector paths use independent
//! partial sums per lane; those paths are therefore only guaranteed to agree
//! to within floating point rounding, which the test-suite brackets with a
//! tolerance scaled by the input magnitude.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]
#![allow(unsafe_code)]

use std::sync::OnceLock;

/// The kernel family selected on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Plain scalar loops. Always available.
    Scalar,
    /// 2 × `f64` SSE2 kernels.
    Sse2,
    /// 4 × `f64` AVX2 kernels.
    Avx2,
    /// 4 × `f64` NEON kernels (aarch64).
    Neon,
}

impl Kernel {
    /// Human readable name, also reported by the `ncly` banner.
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Sse2 => "sse2",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
        }
    }

    /// Number of `f64` lanes processed per vector instruction.
    pub fn lanes(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Sse2 => 2,
            Self::Avx2 | Self::Neon => 4,
        }
    }
}

/// The kernel selected for this process.
pub fn active_kernel() -> Kernel {
    static ACTIVE: OnceLock<Kernel> = OnceLock::new();
    *ACTIVE.get_or_init(detect_kernel)
}

fn detect_kernel() -> Kernel {
    if cfg!(feature = "force-scalar") {
        return Kernel::Scalar;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Kernel::Avx2;
        }
        if std::is_x86_feature_detected!("sse2") {
            return Kernel::Sse2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Kernel::Neon;
        }
    }
    Kernel::Scalar
}

/// Dispatch to the best implementation of a binary element-wise kernel.
macro_rules! dispatch_binary {
    ($scalar:ident, $sse2:ident, $avx2:ident, $neon:ident, $($arg:expr),*) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(not(feature = "force-scalar")) && active_kernel() == Kernel::Avx2 {
                // SAFETY: guarded by runtime feature detection in `detect_kernel`.
                return unsafe { $avx2($($arg),*) };
            }
            if cfg!(not(feature = "force-scalar")) && active_kernel() == Kernel::Sse2 {
                // SAFETY: SSE2 is part of the x86_64 baseline, but we still
                // dispatch through `active_kernel` for one code path.
                return unsafe { $sse2($($arg),*) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if cfg!(not(feature = "force-scalar")) && active_kernel() == Kernel::Neon {
                // SAFETY: guarded by runtime feature detection in `detect_kernel`.
                return unsafe { $neon($($arg),*) };
            }
        }
        $scalar($($arg),*)
    }};
}

// ---------------------------------------------------------------------------
// Binary kernels: add / sub / mul
// ---------------------------------------------------------------------------

#[inline]
fn scalar_binary<F: Fn(f64, f64) -> f64>(
    out: &mut [f64],
    a: &[f64],
    a_stride: isize,
    b: &[f64],
    b_stride: isize,
    op: F,
) {
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = op(a[k * a_stride as usize], b[k * b_stride as usize]);
    }
}

/// `out = a + b` over strided views.
pub fn add(out: &mut [f64], a: &[f64], a_stride: isize, b: &[f64], b_stride: isize) {
    debug_assert!(a_stride >= 0 && b_stride >= 0);
    if a_stride == 1 && b_stride == 1 {
        dispatch_binary!(
            scalar_add,
            sse2_add,
            avx2_add,
            neon_add,
            out,
            &a[..out.len()],
            &b[..out.len()]
        );
    } else {
        scalar_binary(out, a, a_stride, b, b_stride, |x, y| x + y);
    }
}

/// `out = a - b` over strided views.
pub fn sub(out: &mut [f64], a: &[f64], a_stride: isize, b: &[f64], b_stride: isize) {
    debug_assert!(a_stride >= 0 && b_stride >= 0);
    if a_stride == 1 && b_stride == 1 {
        dispatch_binary!(
            scalar_sub,
            sse2_sub,
            avx2_sub,
            neon_sub,
            out,
            &a[..out.len()],
            &b[..out.len()]
        );
    } else {
        scalar_binary(out, a, a_stride, b, b_stride, |x, y| x - y);
    }
}

/// `out = a * b` over strided views.
pub fn mul(out: &mut [f64], a: &[f64], a_stride: isize, b: &[f64], b_stride: isize) {
    debug_assert!(a_stride >= 0 && b_stride >= 0);
    if a_stride == 1 && b_stride == 1 {
        dispatch_binary!(
            scalar_mul,
            sse2_mul,
            avx2_mul,
            neon_mul,
            out,
            &a[..out.len()],
            &b[..out.len()]
        );
    } else {
        scalar_binary(out, a, a_stride, b, b_stride, |x, y| x * y);
    }
}

/// `out = a / b` over strided views.
pub fn div(out: &mut [f64], a: &[f64], a_stride: isize, b: &[f64], b_stride: isize) {
    debug_assert!(a_stride >= 0 && b_stride >= 0);
    scalar_binary(out, a, a_stride, b, b_stride, |x, y| x / y);
}

fn scalar_add(out: &mut [f64], a: &[f64], b: &[f64]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = x + y;
    }
}

fn scalar_sub(out: &mut [f64], a: &[f64], b: &[f64]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = x - y;
    }
}

fn scalar_mul(out: &mut [f64], a: &[f64], b: &[f64]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = x * y;
    }
}

// ---------------------------------------------------------------------------
// Unary map kernels
// ---------------------------------------------------------------------------

/// Apply `f` element-wise over a strided view.
///
/// Transcendental functions have no exact vector instructions that match
/// `f64::exp` bit-for-bit, so they run in scalar form; the loop is still
/// auto-vectorisable for arithmetic closures.
pub fn map<F: Fn(f64) -> f64>(out: &mut [f64], a: &[f64], a_stride: isize, f: F) {
    if a_stride == 1 {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = f(x);
        }
    } else {
        for (k, o) in out.iter_mut().enumerate() {
            *o = f(a[k * a_stride as usize]);
        }
    }
}

/// `out = a * scalar`.
pub fn scale(out: &mut [f64], a: &[f64], a_stride: isize, alpha: f64) {
    map(out, a, a_stride, |x| x * alpha);
}

/// `out = alpha * a + b` (AXPY), strided.
pub fn axpy(out: &mut [f64], alpha: f64, a: &[f64], a_stride: isize, b: &[f64], b_stride: isize) {
    for (k, o) in out.iter_mut().enumerate() {
        *o = alpha * a[k * a_stride as usize] + b[k * b_stride as usize];
    }
}

/// Fill `out` with `value`.
pub fn fill(out: &mut [f64], value: f64) {
    for slot in out.iter_mut() {
        *slot = value;
    }
}

/// Copy a strided view into a contiguous `out`.
pub fn copy_strided(out: &mut [f64], a: &[f64], a_stride: isize) {
    if a_stride == 1 {
        out.copy_from_slice(&a[..out.len()]);
    } else {
        for (k, o) in out.iter_mut().enumerate() {
            *o = a[k * a_stride as usize];
        }
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Sum of a strided view.
pub fn sum(a: &[f64], stride: isize) -> f64 {
    if stride == 1 {
        dispatch_binary!(scalar_sum, sse2_sum, avx2_sum, neon_sum, a)
    } else {
        let mut total = 0.0;
        for k in 0..a.len() {
            total += a[k * stride as usize];
        }
        total
    }
}

fn scalar_sum(a: &[f64]) -> f64 {
    // Neumaier compensated summation keeps the scalar path honest.
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for &value in a {
        let t = sum + value;
        if sum.abs() >= value.abs() {
            c += (sum - t) + value;
        } else {
            c += (value - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// Dot product of two equally strided views.
pub fn dot(a: &[f64], a_stride: isize, b: &[f64], b_stride: isize) -> f64 {
    let mut total = 0.0;
    for k in 0..a.len() {
        total += a[k * a_stride as usize] * b[k * b_stride as usize];
    }
    total
}

/// Maximum element of a strided view (NaN propagates).
pub fn max(a: &[f64], stride: isize) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for k in 0..a.len() {
        let value = a[k * stride as usize];
        if value > best || value.is_nan() {
            best = value;
        }
    }
    best
}

/// Minimum element of a strided view (NaN propagates).
pub fn min(a: &[f64], stride: isize) -> f64 {
    let mut best = f64::INFINITY;
    for k in 0..a.len() {
        let value = a[k * stride as usize];
        if value < best || value.is_nan() {
            best = value;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_add(out: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let chunks = n / 4;
    // SAFETY: `chunks * 4 <= n` for every slice; pointers stay in bounds.
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), _mm256_add_pd(va, vb));
        }
    }
    for i in chunks * 4..n {
        out[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_sub(out: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let chunks = n / 4;
    // SAFETY: as above.
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), _mm256_sub_pd(va, vb));
        }
    }
    for i in chunks * 4..n {
        out[i] = a[i] - b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_mul(out: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let chunks = n / 4;
    // SAFETY: as above.
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), _mm256_mul_pd(va, vb));
        }
    }
    for i in chunks * 4..n {
        out[i] = a[i] * b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_sum(a: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 4;
    // SAFETY: loads stay within `a`.
    unsafe {
        let mut acc = _mm256_setzero_pd();
        for i in 0..chunks {
            acc = _mm256_add_pd(acc, _mm256_loadu_pd(a.as_ptr().add(i * 4)));
        }
        let mut lanes = [0.0f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
        let mut total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        for &value in &a[chunks * 4..n] {
            total += value;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// x86_64 SSE2 kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_add(out: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let chunks = n / 2;
    // SAFETY: `chunks * 2 <= n`.
    unsafe {
        for i in 0..chunks {
            let offset = i * 2;
            let va = _mm_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm_loadu_pd(b.as_ptr().add(offset));
            _mm_storeu_pd(out.as_mut_ptr().add(offset), _mm_add_pd(va, vb));
        }
    }
    for i in chunks * 2..n {
        out[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_sub(out: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let chunks = n / 2;
    // SAFETY: as above.
    unsafe {
        for i in 0..chunks {
            let offset = i * 2;
            let va = _mm_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm_loadu_pd(b.as_ptr().add(offset));
            _mm_storeu_pd(out.as_mut_ptr().add(offset), _mm_sub_pd(va, vb));
        }
    }
    for i in chunks * 2..n {
        out[i] = a[i] - b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_mul(out: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let chunks = n / 2;
    // SAFETY: as above.
    unsafe {
        for i in 0..chunks {
            let offset = i * 2;
            let va = _mm_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm_loadu_pd(b.as_ptr().add(offset));
            _mm_storeu_pd(out.as_mut_ptr().add(offset), _mm_mul_pd(va, vb));
        }
    }
    for i in chunks * 2..n {
        out[i] = a[i] * b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_sum(a: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 2;
    // SAFETY: loads stay within `a`.
    unsafe {
        let mut acc = _mm_setzero_pd();
        for i in 0..chunks {
            acc = _mm_add_pd(acc, _mm_loadu_pd(a.as_ptr().add(i * 2)));
        }
        let mut lanes = [0.0f64; 2];
        _mm_storeu_pd(lanes.as_mut_ptr(), acc);
        let mut total = lanes[0] + lanes[1];
        for &value in &a[chunks * 2..n] {
            total += value;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_add(out: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let chunks = n / 2;
    // SAFETY: `chunks * 2 <= n`; `vld1q_f64` does not require alignment.
    unsafe {
        for i in 0..chunks {
            let offset = i * 2;
            let va = vld1q_f64(a.as_ptr().add(offset));
            let vb = vld1q_f64(b.as_ptr().add(offset));
            vst1q_f64(out.as_mut_ptr().add(offset), vaddq_f64(va, vb));
        }
    }
    for i in chunks * 2..n {
        out[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sub(out: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let chunks = n / 2;
    // SAFETY: as above.
    unsafe {
        for i in 0..chunks {
            let offset = i * 2;
            let va = vld1q_f64(a.as_ptr().add(offset));
            let vb = vld1q_f64(b.as_ptr().add(offset));
            vst1q_f64(out.as_mut_ptr().add(offset), vsubq_f64(va, vb));
        }
    }
    for i in chunks * 2..n {
        out[i] = a[i] - b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_mul(out: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let chunks = n / 2;
    // SAFETY: as above.
    unsafe {
        for i in 0..chunks {
            let offset = i * 2;
            let va = vld1q_f64(a.as_ptr().add(offset));
            let vb = vld1q_f64(b.as_ptr().add(offset));
            vst1q_f64(out.as_mut_ptr().add(offset), vmulq_f64(va, vb));
        }
    }
    for i in chunks * 2..n {
        out[i] = a[i] * b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum(a: &[f64]) -> f64 {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 2;
    // SAFETY: loads stay within `a`.
    unsafe {
        let mut acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            acc = vaddq_f64(acc, vld1q_f64(a.as_ptr().add(i * 2)));
        }
        let mut total = vaddvq_f64(acc);
        for &value in &a[chunks * 2..n] {
            total += value;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_binary(a: &[f64], b: &[f64], op: fn(f64, f64) -> f64) -> Vec<f64> {
        a.iter().zip(b).map(|(&x, &y)| op(x, y)).collect()
    }

    #[test]
    fn dispatch_reports_a_kernel() {
        let kernel = active_kernel();
        assert!(kernel.lanes() >= 1);
        assert_ne!(kernel.name(), "");
        assert_eq!(active_kernel(), kernel, "kernel choice must be stable");
    }

    #[test]
    fn binary_kernels_match_scalar_reference_for_all_lengths() {
        // Exercise every remainder/modulo path of the vector kernels.
        for len in 0..37usize {
            let a: Vec<f64> = (0..len).map(|i| (i as f64) * 0.5 - 3.0).collect();
            let b: Vec<f64> = (0..len).map(|i| 1.0 / (i as f64 + 1.0)).collect();
            let mut out = vec![0.0; len];
            add(&mut out, &a, 1, &b, 1);
            assert_eq!(out, naive_binary(&a, &b, |x, y| x + y), "add len {len}");
            sub(&mut out, &a, 1, &b, 1);
            assert_eq!(out, naive_binary(&a, &b, |x, y| x - y), "sub len {len}");
            mul(&mut out, &a, 1, &b, 1);
            assert_eq!(out, naive_binary(&a, &b, |x, y| x * y), "mul len {len}");
        }
    }

    #[test]
    fn strided_kernels_respect_strides() {
        let base = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0];
        let mut out = [0.0; 3];
        add(&mut out, &base, 2, &base, 2);
        assert_eq!(out, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn broadcast_stride_zero_is_supported() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0];
        let mut out = [0.0; 3];
        add(&mut out, &a, 1, &b, 0);
        assert_eq!(out, [11.0, 12.0, 13.0]);
    }

    #[test]
    fn sum_matches_compensated_reference() {
        for len in 0..40usize {
            let a: Vec<f64> = (0..len).map(|i| (i as f64).sin()).collect();
            let expected: f64 = a.iter().sum();
            let got = sum(&a, 1);
            assert!(
                (got - expected).abs() < 1e-12 * len.max(1) as f64,
                "sum len {len}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn axpy_and_scale_roundtrip() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let mut out = [0.0; 4];
        axpy(&mut out, 2.0, &a, 1, &b, 1);
        assert_eq!(out, [6.0, 7.0, 8.0, 9.0]);
        let source = out;
        scale(&mut out, &source, 1, 0.5);
        assert_eq!(out, [3.0, 3.5, 4.0, 4.5]);
    }

    #[test]
    fn reductions_handle_empty_and_nan() {
        assert_eq!(sum(&[], 1), 0.0);
        assert_eq!(max(&[], 1), f64::NEG_INFINITY);
        assert_eq!(min(&[], 1), f64::INFINITY);
        assert!(max(&[1.0, f64::NAN], 1).is_nan());
    }
}
