//! Forward-mode differentiation with dual numbers.
//!
//! A [`Dual`] is `v + d·ε` with `ε² = 0`. The implementation is a thin wrapper
//! over two `f64`s; every operator is `#[inline]` so chains of dual arithmetic
//! optimise to the same code as the underlying scalar expression.

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// A dual number: `value + tangent·ε`, where `ε² = 0`.
///
/// ```
/// use ncly_autodiff::Dual;
///
/// let x = Dual::variable(2.0);
/// let y = x.exp();
/// // d/dx e^x = e^x
/// assert!((y.tangent - y.value).abs() < 1e-15);
/// ```
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Dual {
    /// The primal value.
    pub value: f64,
    /// The derivative in the seeded direction.
    pub tangent: f64,
}

impl Dual {
    /// A constant: tangent zero.
    #[inline]
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            tangent: 0.0,
        }
    }

    /// An independent variable seeded with tangent 1.
    #[inline]
    #[must_use]
    pub fn variable(value: f64) -> Self {
        Self {
            value,
            tangent: 1.0,
        }
    }

    /// An independent variable with an explicit tangent.
    ///
    /// Use `tangent = 0` for variables you are not differentiating with respect
    /// to, which is what makes a single forward pass produce one column of the
    /// Jacobian.
    #[inline]
    #[must_use]
    pub fn with_tangent(value: f64, tangent: f64) -> Self {
        Self { value, tangent }
    }

    /// `e^self`.
    #[inline]
    #[must_use]
    pub fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            tangent: value * self.tangent,
        }
    }

    /// `ln(self)`.
    #[inline]
    #[must_use]
    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            tangent: self.tangent / self.value,
        }
    }

    /// `sqrt(self)`.
    #[inline]
    #[must_use]
    pub fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        Self {
            value,
            tangent: if value == 0.0 {
                f64::INFINITY * self.tangent
            } else {
                self.tangent / (2.0 * value)
            },
        }
    }

    /// `sin(self)`.
    #[inline]
    #[must_use]
    pub fn sin(self) -> Self {
        Self {
            value: self.value.sin(),
            tangent: self.tangent * self.value.cos(),
        }
    }

    /// `cos(self)`.
    #[inline]
    #[must_use]
    pub fn cos(self) -> Self {
        Self {
            value: self.value.cos(),
            tangent: -self.tangent * self.value.sin(),
        }
    }

    /// `tanh(self)`.
    #[inline]
    #[must_use]
    pub fn tanh(self) -> Self {
        let value = self.value.tanh();
        Self {
            value,
            tangent: self.tangent * (1.0 - value * value),
        }
    }

    /// `self^n` for integer `n`, computed by squaring.
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        let value = self.value.powi(n);
        Self {
            value,
            tangent: self.tangent * f64::from(n) * self.value.powi(n - 1),
        }
    }

    /// `self^p` for real `p`.
    #[inline]
    #[must_use]
    pub fn powf(self, p: f64) -> Self {
        Self {
            value: self.value.powf(p),
            tangent: self.tangent * p * self.value.powf(p - 1.0),
        }
    }

    /// `|self|`, convention `d|x|/dx = sign(x)` with `sign(0) = 0`.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        let sign = if self.value > 0.0 {
            1.0
        } else if self.value < 0.0 {
            -1.0
        } else {
            0.0
        };
        Self {
            value: self.value.abs(),
            tangent: self.tangent * sign,
        }
    }

    /// `1/self`.
    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        let value = 1.0 / self.value;
        Self {
            value,
            tangent: -self.tangent * value * value,
        }
    }

    /// The tangent, ignoring the primal value.
    #[inline]
    #[must_use]
    pub fn derivative(self) -> f64 {
        self.tangent
    }
}

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            tangent: self.tangent * rhs.value + self.value * rhs.tangent,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let value = self.value / rhs.value;
        Self {
            value,
            tangent: (self.tangent - value * rhs.tangent) / rhs.value,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            tangent: -self.tangent,
        }
    }
}

impl AddAssign for Dual {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
        self.tangent += rhs.tangent;
    }
}

/// Implement the four arithmetic operators between `Dual` and a primitive.
///
/// Every instantiation is for a type whose conversion to `f64` is lossless, so
/// the macro takes a conversion expression as its second parameter. Keeping the
/// conversion explicit is what lets `i64`/`u64`/`usize` participate without a
/// silent precision loss: those pass `<ty>::to_f64` semantics in the form of a
/// cast that is documented as approximate.
macro_rules! dual_scalar_ops {
    ($($ty:ty => $to:expr),* $(,)?) => {$(
        impl Add<$ty> for Dual {
            type Output = Dual;
            #[inline]
            fn add(self, rhs: $ty) -> Dual {
                let rhs: f64 = ($to)(rhs);
                Dual { value: self.value + rhs, tangent: self.tangent }
            }
        }
        impl Sub<$ty> for Dual {
            type Output = Dual;
            #[inline]
            fn sub(self, rhs: $ty) -> Dual {
                let rhs: f64 = ($to)(rhs);
                Dual { value: self.value - rhs, tangent: self.tangent }
            }
        }
        impl Mul<$ty> for Dual {
            type Output = Dual;
            #[inline]
            fn mul(self, rhs: $ty) -> Dual {
                let rhs: f64 = ($to)(rhs);
                Dual { value: self.value * rhs, tangent: self.tangent * rhs }
            }
        }
        impl Div<$ty> for Dual {
            type Output = Dual;
            #[inline]
            fn div(self, rhs: $ty) -> Dual {
                let rhs: f64 = ($to)(rhs);
                Dual { value: self.value / rhs, tangent: self.tangent / rhs }
            }
        }
        impl Add<Dual> for $ty {
            type Output = Dual;
            #[inline]
            fn add(self, rhs: Dual) -> Dual { rhs + self }
        }
        impl Mul<Dual> for $ty {
            type Output = Dual;
            #[inline]
            fn mul(self, rhs: Dual) -> Dual { rhs * self }
        }
    )*};
}

// Lossless: every value of these types is exactly representable in `f64`.
dual_scalar_ops!(f32 => f64::from, f64 => |v: f64| v, i32 => f64::from, u32 => f64::from);
// Not lossless in general (`i64`/`usize` exceed 2^53), so the cast is explicit
// and documented rather than hidden behind `From`.
dual_scalar_ops!(i64 => |v: i64| v as f64, u64 => |v: u64| v as f64, usize => |v: usize| v as f64);

impl fmt::Debug for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual({}, ẋ={})", self.value, self.tangent)
    }
}

impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.value, self.tangent)
    }
}

impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl From<Dual> for f64 {
    fn from(dual: Dual) -> Self {
        dual.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central finite difference of `f` at `x` with step `h`.
    fn central_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    fn check(f_dual: impl Fn(Dual) -> Dual, f: impl Fn(f64) -> f64, x: f64) {
        let dual = f_dual(Dual::variable(x));
        let numeric = central_diff(&f, x, 1e-6);
        assert!(
            (dual.value - f(x)).abs() < 1e-10,
            "value mismatch at {x}: {} vs {}",
            dual.value,
            f(x)
        );
        assert!(
            (dual.tangent - numeric).abs() < 1e-5f64.max(numeric.abs() * 1e-5),
            "derivative mismatch at {x}: {} vs {}",
            dual.tangent,
            numeric
        );
    }

    #[test]
    fn elementary_rules_match_finite_differences() {
        for x in [0.3, 1.0, 2.5, -1.7] {
            check(|d| d.exp(), |v| v.exp(), x);
            check(|d| d.sin(), |v| v.sin(), x);
            check(|d| d.cos(), |v| v.cos(), x);
            check(|d| d.tanh(), |v| v.tanh(), x);
            check(|d| d.abs(), |v| v.abs(), x);
            check(|d| d * d, |v| v * v, x);
            check(|d| d.powi(3), |v| v.powi(3), x);
        }
        for x in [0.3, 1.0, 2.5] {
            check(|d| d.ln(), |v| v.ln(), x);
            check(|d| d.sqrt(), |v| v.sqrt(), x);
            check(|d| d.recip(), |v| 1.0 / v, x);
            check(|d| d.powf(1.5), |v| v.powf(1.5), x);
        }
    }

    #[test]
    fn product_rule_includes_both_terms() {
        let x = Dual::variable(3.0);
        let y = Dual::with_tangent(4.0, 0.0);
        let z = x * y;
        assert_eq!(z.value, 12.0);
        assert_eq!(z.tangent, 4.0, "only the x-term contributes");
    }

    #[test]
    fn quotient_rule() {
        let x = Dual::variable(3.0);
        let y = Dual::constant(4.0);
        assert!(((x / y).tangent - 0.25).abs() < 1e-15);
    }

    #[test]
    fn chain_rule_composes() {
        // f(x) = sin(exp(x^2)) ; f'(x) = cos(exp(x^2)) * exp(x^2) * 2x
        let x = 0.5;
        let dual = (Dual::variable(x) * Dual::variable(x)).exp().sin();
        let expected = (x * x).exp().cos() * (x * x).exp() * 2.0 * x;
        assert!((dual.tangent - expected).abs() < 1e-12);
    }

    #[test]
    fn constants_have_zero_tangent() {
        let c = Dual::constant(5.0);
        assert_eq!((c * c).tangent, 0.0);
        assert_eq!(c.exp().tangent, 0.0);
    }

    #[test]
    fn scalar_mixing() {
        let x = Dual::variable(2.0);
        assert_eq!((x * 3.0f64).tangent, 3.0);
        assert_eq!((3.0f64 * x).tangent, 3.0);
        assert_eq!((x + 1i32).value, 3.0);
        assert_eq!((x / 2.0f64).tangent, 0.5);
    }

    #[test]
    fn abs_at_zero_is_zero_slope() {
        assert_eq!(Dual::variable(0.0).abs().tangent, 0.0);
    }
}
