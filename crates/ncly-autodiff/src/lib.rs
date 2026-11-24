//! Automatic differentiation for NCLY.
//!
//! Two complementary engines live here.
//!
//! # Forward mode — [`Dual`]
//!
//! A dual number carries a value and a tangent:
//!
//! ```text
//! x = a + b·ε    with ε² = 0
//! ```
//!
//! All the rules follow from that algebra: `(a + bε)(c + dε) = ac + (ad + bc)ε`
//! because the `bdε²` term vanishes. Forward mode computes a
//! Jacobian-vector product `J·v` for the cost of one evaluation, which makes it
//! the right tool when the input dimension is small (or when differentiating
//! with respect to a single direction).
//!
//! ```
//! use ncly_autodiff::Dual;
//!
//! // f(x) = sin(x^2) at x = 1.5 ; f'(x) = 2x·cos(x^2)
//! let x = Dual::variable(1.5);
//! let y = (x * x).sin();
//! assert!((y.tangent - 2.0 * 1.5 * (1.5f64 * 1.5).cos()).abs() < 1e-14);
//! ```
//!
//! # Reverse mode — [`Tape`] and [`Variable`]
//!
//! Reverse mode records every operation into a tape, then walks it backwards
//! accumulating adjoints. One backward sweep yields the full gradient at a cost
//! proportional to the forward pass, independent of the number of inputs.
//!
//! ```
//! use ncly_autodiff::Variable;
//!
//! let x = Variable::new(2.0, "x");
//! let y = Variable::new(3.0, "y");
//! let loss = (&x * &y).powi(2).sum();
//! loss.backward();
//!
//! // d/dx (xy)^2 = 2x y^2 = 36 ; d/dy = 2 x^2 y = 24
//! assert!((x.grad_scalar().unwrap() - 36.0).abs() < 1e-12);
//! assert!((y.grad_scalar().unwrap() - 24.0).abs() < 1e-12);
//! ```
//!
//! # Correctness strategy
//!
//! Every rule is validated against central finite differences in
//! `tests/gradient_check.rs`, and against the forward-mode engine (which shares
//! no code with the reverse pass) in the unit tests. See `docs/autodiff.md`.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

mod dual;
pub mod ops;
mod variable;

pub use dual::Dual;
pub use variable::{LocalGrad, Tape, TapeGuard, Variable, VariableNode};
