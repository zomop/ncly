//! Multi-input higher-level operations built on the tape.
//!
//! These wrappers exist so callers do not have to spell out the local partials
//! for operations whose derivatives are easy to get subtly wrong.

use crate::variable::Variable;
use ncly_core::Result;

/// The `l2` norm `sqrt(Σ xᵢ²)`.
///
/// The gradient is `x / ‖x‖`, which is `NaN` at the origin; NCLY reports it as
/// `NaN` rather than silently inventing a subgradient, so callers can detect
/// the kink.
pub fn norm(value: &Variable) -> Variable {
    value.mul(value).sum().sqrt()
}

/// `log(Σ exp(xᵢ))` over all axes.
pub fn logsumexp(value: &Variable) -> Variable {
    value.logsumexp()
}

/// Mean squared error between two variables of identical shape.
pub fn mse(prediction: &Variable, target: &Variable) -> Result<Variable> {
    let diff = prediction.sub(target);
    let squared = diff.mul(&diff);
    Ok(squared.mean())
}

/// Element-wise `softplus(x) = ln(1 + e^x)`, computed in the log-sum-exp form to
/// avoid overflow for large `x`.
pub fn softplus(value: &Variable) -> Variable {
    value.logsumexp()
}

/// `sigmoid(x) = 1 / (1 + e^{-x})`.
pub fn sigmoid(value: &Variable) -> Variable {
    let one = Variable::on_tape(value.tape(), ncly_tensor::Tensor::scalar(1.0), "one");
    let denominator = one.add(&value.neg().exp());
    denominator.recip()
}
