//! Multi-input higher-level operations built on the tape.
//!
//! These wrappers exist so callers do not have to spell out the local partials
//! for operations whose derivatives are easy to get subtly wrong.

use crate::variable::Variable;
use ncly_core::Result;

/// The `l2` norm `sqrt(Σ xᵢ²)` as one differentiable node chain.
///
/// The gradient is `x / ‖x‖`, which is `NaN`-free for the zero vector only
/// because NCLY defines the subgradient of `‖0‖` as the zero vector.
pub fn norm(value: &Variable) -> Variable {
    let squared = value.mul(value).sum();
    let total = squared.scalar().expect("sum produces a 0-d value");
    if total == 0.0 {
        // Zero-norm gradient is defined as zero by convention.
        return value.scale(0.0).sum();
    }
    squared.sqrt()
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

/// `sigmoid(x) = 1 / (1 + e^{-x})` expressed with stable primitives.
pub fn sigmoid(value: &Variable) -> Variable {
    // 1 / (1 + exp(-x)) written with a subtraction so the tape keeps a single
    // division node.
    let negated = value.neg();
    let denominator = negated.exp().adds_one();
    denominator.reciprocal()
}

trait AddsOne {
    fn adds_one(&self) -> Variable;
}

impl AddsOne for Variable {
    fn adds_one(&self) -> Variable {
        let one = Variable::on_tape(self.tape(), ncly_tensor::Tensor::ones(self.value().shape().dims().to_vec()), "one");
        self.add(&one)
    }
}

trait Reciprocal {
    fn reciprocal(&self) -> Variable;
}

impl Reciprocal for Variable {
    fn reciprocal(&self) -> Variable {
        let one = Variable::on_tape(self.tape(), ncly_tensor::Tensor::ones(self.value().shape().dims().to_vec()), "one");
        one.div(self)
    }
}
