//! Gradient checking: reverse mode and forward mode against finite differences.
//!
//! Three independent engines are compared:
//!
//! 1. **Finite differences** — central differences with a step sized for the
//!    function (`h = tol^(1/3) · max(1,|x|)`), which is the ground truth for
//!    *correctness*, as opposed to analytic beauty.
//! 2. **Reverse mode** — the tape in `ncly-autodiff`.
//! 3. **Forward mode** — dual numbers, which share no code with the tape.
//!
//! If all three agree, a bug would have to appear identically in an
//! accumulation-based algorithm and a propagation-based one, which is a strong
//! signal that the rule itself is wrong rather than its implementation.

use ncly_autodiff::{ops, Dual, Tape, Variable};
use ncly_tensor::Tensor;
use ncly_integration_tests::{assert_close, central_difference, max_abs_diff, Rng};

/// Tolerance for analytic-vs-analytic comparisons (bit-level agreement expected).
const EXACT: f64 = 1e-10;

/// Tolerance for finite-difference comparisons.
const NUMERIC: f64 = 1e-5;

#[test]
fn scalar_compositions_match_finite_differences() {
    // Each entry: (name, function of Dual, function of f64).
    let cases: Vec<(&str, fn(Dual) -> Dual, fn(f64) -> f64)> = vec![
        ("x^2 + 3x", |d| d * d + 3.0f64 * d, |x| x * x + 3.0 * x),
        ("sin(x)cos(x)", |d| d.sin() * d.cos(), |x| x.sin() * x.cos()),
        ("exp(-x^2)", |d| -(d * d).exp(), |x| -(x * x).exp()),
        ("ln(1+x^2)", |d| (1.0f64 + d * d).ln(), |x| (1.0 + x * x).ln()),
        (
            "tanh(x) + x^3",
            |d| d.tanh() + d.powi(3),
            |x| x.tanh() + x.powi(3),
        ),
        (
            "sqrt(1+x^2)",
            |d| (1.0f64 + d * d).sqrt(),
            |x| (1.0 + x * x).sqrt(),
        ),
    ];

    for (name, dual_fn, scalar_fn) in cases {
        for &x0 in &[0.4, 1.3, -0.7, 2.2] {
            let forward = dual_fn(Dual::variable(x0)).tangent;
            let numeric = central_difference(scalar_fn, x0, NUMERIC);

            // Reverse mode over the same expression, written with Variables.
            let tape = Tape::new();
            let guard = Variable::with_tape(&tape);
            let x = Variable::auto(x0, "x");
            let value = match name {
                "x^2 + 3x" => (&x * &x).add(&x.scale(3.0)),
                "sin(x)cos(x)" => x.sin().mul(&x.cos()),
                "exp(-x^2)" => x.mul(&x).neg().exp(),
                "ln(1+x^2)" => x.mul(&x).add(&Variable::auto(1.0, "one")).ln(),
                "tanh(x) + x^3" => x.tanh().add(&x.powi(3)),
                "sqrt(1+x^2)" => x.mul(&x).add(&Variable::auto(1.0, "one")).sqrt(),
                _ => unreachable!(),
            };
            value.sum().backward();
            let reverse = x.grad_scalar().unwrap();
            drop(guard);

            let scale = numeric.abs().max(1.0);
            assert!(
                (forward - numeric).abs() <= NUMERIC * scale,
                "{name} at {x0}: forward {forward:e} vs numeric {numeric:e}"
            );
            assert!(
                (reverse - numeric).abs() <= NUMERIC * scale,
                "{name} at {x0}: reverse {reverse:e} vs numeric {numeric:e}"
            );
            assert!(
                (forward - reverse).abs() <= EXACT * scale,
                "{name} at {x0}: forward {forward:e} vs reverse {reverse:e}"
            );
        }
    }
}

#[test]
fn multivariate_gradient_matches_finite_differences() {
    // f(x, y) = sin(x·y) + x²y - e^(x-y)
    // ∂f/∂x = y cos(xy) + 2xy - e^(x-y)
    // ∂f/∂y = x cos(xy) + x² + e^(x-y)
    let (x0, y0) = (0.7, -1.2);

    let tape = Tape::new();
    let guard = Variable::with_tape(&tape);
    let x = Variable::auto(x0, "x");
    let y = Variable::auto(y0, "y");
    let product = &x * &y;
    let f = product
        .sin()
        .add(&x.mul(&y).mul(&x))
        .sub(&x.sub(&y).exp());
    f.sum().backward();
    let (gx, gy) = (x.grad_scalar().unwrap(), y.grad_scalar().unwrap());
    drop(guard);

    let scalar = |a: f64, b: f64| (a * b).sin() + a * a * b - (a - b).exp();
    let numeric_x = central_difference(|a| scalar(a, y0), x0, NUMERIC);
    let numeric_y = central_difference(|b| scalar(x0, b), y0, NUMERIC);

    assert!((gx - numeric_x).abs() < NUMERIC * 5.0, "df/dx {gx} vs {numeric_x}");
    assert!((gy - numeric_y).abs() < NUMERIC * 5.0, "df/dy {gy} vs {numeric_y}");
}

#[test]
fn tensor_gradients_match_finite_differences_elementwise() {
    // f(A) = sum(exp(A) · sin(A) · A) over a 3x4 tensor.
    let mut rng = Rng::new(0xC0FFEE);
    let values: Vec<f64> = (0..12).map(|_| rng.range(-1.0, 1.0)).collect();
    let shape = [3usize, 4];

    let tape = Tape::new();
    let guard = Variable::with_tape(&tape);
    let a = Variable::auto_tensor(Tensor::from_vec(values.clone(), shape).unwrap(), "a");
    let f = a
        .exp()
        .mul(&a.sin())
        .mul(&a)
        .sum();
    f.backward();
    let gradient = a.grad().to_vec();
    drop(guard);

    let scalar = |data: &[f64]| -> f64 {
        data.iter().map(|&v| v.exp() * v.sin() * v).sum()
    };
    let mut numeric = vec![0.0; values.len()];
    for index in 0..values.len() {
        let h = NUMERIC.cbrt() * values[index].abs().max(1.0);
        let mut plus = values.clone();
        plus[index] += h;
        let mut minus = values.clone();
        minus[index] -= h;
        numeric[index] = (scalar(&plus) - scalar(&minus)) / (2.0 * h);
    }
    assert_close(&gradient, &numeric, NUMERIC * 20.0, "tensor gradient");
}

#[test]
fn jacobian_of_vector_function_via_forward_mode() {
    // f(x) = [x1·x2, sin(x1) + x2²] at (0.8, 1.5)
    // J = [[x2, x1], [cos(x1), 2x2]]
    let (x0, y0) = (0.8, 1.5);
    let jacobian = |direction: (f64, f64)| {
        let x = Dual::with_tangent(x0, direction.0);
        let y = Dual::with_tangent(y0, direction.1);
        [(x * y).tangent, (x.sin() + y * y).tangent]
    };
    let column_x = jacobian((1.0, 0.0));
    let column_y = jacobian((0.0, 1.0));
    assert!((column_x[0] - y0).abs() < EXACT);
    assert!((column_x[1] - x0.cos()).abs() < EXACT);
    assert!((column_y[0] - x0).abs() < EXACT);
    assert!((column_y[1] - 2.0 * y0).abs() < EXACT);
}

#[test]
fn softmax_like_pipeline_is_consistent() {
    // A three-layer scalar "network" evaluated both ways.
    let build = |x: &Variable| -> Variable {
        let w1 = Variable::auto(0.5, "w1");
        let w2 = Variable::auto(-1.5, "w2");
        let h1 = x.mul(&w1).tanh();
        let h2 = h1.mul(&w2).exp();
        ops::sigmoid(&h2).ln().neg()
    };
    let x0 = 0.9;
    let tape = Tape::new();
    let guard = Variable::with_tape(&tape);
    let x = Variable::auto(x0, "x");
    let loss = build(&x);
    loss.backward();
    let gradient = x.grad_scalar().unwrap();
    drop(guard);

    let numeric = central_difference(
        |v| {
            let inner = Tape::new();
            let _g = Variable::with_tape(&inner);
            let xx = Variable::auto(v, "x");
            build(&xx).scalar().unwrap()
        },
        x0,
        NUMERIC,
    );
    assert!(
        (gradient - numeric).abs() < NUMERIC * 10.0,
        "pipeline gradient {gradient:e} vs numeric {numeric:e}"
    );
}

#[test]
fn matmul_gradient_matches_finite_differences() {
    let mut rng = Rng::new(7);
    let m = 3;
    let k = 2;
    let n = 4;
    let a_values: Vec<f64> = (0..m * k).map(|_| rng.range(-1.0, 1.0)).collect();
    let b_values: Vec<f64> = (0..k * n).map(|_| rng.range(-1.0, 1.0)).collect();

    let tape = Tape::new();
    let guard = Variable::with_tape(&tape);
    let a = Variable::auto_tensor(Tensor::from_vec(a_values.clone(), [m, k]).unwrap(), "a");
    let b = Variable::auto_tensor(Tensor::from_vec(b_values.clone(), [k, n]).unwrap(), "b");
    // f = sum((A·B)²) — nonlinear in both operands.
    let product = a.matmul(&b).unwrap();
    product.mul(&product).sum().backward();
    let grad_a = a.grad().to_vec();
    let grad_b = b.grad().to_vec();
    drop(guard);

    let scalar = |a: &[f64], b: &[f64]| -> f64 {
        let mut total = 0.0;
        for i in 0..m {
            for j in 0..n {
                let mut entry = 0.0;
                for p in 0..k {
                    entry += a[i * k + p] * b[p * n + j];
                }
                total += entry * entry;
            }
        }
        total
    };

    let mut numeric_a = vec![0.0; a_values.len()];
    for index in 0..a_values.len() {
        let h = 1e-6;
        let mut plus = a_values.clone();
        plus[index] += h;
        let mut minus = a_values.clone();
        minus[index] -= h;
        numeric_a[index] = (scalar(&plus, &b_values) - scalar(&minus, &b_values)) / (2.0 * h);
    }
    let mut numeric_b = vec![0.0; b_values.len()];
    for index in 0..b_values.len() {
        let h = 1e-6;
        let mut plus = b_values.clone();
        plus[index] += h;
        let mut minus = b_values.clone();
        minus[index] -= h;
        numeric_b[index] = (scalar(&a_values, &plus) - scalar(&a_values, &minus)) / (2.0 * h);
    }

    assert!(
        max_abs_diff(&grad_a, &numeric_a) < 1e-5,
        "matmul grad A: max diff {}",
        max_abs_diff(&grad_a, &numeric_a)
    );
    assert!(
        max_abs_diff(&grad_b, &numeric_b) < 1e-5,
        "matmul grad B: max diff {}",
        max_abs_diff(&grad_b, &numeric_b)
    );
}

#[test]
fn broadcasting_gradient_matches_finite_differences() {
    // f(x (3,), y (2,1)) = sum((x + y)²)
    let x_values = vec![0.3, -1.1, 0.7];
    let y_values = vec![1.4, -0.6];

    let tape = Tape::new();
    let guard = Variable::with_tape(&tape);
    let x = Variable::auto_tensor(Tensor::from_vec(x_values.clone(), [3]).unwrap(), "x");
    let y = Variable::auto_tensor(Tensor::from_vec(y_values.clone(), [2, 1]).unwrap(), "y");
    let sum = x.add(&y);
    sum.mul(&sum).sum().backward();
    let grad_x = x.grad().to_vec();
    let grad_y = y.grad().to_vec();
    drop(guard);

    let scalar = |x: &[f64], y: &[f64]| -> f64 {
        let mut total = 0.0;
        for i in 0..2 {
            for j in 0..3 {
                let v = x[j] + y[i];
                total += v * v;
            }
        }
        total
    };
    let mut numeric_x = vec![0.0; 3];
    for index in 0..3 {
        let h = 1e-6;
        let mut plus = x_values.clone();
        plus[index] += h;
        let mut minus = x_values.clone();
        minus[index] -= h;
        numeric_x[index] = (scalar(&plus, &y_values) - scalar(&minus, &y_values)) / (2.0 * h);
    }
    let mut numeric_y = vec![0.0; 2];
    for index in 0..2 {
        let h = 1e-6;
        let mut plus = y_values.clone();
        plus[index] += h;
        let mut minus = y_values.clone();
        minus[index] -= h;
        numeric_y[index] = (scalar(&x_values, &plus) - scalar(&x_values, &minus)) / (2.0 * h);
    }
    assert_close(&grad_x, &numeric_x, 1e-5, "broadcast grad x");
    assert_close(&grad_y, &numeric_y, 1e-5, "broadcast grad y");
}
