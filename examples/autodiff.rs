//! Forward mode and reverse mode on the same function.
//!
//! `f(x, y) = sin(x·y) + x²y`, evaluated at `(0.7, -1.2)`.
//!
//! Run with: `cargo run --release --example autodiff`

use ncly::autodiff::{Dual, Tape, Variable};
use ncly::{NclyError, Tensor};

fn main() -> Result<(), NclyError> {
    let (x0, y0) = (0.7_f64, -1.2_f64);

    // --- Forward mode: one JVP per direction -------------------------------
    let dx = {
        let x = Dual::with_tangent(x0, 1.0);
        let y = Dual::with_tangent(y0, 0.0);
        (x * y).sin().add(x * x * y).tangent
    };
    let dy = {
        let x = Dual::with_tangent(x0, 0.0);
        let y = Dual::with_tangent(y0, 1.0);
        (x * y).sin().add(x * x * y).tangent
    };
    println!("forward   ∂f/∂x = {dx:.12}");
    println!("forward   ∂f/∂y = {dy:.12}");

    // --- Reverse mode: one sweep for both partials -------------------------
    let tape = Tape::new();
    let _guard = Variable::with_tape(&tape);
    let x = Variable::auto(x0, "x");
    let y = Variable::auto(y0, "y");
    let f = x.mul(&y).sin().add(&x.mul(&x).mul(&y));
    f.backward();
    println!("reverse   ∂f/∂x = {:.12}", x.grad_scalar()?);
    println!("reverse   ∂f/∂y = {:.12}", y.grad_scalar()?);

    // The analytic gradient, for comparison.
    // ∂f/∂x = y cos(xy) + 2xy ; ∂f/∂y = x cos(xy) + x²
    let (c, s) = ((x0 * y0).cos(), (x0 * y0).sin());
    println!("analytic  ∂f/∂x = {:.12}", y0 * c + 2.0 * x0 * y0);
    println!("analytic  ∂f/∂y = {:.12}", x0 * c + x0 * x0);
    println!("(sin(xy) = {s:.12})");

    // --- Reverse mode over tensors -----------------------------------------
    let tape = Tape::new();
    let _guard = Variable::with_tape(&tape);
    let w = Variable::auto_tensor(Tensor::from_vec(vec![0.5, -1.5, 2.0], [3])?, "w");
    let loss = w.mul(&w)?.sum(); // ‖w‖²
    loss.backward();
    println!("∇‖w‖²  = {:?}", w.grad().to_vec());
    Ok(())
}
