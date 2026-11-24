//! Broadcasting, views and reductions.
//!
//! Run with: `cargo run --release -p ncly --example tensors`

use ncly::{NclyError, Tensor};

fn main() -> Result<(), NclyError> {
    // A (3, 4) grid and a (4,) row broadcast against it.
    let grid = Tensor::from_vec((0..12).map(f64::from).collect(), [3, 4])?;
    let row = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], [4])?;
    let shifted = grid.add(&row)?;
    assert_eq!(shifted.shape().dims(), &[3, 4]);
    assert_eq!(shifted.get(&[1, 2])?, 6.0 + 30.0);

    // A column of shape (3, 1) broadcasts over the same grid.
    let column = Tensor::from_vec(vec![100.0, 200.0, 300.0], [3, 1])?;
    let outer = grid.add(&column)?;
    assert_eq!(outer.get(&[2, 0])?, 8.0 + 300.0);

    // Views share storage: transposing does not copy.
    let transposed = grid.transpose();
    assert_eq!(transposed.shape().dims(), &[4, 3]);
    assert_eq!(transposed.get(&[2, 1])?, grid.get(&[1, 2])?);

    // Reductions use compensated summation.
    println!("sum    = {}", grid.sum());
    println!("mean   = {}", grid.mean());
    println!("max    = {}", grid.max());
    println!("column sums = {:?}", grid.sum_axis(0)?.to_vec());

    // Element-wise math composes on views, not only on contiguous buffers.
    // The shift matters: `grid` starts at 0, and `ln(0)` is `-inf`, so taking the
    // logarithm of the raw grid would print `NaN` and demonstrate nothing.
    let positive = grid.adds(1.0);
    let smooth = positive.ln().mul(&positive.sqrt())?;
    assert!(smooth.all_finite(), "every element must be finite");
    println!("ln(1+x)*sqrt(1+x) sum = {}", smooth.sum());
    Ok(())
}
