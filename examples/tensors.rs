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
    let smooth = grid.ln().mul(&grid.sqrt())?;
    println!("ln(x)*sqrt(x) sum = {}", smooth.sum());
    Ok(())
}
