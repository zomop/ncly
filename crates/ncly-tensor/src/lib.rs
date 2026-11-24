//! The NCLY tensor engine.
//!
//! A [`Tensor`] is an `f64` buffer plus a [`Layout`](ncly_core::Layout) describing how logical
//! indices map onto that buffer. The same type therefore represents:
//!
//! | Concept | Layout |
//! | --- | --- |
//! | Contiguous tensor | canonical strides, offset 0 |
//! | Transpose | strides swapped, no copy |
//! | Slice / narrow | reduced shape, shifted offset |
//! | Broadcast operand | `0` strides on expanded axes |
//!
//! Storage is reference counted, so `transpose`, `narrow` and `broadcast_to` are
//! `O(rank)` and never copy data. Operations that must produce new values
//! (`add`, `exp`, `matmul`, ...) allocate exactly one output buffer.
//!
//! # Example
//!
//! ```
//! use ncly_tensor::Tensor;
//!
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])?;
//! let b = Tensor::ones([2, 2]);
//! let c = a.matmul(&b)?;
//! assert_eq!(c.to_vec(), vec![3.0, 3.0, 7.0, 7.0]);
//! # Ok::<(), ncly_core::NclyError>(())
//! ```
//!
//! # Numerical conventions
//!
//! * Element-wise arithmetic follows IEEE-754: `0/0` is `NaN`, `1/0` is `+inf`.
//!   NCLY does not silently substitute zeros, because gradients through such
//!   points are meaningless and hiding them makes debugging harder.
//! * Reductions are accumulated with compensated summation on the scalar path
//!   (see `ncly_simd`), so `sum` over `1e6` elements is accurate to a few ulps
//!   of the result rather than a few ulps of the largest term.
//! * Empty tensors are valid. Reducing an empty tensor yields the identity of
//!   the operation: `0` for `sum`, `-inf` for `max`.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

mod matmul;
mod ops;
mod tensor;

pub use matmul::{matmul, matmul_into};
pub use ops::{BinaryOp, UnaryOp};
pub use tensor::Tensor;
