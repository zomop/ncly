//! NCLY — numerical computing for Rust.
//!
//! `ncly` is the umbrella crate: it re-exports the workspace crates behind one
//! namespace so applications can depend on a single package.
//!
//! ```
//! use ncly::{Tensor, UnaryOp};
//!
//! let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3])?;
//! let y = x.mul(&x)?.sum();
//! assert_eq!(y, 14.0);
//! # Ok::<(), ncly_core::NclyError>(())
//! ```
//!
//! # Crate map
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`core`] | shapes, strides, layouts, errors |
//! | [`tensor`] | `n`-d strided arrays and element-wise math |
//! | [`simd`] | runtime-dispatched kernels |
//! | [`autodiff`] | forward- and reverse-mode differentiation |
//! | [`linalg`] | LU/QR/solvers/eigenvalues |
//! | [`sparse`] | CSR/CSC storage and kernels |
//! | [`graph`] | dataflow DAG with shape inference and scheduling |
//! | [`compiler`] | constant folding, CSE, dead-node elimination |

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

pub use ncly_core as core;
pub use ncly_simd as simd;
pub use ncly_tensor as tensor;

pub use ncly_autodiff as autodiff;
pub use ncly_linalg as linalg;
pub use ncly_sparse as sparse;
pub use ncly_graph as graph;
pub use ncly_compiler as compiler;

pub use ncly_autodiff::{Dual, Tape, Variable};
pub use ncly_core::{NclyError, Result, Shape, Strides};
pub use ncly_linalg::{lu, qr};
pub use ncly_sparse::CsrMatrix;
pub use ncly_graph::{Graph, NodeId};
pub use ncly_tensor::{BinaryOp, Tensor, UnaryOp};

/// Version of the NCLY workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The SIMD kernel family selected on this machine.
pub fn active_simd_kernel() -> &'static str {
    ncly_simd::active_kernel().name()
}
