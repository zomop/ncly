//! Foundational types shared by every NCLY crate.
//!
//! `ncly-core` deliberately contains no tensor math. It owns the vocabulary that
//! the rest of the workspace builds on:
//!
//! * [`Shape`] and [`Strides`] — axis metadata.
//! * [`Layout`] — shape + strides + storage offset, i.e. a strided view.
//! * [`AxisError`] and [`NclyError`] — structured, non-panicking failures.
//! * [`scalar`] — floating point helpers used by decompositions.
//!
//! Keeping these types in a dependency-free crate means `ncly-autodiff`,
//! `ncly-sparse` and `ncly-graph` can agree on one set of definitions without
//! pulling in the tensor engine.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod layout;
pub mod scalar;
pub mod shape;

pub use error::{AxisError, NclyError, Result};
pub use layout::Layout;
pub use shape::{dyn_canonical_strides, Broadcast, Shape, Strides};
