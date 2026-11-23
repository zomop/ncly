//! Structured errors.
//!
//! NCLY never panics on user input: shape mismatches, out-of-bounds indices and
//! unsupported layouts all surface as [`NclyError`] values so that callers can
//! recover (the Python bindings, for example, translate them into Python
//! exceptions).

use core::fmt;

/// Errors that arise while reasoning about a single axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisError {
    /// An axis index was used on a tensor that has fewer dimensions.
    AxisOutOfBounds {
        /// The axis that was requested.
        axis: usize,
        /// The number of dimensions actually available.
        rank: usize,
    },
    /// Two shapes disagreed on the size of an axis.
    IncompatibleLength {
        /// The axis being compared.
        axis: usize,
        /// Length on the left-hand side.
        lhs: usize,
        /// Length on the right-hand side.
        rhs: usize,
    },
    /// The requested broadcast would produce more than the maximum rank.
    RankOverflow {
        /// Maximum rank supported by the engine.
        max_rank: usize,
    },
}

impl fmt::Display for AxisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AxisOutOfBounds { axis, rank } => {
                write!(f, "axis {axis} is out of bounds for a {rank}-dimensional value")
            }
            Self::IncompatibleLength { axis, lhs, rhs } => write!(
                f,
                "cannot broadcast axis {axis}: length {lhs} is incompatible with length {rhs}"
            ),
            Self::RankOverflow { max_rank } => {
                write!(f, "resulting rank exceeds the supported maximum of {max_rank}")
            }
        }
    }
}

/// Every fallible operation in NCLY reports one of these variants.
#[derive(Debug, Clone, PartialEq)]
pub enum NclyError {
    /// An axis-level problem.
    Axis(AxisError),
    /// Two shapes were required to be identical but were not.
    ShapeMismatch {
        /// Operation that failed.
        op: &'static str,
        /// Expected shape.
        expected: String,
        /// Observed shape.
        found: String,
    },
    /// An index was outside the bounds of a shape.
    IndexOutOfBounds {
        /// The offending index, one entry per axis.
        index: Vec<usize>,
        /// The shape that was indexed into.
        shape: String,
    },
    /// The operation is only defined for a restricted set of shapes.
    InvalidShape {
        /// Operation that failed.
        op: &'static str,
        /// Human readable explanation.
        reason: String,
    },
    /// A value was not finite where finiteness is required.
    NonFinite {
        /// Operation that failed.
        op: &'static str,
        /// Position of the offending element in the flat buffer.
        flat_index: usize,
    },
    /// A matrix was singular (or numerically indistinguishable from singular).
    Singular {
        /// Operation that failed.
        op: &'static str,
        /// The pivot at which the algorithm gave up.
        pivot: usize,
    },
    /// An iterative algorithm failed to converge within its budget.
    Convergence {
        /// Operation that failed.
        op: &'static str,
        /// Number of iterations performed.
        iterations: usize,
        /// Final residual, when meaningful.
        residual: Option<f64>,
    },
    /// Data was supplied with an inconsistent structure (sparse formats).
    MalformedData {
        /// Operation that failed.
        op: &'static str,
        /// Human readable explanation.
        reason: String,
    },
    /// The computational graph is not valid for the requested action.
    Graph {
        /// Human readable explanation.
        reason: String,
    },
    /// A requested feature is recognised but not implemented yet.
    Unsupported {
        /// Operation that was requested.
        op: &'static str,
    },
}

impl fmt::Display for NclyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Axis(err) => write!(f, "{err}"),
            Self::ShapeMismatch {
                op,
                expected,
                found,
            } => write!(f, "{op}: expected shape {expected}, found {found}"),
            Self::IndexOutOfBounds { index, shape } => {
                write!(f, "index {index:?} is out of bounds for shape {shape}")
            }
            Self::InvalidShape { op, reason } => write!(f, "{op}: invalid shape ({reason})"),
            Self::NonFinite { op, flat_index } => write!(
                f,
                "{op}: non-finite value encountered at flat index {flat_index}"
            ),
            Self::Singular { op, pivot } => {
                write!(f, "{op}: matrix is singular at pivot {pivot}")
            }
            Self::Convergence {
                op,
                iterations,
                residual,
            } => match residual {
                Some(residual) => write!(
                    f,
                    "{op}: no convergence after {iterations} iterations (residual {residual:e})"
                ),
                None => write!(f, "{op}: no convergence after {iterations} iterations"),
            },
            Self::MalformedData { op, reason } => write!(f, "{op}: malformed data ({reason})"),
            Self::Graph { reason } => write!(f, "invalid computational graph: {reason}"),
            Self::Unsupported { op } => write!(f, "{op} is not supported by this build"),
        }
    }
}

impl std::error::Error for NclyError {}

impl From<AxisError> for NclyError {
    fn from(value: AxisError) -> Self {
        Self::Axis(value)
    }
}

/// Convenience alias used across the workspace.
pub type Result<T, E = NclyError> = core::result::Result<T, E>;

impl NclyError {
    /// Build a [`NclyError::ShapeMismatch`].
    pub fn shape_mismatch(op: &'static str, expected: impl Into<String>, found: impl Into<String>) -> Self {
        Self::ShapeMismatch {
            op,
            expected: expected.into(),
            found: found.into(),
        }
    }

    /// Build a [`NclyError::InvalidShape`].
    pub fn invalid_shape(op: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidShape {
            op,
            reason: reason.into(),
        }
    }

    /// Build a [`NclyError::MalformedData`].
    pub fn malformed(op: &'static str, reason: impl Into<String>) -> Self {
        Self::MalformedData {
            op,
            reason: reason.into(),
        }
    }
}
