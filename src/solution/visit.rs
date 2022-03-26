use pyo3::pyclass;

use crate::problem::{NodeIndex, Problem, TimeIndex};

/// A `Visit` is a visit to a `node` at a specific `time`.
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Visit {
    #[pyo3(get)]
    /// The node we're visiting.
    pub node: NodeIndex,
    #[pyo3(get)]
    /// The time at which delivery starts.
    pub time: TimeIndex,
}

#[derive(Debug)]
pub enum Error {
    InvalidNode(usize),
    InvalidTime(usize),
}

impl Visit {
    /// Construct a new visit, ensuring that the node and time index are valid.
    pub fn new(problem: &Problem, node: NodeIndex, time: TimeIndex) -> Result<Visit, Error> {
        use Error::*;
        if node >= problem.nodes().len() {
            return Err(InvalidNode(node));
        }

        if time >= problem.timesteps() {
            return Err(InvalidTime(time));
        }

        Ok(Self { node, time })
    }

    /// Construct a new visit without checkout its validity.
    /// Prefer to use `Visit::new` unless you have ensured that `node` and `time` are within bounds.
    pub fn new_unchecked(node: NodeIndex, time: TimeIndex) -> Visit {
        Self { node, time }
    }
}
