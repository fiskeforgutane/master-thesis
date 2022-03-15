pub mod problem;
pub mod quants;
pub mod sisrs;
pub mod solution;
use problem::Problem;
use pyo3::prelude::*;
use solution::Visit;

#[pyclass]
pub struct Solution {
    #[pyo3(get, set)]
    pub routes: Vec<Vec<Visit>>,
}

#[pymethods]
impl Solution {
    #[new]
    pub fn new(routes: Vec<Vec<Visit>>) -> Self {
        Self { routes }
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn master(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Problem>()?;
    m.add_class::<Solution>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
