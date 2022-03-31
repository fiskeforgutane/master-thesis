use std::sync::Arc;

use crate::problem::Problem;

pub trait Initialization {
    type Out;
    fn new(&self, problem: Arc<Problem>) -> Self::Out;
}

impl<F, O> Initialization for F
where
    F: Fn(Arc<Problem>) -> O,
{
    type Out = O;
    fn new(&self, problem: Arc<Problem>) -> Self::Out {
        self(problem)
    }
}
