use std::sync::Arc;

use crate::{models::quantity::ModelObjectiveWeights, problem::Problem};

pub trait Initialization {
    type Out;
    fn new(
        &self,
        problem: Arc<Problem>,
        objective_weights: Arc<ModelObjectiveWeights>,
    ) -> Self::Out;
}

impl<F, O> Initialization for F
where
    F: Fn(Arc<Problem>) -> O,
{
    type Out = O;
    fn new(&self, problem: Arc<Problem>, _: Arc<ModelObjectiveWeights>) -> Self::Out {
        self(problem)
    }
}
