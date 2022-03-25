use crate::problem::Problem;

pub trait Initialization {
    type Out;
    fn new(&self, problem: &Problem) -> Self::Out;
}

impl<F, O> Initialization for F
where
    F: Fn(&Problem) -> O,
{
    type Out = O;
    fn new(&self, problem: &Problem) -> Self::Out {
        self(problem)
    }
}
