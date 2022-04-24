use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::{models::quantity::QuantityLp, problem::Problem};

pub trait Initialization {
    type Out;
    fn new(&self, problem: Arc<Problem>, quantites: Rc<RefCell<QuantityLp>>) -> Self::Out;
}

impl<F, O> Initialization for F
where
    F: Fn(Arc<Problem>, Rc<RefCell<QuantityLp>>) -> O,
{
    type Out = O;
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
        self(problem, quantities)
    }
}
