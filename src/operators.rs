use crate::quants::SisrsOutput;

pub trait OrderOperator {
    /// Returns the orders after applying the operator on them
    ///
    /// # Arguments
    ///
    /// * `orders` - A Vector that holds the current orders
    /// * `output` - A SISRs output containing routes and loads
    /// * `problem` - The problem
    fn apply(&self, orders: &Vec<Order>, output: &SisrsOutput, problem: &Problem) -> Vec<Order>;
}

pub struct MinMaxOperator {}
pub struct EqualQuantsOperator {}
pub struct VesselCapOperator {}
pub struct VesselCap2Operator {}
pub struct VesselCap3Operator {}
pub struct VesselCap4Operator {}

impl OrderOperator for MinMaxOperator {
    fn apply(&self, orders: Vec<Order>) -> Vec<Order> {
        todo!()
    }
}
