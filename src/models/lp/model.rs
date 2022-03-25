use crate::models::utils::{vars, AddVars};

use super::sets_and_parameters::{Parameters, Sets};
use grb::prelude::*;
use log::trace;

pub struct Variables {}

pub struct LpSolver {}

#[allow(non_snake_case)]
impl LpSolver {
    /// Builds the lp determining the quantities
    pub fn build(sets: &Sets, parameters: &Parameters) -> grb::Result<(Model, Variables)> {
        trace!("Building lp for determining quantities");

        let mut model = Model::new(&format!("quant_lp"))?;

        // assign som variables to save some space later
        let P = &sets.P;
        let V = &sets.V;
        let N = &sets.N;
        let J = &sets.J;
        let J_n = &sets.J_n;
        let J_v = &sets.J_v;

        //*************CREATE VARIABLES*************//

        // loaded/unloaded at visit j of product p
        let x = (J.len(), P.len()).cont(&mut model, &"x")?;

        // inventory at the node assiciated with the visit at the beginning of the visit
        let s = (J.len(), P.len()).free(&mut model, &"s")?;

        let indices = todo!();
        let w_minus = vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_minus",
        );

        // shortage at the node associated with the visit at the beginning of the visit

        todo!()
    }
}
