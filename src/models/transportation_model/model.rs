use super::sets_and_params::{Parameters, Sets};
use crate::models::utils::{AddVars, ConvertVars};
use crate::problem::ProductIndex;
use grb::prelude::*;
use itertools::iproduct;
use log::info;

pub struct TransportationSolver {}

#[allow(non_snake_case)]
impl TransportationSolver {
    /// builds the transportation model
    pub fn build(
        sets: &Sets,
        parameters: &Parameters,
        product: usize,
    ) -> grb::Result<(Model, Variables)> {
        info!("Building transportation model for product {}", product);

        let mut model = Model::new(&format!("transport_model_{}", product))?;

        //*************CREATE VARIABLES*************//
        let N = sets.N.len();
        let H = sets.H.len();

        // quantity transported from node i to node j with cargo h
        let x: Vec<Vec<Vec<Var>>> = (N, N, H).cont(&mut model, &"x")?;

        // 1 if the h'th cargo from node i to node j exists, 0 otherwise
        let y: Vec<Vec<Vec<Var>>> = (N, N, H).binary(&mut model, &"y")?;

        // itegrate all the variables into the model
        model.update()?;

        // ******************** ADD CONSTRAINTS ********************
        // ensure that enough is picked up at the production nodes
        for i in &sets.N {
            if parameters.J[*i] < 0 {
                continue;
            }
            let lhs = iproduct!(&sets.N, &sets.H)
                .map(|(j, h)| &x[*i][*j][*h])
                .grb_sum();
            model.add_constr(&format!("1_{i}"), c!(lhs >= parameters.Q[product][*i]))?;
        }

        // ensure that enough is delivered at the consumption nodes
        for j in &sets.N {
            if parameters.J[*j] > 0 {
                continue;
            }
            let lhs = iproduct!(&sets.N, &sets.H)
                .map(|(i, h)| &x[*i][*j][*h])
                .grb_sum();
            model.add_constr(&format!("2_{j}"), c!(lhs >= parameters.Q[product][*j]))?;
        }

        // lower limit on transportation amount
        for (i, j, h) in iproduct!(&sets.N, &sets.N, &sets.H) {
            // lower bound on how much can be transported from node i to node j
            let lhs = parameters.lower_Q[product][*i][*j] * y[*i][*j][*h];
            model.add_constr(
                &format!("lower_transportation_limit_{i}_{j}_{h}"),
                c!(lhs <= x[*i][*j][*h]),
            )?;
        }

        // lower limit on transportation amount
        for (i, j, h) in iproduct!(&sets.N, &sets.N, &sets.H) {
            // lower bound on how much can be transported from node i to node j
            let rhs = parameters.upper_Q[product][*i][*j] * y[*i][*j][*h];
            model.add_constr(
                &format!("upper_transportation_limit_{i}_{j}_{h}"),
                c!(x[*i][*j][*h] <= rhs),
            )?;
        }

        // symmetry braking constraint stating that if cargo h from node i to node j with h > 1, then cargo h-1 must also exist
        for (i, j, h) in iproduct!(&sets.N, &sets.N, &sets.H[1..]) {
            model.add_constr(
                &format!("symmetry_{}_{}_{}", i, j, h),
                c!(y[*i][*j][h - 1] >= y[*i][*j][*h]),
            )?;
        }

        // set objective which is the cost of transporting the cargoes that exist
        let transport_costs = iproduct!(&sets.N, &sets.N, &sets.H)
            .map(|(i, j, h)| parameters.C[*i][*j] * parameters.epsilon[*i][*j] * y[*i][*j][*h])
            .grb_sum();

        let fixed_port_costs = iproduct!(&sets.N, &sets.N, &sets.H)
            .map(|(i, j, h)| (parameters.C_fixed[*i] + parameters.C_fixed[*j]) * y[*i][*j][*h])
            .grb_sum();

        model.set_objective(transport_costs + fixed_port_costs, Minimize)?;

        model.update()?;

        info!(
            "Successfully built transportation model for product {}",
            product
        );
        Ok((model, Variables::new(x, y)))
    }

    pub fn solve(
        sets: &Sets,
        parameters: &Parameters,
        product: ProductIndex,
    ) -> Result<TransportationResult, grb::Error> {
        // build model
        let (m, vars) = TransportationSolver::build(sets, parameters, product)?;
        let mut model = m;

        // optimize model
        model.optimize()?;

        TransportationResult::new(&vars, &model)
    }
}

#[derive(Debug, Clone)]
pub struct TransportationResult {
    /// nonzero quantities transported from node i to j with cargo h
    pub x: Vec<Vec<Vec<f64>>>,
    /// nonzero y variables, not necessary fo results but good for testing
    pub y: Vec<Vec<Vec<f64>>>,
}

impl TransportationResult {
    pub fn new(variables: &Variables, model: &Model) -> Result<TransportationResult, grb::Error> {
        let x = variables.x.convert(model)?;
        let y = variables.y.convert(model)?;

        Ok(TransportationResult { x, y })
    }
}

pub struct Variables {
    x: Vec<Vec<Vec<Var>>>,
    y: Vec<Vec<Vec<Var>>>,
}
impl Variables {
    pub fn new(x: Vec<Vec<Vec<Var>>>, y: Vec<Vec<Vec<Var>>>) -> Variables {
        Variables { x, y }
    }
}
