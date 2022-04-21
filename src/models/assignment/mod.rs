use grb::{c, expr::GurobiSum, Model, Var};
use itertools::iproduct;

use crate::{
    models::utils::AddVars,
    problem::{Problem, VesselIndex},
};

pub struct Variables {
    pub x: Vec<Vec<Var>>,
    pub y: Vec<Vec<Var>>,
}

#[allow(non_snake_case)]
pub struct AssignmentMIP {
    pub model: Model,
    pub vars: Variables,
}

#[allow(non_snake_case)]
impl AssignmentMIP {
    pub fn new(
        problem: &Problem,
        vessel_idx: VesselIndex,
        to_load: &[f64],
    ) -> grb::Result<AssignmentMIP> {
        let mut model = Model::new(&format!("assignment model for vessel {:?}", vessel_idx))?;
        model.set_param(grb::param::OutputFlag, 0)?;

        // compartments
        let c = problem.vessels()[vessel_idx].compartments().len();
        let p = problem.products();

        let vars = Variables {
            x: (p, c).cont(&mut model, "x")?,
            y: (p, c).binary(&mut model, "y")?,
        };

        Self::to_load_constraints(&mut model, problem, vessel_idx, to_load, &vars.x)?;
        Self::capacity_constraints(&mut model, problem, vessel_idx, &vars.x, &vars.y)?;
        Self::no_mix_constraints(&mut model, problem, vessel_idx, &vars.y)?;

        Ok(AssignmentMIP { model, vars })
    }

    /// Bounds the model to not load more of a product than it is supposed to
    fn to_load_constraints(
        model: &mut Model,
        problem: &Problem,
        vessel_idx: VesselIndex,
        to_load: &[f64],
        x: &Vec<Vec<Var>>,
    ) -> grb::Result<()> {
        // compartments
        let C = problem.vessels()[vessel_idx].compartments().len();
        // products
        let P = problem.products();

        for p in 0..P {
            let lhs = (0..C).map(|c| x[p][c]).grb_sum();
            model.add_constr(&format!("to_load_{}", p), c!(lhs <= to_load[p]))?;
        }

        Ok(())
    }

    /// Bounds the model to not load more in a compartment than there is room for
    fn capacity_constraints(
        model: &mut Model,
        problem: &Problem,
        vessel_idx: VesselIndex,
        x: &Vec<Vec<Var>>,
        y: &Vec<Vec<Var>>,
    ) -> grb::Result<()> {
        // compartments
        let C = problem.vessels()[vessel_idx].compartments().len();
        // products
        let P = problem.products();

        for (p, c) in iproduct!(0..P, 0..C) {
            let rhs = problem.vessels()[vessel_idx].compartments()[c].0 * y[p][c];
            model.add_constr(&format!("capacity_{}_{}", p, c), c!(x[p][c] <= rhs))?;
        }

        Ok(())
    }

    /// Enforces no mixing of products
    fn no_mix_constraints(
        model: &mut Model,
        problem: &Problem,
        vessel_idx: VesselIndex,
        y: &Vec<Vec<Var>>,
    ) -> grb::Result<()> {
        // compartments
        let C = problem.vessels()[vessel_idx].compartments().len();
        // products
        let P = problem.products();

        for c in 0..C {
            let lhs = (0..P).map(|p| y[p][c]).grb_sum();
            model.add_constr(&format!("no_mix_{}", c), c!(lhs <= 1.0))?;
        }

        Ok(())
    }

    /// Returns wether the provided products can be distributed into compartments without mixing them
    pub fn is_assignable(problem: &Problem, vessel_idx: VesselIndex, to_load: &[f64]) -> bool {
        let mut lp = Self::new(problem, vessel_idx, to_load).expect(
            format!(
                "failed to build assignemnt model for vessel {} and loading: {:?}",
                vessel_idx, to_load
            )
            .as_str(),
        );
        lp.model
            .optimize()
            .expect("failed to optimize assignment model");

        let status = lp
            .model
            .status()
            .expect("failed to retrieve status of assignment model");

        if matches!(status, grb::Status::Optimal) {
            return true;
        } else {
            return false;
        }
    }
}
