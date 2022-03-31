use grb::{c, expr::GurobiSum, Model, Var};
use itertools::{iproduct, Itertools};
use pyo3::pyclass;

use crate::{
    models::utils::AddVars,
    problem::{NodeIndex, NodeType, Problem, ProductIndex, TimeIndex, VesselIndex},
    solution::{routing::RoutingSolution, Visit},
};

use super::utils::ConvertVars;

pub struct Variables {
    pub w: Vec<Vec<Vec<Var>>>,
    pub x: Vec<Vec<Vec<Vec<Var>>>>,
    pub s: Vec<Vec<Vec<Var>>>,
    pub l: Vec<Vec<Vec<Var>>>,
}

#[pyclass]
pub struct F64Variables {
    #[pyo3(get)]
    pub w: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub x: Vec<Vec<Vec<Vec<f64>>>>,
    #[pyo3(get)]
    pub s: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub l: Vec<Vec<Vec<f64>>>,
}

pub struct QuantityLp {
    pub model: Model,
    pub vars: Variables,
    pub semicont: bool,
}

impl QuantityLp {
    fn inventory_constraints(
        model: &mut Model,
        problem: &Problem,
        s: &[Vec<Vec<Var>>],
        w: &[Vec<Vec<Var>>],
        x: &[Vec<Vec<Vec<Var>>>],
        t: usize,
        n: usize,
        v: usize,
        p: usize,
    ) -> grb::Result<()> {
        for (n, p) in iproduct!(0..n, 0..p) {
            let s0 = problem.nodes()[n].initial_inventory()[p];
            let name = &format!("s_init_{:?}", (n, p));
            model.add_constr(name, c!(s[0][n][p] == s0))?;
        }

        for (t, n, p) in iproduct!(0..t, 0..n, 0..p) {
            let node = &problem.nodes()[n];
            // Whether to enable/disable upper and lower soft constraint, respectively
            let (u, l): (f64, f64) = match node.r#type() {
                NodeType::Consumption => (0.0, 1.0),
                NodeType::Production => (1.0, 0.0),
            };

            let ub = node.capacity()[p];
            let lb = 0.0;

            let c_ub = c!(s[t][n][p] <= ub + u * w[t][n][p]);
            let c_lb = c!(s[t][n][p] >= lb - l * w[t][n][p]);

            model.add_constr(&format!("s_ub_{:?}", (t, n, p)), c_ub)?;
            model.add_constr(&format!("s_lb_{:?}", (t, n, p)), c_lb)?;
        }

        for (t, n, p) in iproduct!(1..t, 0..n, 0..p) {
            let node = &problem.nodes()[n];
            let i = Self::multiplier(node.r#type());
            let external = (0..v).map(|v| x[t - 1][n][v][p]).grb_sum();
            let internal = node.inventory_changes()[t - 1][p];
            let constr = c!(s[t][n][p] == s[t - 1][n][p] + internal - i * external);
            model.add_constr(&format!("s_bal_{:?}", (t, n, p)), constr)?;
        }

        Ok(())
    }

    fn load_constraints(
        model: &mut Model,
        problem: &Problem,
        l: &[Vec<Vec<Var>>],
        x: &[Vec<Vec<Vec<Var>>>],
        t: usize,
        n: usize,
        v: usize,
        p: usize,
    ) -> grb::Result<()> {
        for (v, p) in iproduct!(0..v, 0..p) {
            let initial = &problem.vessels()[v].initial_inventory()[p];
            let name = format!("l_init_{:?}", (v, p));
            model.add_constr(&name, c!(l[0][v][p] == initial))?;
        }

        for (t, v) in iproduct!(0..t, 0..v) {
            let vessel = &problem.vessels()[v];
            let name = format!("l_ub_{:?}", (t, v));
            let used = (0..p).map(|p| l[t][v][p]).grb_sum();

            model.add_constr(&name, c!(used <= vessel.capacity()))?;
        }

        for (t, v, p) in iproduct!(1..t, 0..v, 0..p) {
            let i = |i: usize| Self::multiplier(problem.nodes()[i].r#type());
            let name = format!("l_bal_{:?}", (t, v, p));
            let lhs = l[t][v][p];
            let rhs = l[t - 1][v][p] + (0..n).map(|n| i(n) * x[t - 1][n][v][p]).grb_sum();
            model.add_constr(&name, c!(lhs == rhs))?;
        }

        Ok(())
    }

    fn rate_constraints(
        model: &mut Model,
        problem: &Problem,
        x: &[Vec<Vec<Vec<Var>>>],
        t: usize,
        n: usize,
        v: usize,
        p: usize,
    ) -> grb::Result<()> {
        for t in 0..t {
            for n in 0..n {
                for v in 0..v {
                    let lhs = (0..p).map(|p| x[t][n][v][p]).grb_sum();
                    let rhs = problem.nodes()[n].max_loading_amount();
                    let name = format!("x_rate_{:?}", (t, n, v));
                    model.add_constr(&name, c!(lhs <= rhs))?;
                }
            }
        }

        Ok(())
    }

    fn multiplier(kind: NodeType) -> f64 {
        match kind {
            NodeType::Consumption => -1.0,
            NodeType::Production => 1.0,
        }
    }

    pub fn new(problem: &Problem) -> grb::Result<Self> {
        let mut model = Model::new(&format!("quantities"))?;
        model.set_param(grb::param::OutputFlag, 1)?;

        let t = problem.timesteps();
        let n = problem.nodes().len();
        let v = problem.vessels().len();
        let p = problem.products();

        // Inventory violation for nodes. Note that this is either excess or shortage,
        // depending on whether it is a production node or a consumption node.
        let w = (t, n, p).cont(&mut model, "w")?;
        let x = (t, n, v, p).cont(&mut model, "x")?;
        let s = (t, n, p).free(&mut model, "s")?;
        let l = (t, v, p).cont(&mut model, "l")?;

        // Add constraints for node inventory, vessel load, and loading/unloading rate
        QuantityLp::inventory_constraints(&mut model, problem, &s, &w, &x, t, n, v, p)?;
        QuantityLp::load_constraints(&mut model, problem, &l, &x, t, n, v, p)?;
        QuantityLp::rate_constraints(&mut model, problem, &x, t, n, v, p)?;

        let obj = w.iter().flatten().flatten().grb_sum();
        model.set_objective(obj, grb::ModelSense::Minimize)?;

        Ok(QuantityLp {
            model,
            vars: Variables { w, x, s, l },
            semicont: false,
        })
    }

    pub fn active<'s>(
        solution: &'s RoutingSolution,
    ) -> impl Iterator<Item = (TimeIndex, NodeIndex, VesselIndex, ProductIndex)> + 's {
        let problem = solution.problem();
        let p = problem.products();

        solution
            .iter_with_terminals()
            .enumerate()
            .flat_map(move |(v, plan)| {
                let vessel = &problem.vessels()[v];

                plan.tuple_windows().flat_map(move |(v1, v2)| {
                    // We must determine when we need to leave `v1.node` in order to make it to `v2.node` in time.
                    // Some additional arithmetic parkour is done to avoid underflow cases (damn usizes).
                    let travel_time = problem.travel_time(v1.node, v2.node, vessel);
                    let departure_time = v2.time - v2.time.min(travel_time);
                    // In addition, we can further restrict the active time periods by looking at the longest possible time
                    // the vessel can spend at the node doing constant loading/unloading.
                    let rate = problem.nodes()[v1.node].min_unloading_amount();
                    let max_loading_time = (vessel.capacity() / rate).ceil() as usize;

                    (v1.time..departure_time.min(v1.time + max_loading_time))
                        .flat_map(move |t| (0..p).map(move |p| (t, v1.node, v, p)))
                })
            })
    }

    // set variable type for x-variable
    pub fn set_semicont(&mut self, semicont: bool) {
        self.semicont = semicont;
    }

    /// Set up the model to be ready to solve quantities for `solution`.
    pub fn configure(&mut self, solution: &RoutingSolution) -> grb::Result<()> {
        // By default: disable `all` variables
        let model = &self.model;
        let problem = solution.problem();

        // Disable all x(t, n, v, p) variables
        model.set_obj_attr_batch(
            grb::attr::UB,
            self.vars.x.iter().flat_map(|xs| {
                xs.iter()
                    .flat_map(|xs| xs.iter().flat_map(|xs| xs.iter().map(|x| (*x, 0.0))))
            }),
        )?;

        let lower = |n: usize| match self.semicont {
            true => problem.nodes()[n].min_unloading_amount(),
            false => 0.0,
        };

        let vtype = match self.semicont {
            true => grb::VarType::SemiCont,
            false => grb::VarType::Continuous,
        };

        // set variable type
        model.set_obj_attr_batch(
            grb::attr::VType,
            Self::active(solution).map(|(t, n, v, p)| (self.vars.x[t][n][v][p], vtype)),
        )?;

        // set lower bound
        model.set_obj_attr_batch(
            grb::attr::LB,
            Self::active(solution).map(|(t, n, v, p)| (self.vars.x[t][n][v][p], lower(n))),
        )?;

        // Re-enable the relevant x(t, n, v, p) variables
        model.set_obj_attr_batch(
            grb::attr::UB,
            Self::active(solution).map(|(t, n, v, p)| (self.vars.x[t][n][v][p], f64::INFINITY)),
        )?;

        model.set_obj_attr_batch(
            grb::attr::VType,
            self.vars.x.iter().flat_map(|xs| {
                xs.iter().flat_map(|xs| {
                    xs.iter()
                        .flat_map(|xs| xs.iter().map(|x| (*x, grb::VarType::SemiCont)))
                })
            }),
        )?;

        Ok(())
    }

    /// Solves the model for the current configuration. Returns the variables
    /// Should also include dual variables for the upper bounds on x, but this is not implemented yet
    pub fn solve(&mut self) -> grb::Result<&Variables> {
        self.model.optimize()?;

        Ok(&self.vars)
    }

    /// Solves the model for the current configuration. Returns the variables converted to `F64Variables` exposed to python.
    /// Should also include dual variables for the upper bounds on `x`, but this is not implemented yet
    pub fn solve_python(&mut self) -> grb::Result<F64Variables> {
        self.model.optimize()?;

        let x = self.vars.x.convert(&self.model)?;
        let s = self.vars.s.convert(&self.model)?;
        let l = self.vars.l.convert(&self.model)?;
        let w = self.vars.w.convert(&self.model)?;

        let v = F64Variables { w, x, s, l };
        Ok(v)
    }
}
