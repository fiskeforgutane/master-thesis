pub mod sparse;

use grb::{add_ctsvar, c, expr::GurobiSum, Model, Var};
use itertools::{iproduct, Itertools};
use pyo3::pyclass;

use crate::{
    models::utils::AddVars,
    problem::{NodeIndex, NodeType, Problem, ProductIndex, TimeIndex, VesselIndex},
    solution::routing::RoutingSolution,
};

use super::utils::ConvertVars;

pub struct Variables {
    pub w: Vec<Vec<Vec<Var>>>,
    pub x: Vec<Vec<Vec<Vec<Var>>>>,
    pub s: Vec<Vec<Vec<Var>>>,
    pub l: Vec<Vec<Vec<Var>>>,
    pub b: Vec<Vec<Vec<Var>>>,
    pub a: Vec<Vec<Vec<Var>>>,
    pub cap_violation: Vec<Vec<Var>>,
    pub violation: Var,
    pub spot: Var,
    pub revenue: Var,
    pub timing: Var,
    pub travel_empty: Var,
    pub travel_at_cap: Var,
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
    #[pyo3(get)]
    pub revenue: f64,
    #[pyo3(get)]
    pub spot: f64,
    #[pyo3(get)]
    pub violation: f64,
    #[pyo3(get)]
    pub timing: f64,
    #[pyo3(get)]
    pub a: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub cap_violation: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub travel_empty: f64,
    #[pyo3(get)]
    pub travel_at_cap: f64,
}

pub struct QuantityLp {
    pub model: Model,
    pub vars: Variables,
    pub semicont: bool,
    pub berth: bool,
}

impl QuantityLp {
    fn inventory_constraints(
        model: &mut Model,
        problem: &Problem,
        s: &[Vec<Vec<Var>>],
        w: &[Vec<Vec<Var>>],
        x: &[Vec<Vec<Vec<Var>>>],
        a: &[Vec<Vec<Var>>],
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

        for (t, n, p) in iproduct!(0..=t, 0..n, 0..p) {
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

        for (t, n, p) in iproduct!(1..=t, 0..n, 0..p) {
            let node = &problem.nodes()[n];
            let i = Self::multiplier(node.r#type());
            let external = (0..v).map(|v| x[t - 1][n][v][p]).grb_sum();
            let internal = node.inventory_changes()[t - 1][p];
            let spot = a
                .get(t)
                .map(|at| 1.0 * at[n][p])
                .unwrap_or(0.0 * a[0][0][0]);
            let constr = c!(s[t][n][p] == s[t - 1][n][p] + internal - i * (external + spot));
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

        for (t, v) in iproduct!(0..=t, 0..v) {
            let vessel = &problem.vessels()[v];
            let name = format!("l_ub_{:?}", (t, v));
            let used = (0..p).map(|p| l[t][v][p]).grb_sum();

            model.add_constr(&name, c!(used <= vessel.capacity()))?;
        }

        for (t, v, p) in iproduct!(1..=t, 0..v, 0..p) {
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

    fn berth_capacity(
        model: &mut Model,
        problem: &Problem,
        x: &[Vec<Vec<Vec<Var>>>],
        t: usize,
        n: usize,
        v: usize,
        p: usize,
        b: &[Vec<Vec<Var>>],
    ) -> grb::Result<()> {
        for (node, time) in iproduct!(0..n, 0..t) {
            let berth_cap = problem.nodes()[node].port_capacity()[time];
            model.add_constr(
                &format!("berth_{}_{}", node, time),
                c!(b[time][node].iter().grb_sum() <= berth_cap),
            )?;
        }

        for (node, time, vessel) in iproduct!(0..n, 0..t, 0..v) {
            let kind = problem.nodes()[node].r#type();
            let big_m = match kind {
                NodeType::Consumption => problem.nodes()[node].max_loading_amount(),
                NodeType::Production => (0..p).map(|p| problem.nodes()[node].capacity()[p]).sum(),
            };
            model.add_constr(
                &format!("force_berth_{}_{}_{}", node, time, vessel),
                c!(x[time][node][vessel].iter().grb_sum() <= big_m * b[time][node][vessel]),
            )?;
        }

        Ok(())
    }

    fn alpha_limits(
        model: &mut Model,
        problem: &Problem,
        a: &[Vec<Vec<Var>>],
        t: usize,
        n: usize,
    ) -> grb::Result<()> {
        // Restrict the amount of alpha we can use in any particular time period.
        for n in 0..n {
            let node = &problem.nodes()[n];

            // Limit of alpha usage per time step
            for t in 0..n {
                model.add_constr(
                    &format!("alpha_period_ub_{}_{}", t, n),
                    c!(a[t][n].iter().grb_sum() <= node.spot_market_limit_per_time()),
                )?;
            }

            // Limit for alpha usage across all time steps
            let sum = (0..t).flat_map(|t| a[t][n].iter()).grb_sum();
            model.add_constr(
                &format!("alpha_ub_{}", n),
                c!(sum <= node.spot_market_limit()),
            )?;
        }

        Ok(())
    }

    fn reset_load_bounds(
        model: &mut Model,
        problem: &Problem,
        l: &[Vec<Vec<Var>>],
        cap_violation: &[Vec<Var>],
    ) -> grb::Result<()> {
        let (v, t, p) = (
            problem.vessels().len(),
            problem.timesteps(),
            problem.products(),
        );

        model.set_obj_attr_batch(
            grb::attr::UB,
            iproduct!(0..v, 0..t).map(|(vessel, time)| {
                (
                    cap_violation[time][vessel],
                    problem.vessels()[vessel].capacity(),
                )
            }),
        )?;

        model.set_obj_attr_batch(
            grb::attr::UB,
            iproduct!(0..v, 0..t).flat_map(|(vessel, time)| {
                (0..p).map(move |product| {
                    (
                        l[time][vessel][product],
                        problem.vessels()[vessel].capacity(),
                    )
                })
            }),
        )?;

        Ok(())
    }

    fn cap_restrictions(
        model: &mut Model,
        problem: &Problem,
        l: &[Vec<Vec<Var>>],
        cap_violation: &[Vec<Var>],
        v: usize,
        t: usize,
        p: usize,
    ) -> grb::Result<()> {
        for (time, vessel) in iproduct!(0..t, 0..v) {
            model.add_constr(
                &format!("setCap_{}_{}", time, vessel),
                c!(cap_violation[time][vessel]
                    == problem.vessels()[vessel].capacity()
                        - (0..p).map(|product| l[time][vessel][product]).grb_sum()),
            )?;
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
        // Disable output logging.
        model.set_param(grb::param::OutputFlag, 0)?;
        // Use primal simplex, instead of the default concurrent solver. Reason: we will use multiple concurrent GAs
        model.set_param(grb::param::Method, 0)?;
        // Restrict to one thread. Also due to using concurrent GAs.
        model.set_param(grb::param::Threads, 1)?;

        let t = problem.timesteps();
        let n = problem.nodes().len();
        let v = problem.vessels().len();
        let p = problem.products();

        // Inventory violation for nodes. Note that this is either excess or shortage,
        // depending on whether it is a production node or a consumption node.
        let w = (t + 1, n, p).cont(&mut model, "w")?;
        let x = (t, n, v, p).cont(&mut model, "x")?;
        let s = (t + 1, n, p).free(&mut model, "s")?;
        let l = (t + 1, v, p).cont(&mut model, "l")?;
        let b = (t, n, v).cont(&mut model, "b")?;
        let a = (t, n, p).cont(&mut model, "a")?;
        let cap_violation = (t, v).cont(&mut model, "cap_violation")?;

        let revenue = add_ctsvar!(model, name: "revenue", bounds: 0.0..)?;
        let timing = add_ctsvar!(model, name: "timinig", bounds: 0.0..)?;
        let spot = add_ctsvar!(model, name: "spot", bounds: 0.0..)?;
        let violation = add_ctsvar!(model, name: "violation", bounds: 0.0..)?;
        let travel_empty = add_ctsvar!(model, name: "travel_empty", bounds: 0.0..)?;
        let travel_at_cap = add_ctsvar!(model, name: "travel_at_cap", bounds: 0.0..)?;

        // Add constraints for node inventory, vessel load, and loading/unloading rate
        QuantityLp::inventory_constraints(&mut model, problem, &s, &w, &x, &a, t, n, v, p)?;
        QuantityLp::load_constraints(&mut model, problem, &l, &x, t, n, v, p)?;
        QuantityLp::rate_constraints(&mut model, problem, &x, t, n, v, p)?;
        QuantityLp::berth_capacity(&mut model, problem, &x, t, n, v, p, &b)?;
        QuantityLp::alpha_limits(&mut model, problem, &a, t, n)?;
        QuantityLp::cap_restrictions(&mut model, problem, &l, &cap_violation, v, t, p)?;

        let violation_expr = w.iter().flatten().flatten().grb_sum();
        // We discount later uses of the spot market; effectively making it desirable to perform spot operations as late as possible
        let spot_expr = iproduct!(0..t, 0..n, 0..p)
            .map(|(t, n, p)| {
                let node = &problem.nodes()[n];
                let unit_price = node.spot_market_unit_price();
                let discount = node.spot_market_discount_factor().powi(t as i32);
                a[t][n][p] * unit_price * discount
            })
            .grb_sum();

        // We use an increasing weight to prefer early deliveries.
        // The purpose of this is to "avoid" many small deliveries versus one larger one, even though it
        // doesn't really matter for the solution itself.
        let revenue_expr = iproduct!(0..t, 0..n, 0..v, 0..p)
            .map(|(t, n, v, p)| problem.nodes()[n].revenue() * x[t][n][v][p])
            .grb_sum();

        let timing_expr = iproduct!(0..t, 0..n, 0..v, 0..p)
            .map(|(t, n, v, p)| x[t][n][v][p] * t as i32)
            .grb_sum();

        // set the travel empty variable equal to all load variables - the weights will be modified in configure
        let travel_empty_expr = iproduct!(0..t, 0..v, 0..p)
            .map(|(t, v, p)| l[t][v][p])
            .grb_sum();
        // do the same with travel at capacity
        let travel_at_capacity_expr = iproduct!(0..t, 0..v)
            .map(|(t, v)| cap_violation[t][v])
            .grb_sum();

        model.add_constr("spot", c!(spot == spot_expr))?;
        model.add_constr("revenue", c!(revenue == revenue_expr))?;
        model.add_constr("violation", c!(violation == violation_expr))?;
        model.add_constr("timing", c!(timing == timing_expr))?;
        model.add_constr("travel_empty", c!(travel_empty_expr == travel_empty))?;
        model.add_constr(
            "travel_at_cap",
            c!(travel_at_capacity_expr == travel_at_cap),
        )?;

        let obj = violation + 0.5_f64 * spot - 1e-6_f64 * revenue
            + 1e-12_f64 * timing
            + travel_empty
            + travel_at_cap;
        model.set_objective(obj, grb::ModelSense::Minimize)?;

        Ok(QuantityLp {
            model,
            vars: Variables {
                w,
                x,
                s,
                l,
                b,
                a,
                cap_violation,
                violation,
                spot,
                revenue,
                timing,
                travel_empty,
                travel_at_cap,
            },
            semicont: false,
            berth: false,
        })
    }

    pub fn add_compartment_constraints(&mut self, problem: &Problem) -> grb::Result<()> {
        // Whether compartment c if assigned to hold product p, for each vessel
        let model = &mut self.model;
        let p = problem.products();
        let t = problem.timesteps();
        for (v, vessel) in problem.vessels().iter().enumerate() {
            let c = vessel.compartments().len();
            let assigned = (t, p, c).binary(model, &format!("compartment_{v}"))?;
            // Product must be carried in compartments
            for t in 0..t {
                // Each product must have enough assigned silos to cover their quantity.
                for p in 0..p {
                    let lhs = self.vars.l[t][v][p];
                    let rhs = assigned[t][p]
                        .iter()
                        .enumerate()
                        .map(|(c, &assigned)| vessel.compartments()[c].0 * assigned)
                        .grb_sum();

                    model.add_constr(&format!("assign_ub_{v}_{t}_{p}"), c!(lhs <= rhs))?;
                }

                // A compartment can only be assigned to at most one feed type
                for c in 0..c {
                    let lhs = (0..p).map(|p| assigned[t][p][c]).grb_sum();
                    model.add_constr(&format!("one_p_per_c_{v}_{t}_{c}"), c!(lhs <= 1.0_f64))?;
                }
            }
        }
        model.update()?;

        Ok(())
    }

    /// Returns the indicies of the x-variables in the LP that are allowed to take positive values
    ///
    /// ## Arguments
    /// * `solution` - the given routing solution that is evaluated
    /// * `tight` - true if all visits (except the origin) should be constrained by the maximum time it takes to fully load or unload the vessel, if false, only reaching the next visit is considered
    pub fn active<'s>(
        solution: &'s RoutingSolution,
        tight: bool,
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
                    let max_loading_time = if problem.origin_visit(v) == v1 || !tight {
                        // do not tighten
                        departure_time
                    } else {
                        // allow to tighten
                        (vessel.capacity() / rate).ceil() as usize
                    };

                    (v1.time..=departure_time.min(v1.time + max_loading_time))
                        .flat_map(move |t| (0..p).map(move |p| (t, v1.node, v, p)))
                })
            })
    }

    /// Set up the model to be ready to solve quantities for `solution`.
    ///
    /// ## Arguments
    /// * `solution` - the given routing solution that is evaluated
    /// * `semicont` - whether to enable semicont deliveries or not
    /// * `berth` - whether to enabel berth restrictions or not
    /// * `load_restrictions` - wether to seek to arrive empty at production nodes and leave them full
    /// * `tight` - true if all visits (except the origin) should be constrained by the maximum time it takes to fully load or unload the vessel, if false, only reaching the next visit is considered
    pub fn configure(
        &mut self,
        solution: &RoutingSolution,
        semicont: bool,
        berth: bool,
        tight: bool,
    ) -> grb::Result<()> {
        self.semicont = semicont;

        self.berth = berth;

        // reset load bounds
        QuantityLp::reset_load_bounds(
            &mut self.model,
            solution.problem(),
            &self.vars.l,
            &self.vars.cap_violation,
        )?;

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
            Self::active(solution, tight).map(|(t, n, v, p)| (self.vars.x[t][n][v][p], vtype)),
        )?;

        let vtype = match self.berth {
            true => grb::VarType::Binary,
            false => grb::VarType::Continuous,
        };

        // set variable type
        model.set_obj_attr_batch(
            grb::attr::VType,
            Self::active(solution, tight).map(|(t, n, v, _)| (self.vars.b[t][n][v], vtype)),
        )?;

        // set lower bound
        model.set_obj_attr_batch(
            grb::attr::LB,
            Self::active(solution, tight).map(|(t, n, v, p)| (self.vars.x[t][n][v][p], lower(n))),
        )?;

        // Re-enable the relevant x(t, n, v, p) variables
        model.set_obj_attr_batch(
            grb::attr::UB,
            Self::active(solution, tight)
                .map(|(t, n, v, p)| (self.vars.x[t][n][v][p], f64::INFINITY)),
        )?;

        self.configure_travel_constraints(solution)?;

        Ok(())
    }

    /// Configure the travel at cap and travel at capacity constraints
    fn configure_travel_constraints(&mut self, solution: &RoutingSolution) -> grb::Result<()> {
        let model = &mut self.model;
        let problem = solution.problem();

        let kind = |n: usize| solution.problem().nodes()[n].r#type();

        let iterator = solution.iter().enumerate().flat_map(|(v, plan)| {
            plan.iter()
                .enumerate()
                .skip(1)
                .map(move |(visit_idx, visit)| (v, plan, visit_idx, visit))
        });

        // identify arrivals at production nodes
        let prod_arrivals = iterator.clone().filter_map(|(v, plan, visit_idx, visit)| {
            // check that the previous visit is not at a production node
            let prev = plan[visit_idx - 1].node;
            if let NodeType::Production = kind(prev) {
                None
            } else {
                match kind(visit.node) {
                    NodeType::Consumption => None,
                    NodeType::Production => Some((visit.time, v)),
                }
            }
        });

        // identify arrivals at the first consumption node after visiting a production node
        let cons_arrivals = iterator.filter_map(|(v, plan, visit_idx, visit)| {
            let prev = plan[visit_idx - 1].node;
            if let NodeType::Consumption = kind(prev) {
                None
            } else {
                match kind(visit.node) {
                    NodeType::Consumption => Some((visit.time, v)),
                    NodeType::Production => None,
                }
            }
        });

        // Disable all coeffiecients in the travel empty and travel at cap constraints
        let travel_empty_constr = model.get_constr_by_name(&"travel_empty")?.unwrap();
        let travel_at_cap_constr = model.get_constr_by_name(&"travel_at_cap")?.unwrap();
        let a = self
            .vars
            .l
            .iter()
            .flatten()
            .flatten()
            .map(|var| (*var, travel_empty_constr, 0.0));
        let b = self
            .vars
            .cap_violation
            .iter()
            .flatten()
            .map(|var| (*var, travel_at_cap_constr, 0.0));

        model.set_coeffs(a.chain(b))?;

        // set the correct coefficients of the travel empty and travel at capacity restricitons to 1
        let l = &self.vars.l;
        // var, constr, coeff for travel emtpy
        let a = prod_arrivals.clone().flat_map(|(t, v)| {
            (0..problem.products()).map(move |p| (l[t][v][p], travel_empty_constr, 1.0))
        });
        // var, constr, coeff for travel at capacity
        let b = cons_arrivals
            .clone()
            .map(|(t, v)| (self.vars.cap_violation[t][v], travel_at_cap_constr, 1.0));

        model.set_coeffs(a.chain(b))?;

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
        let revenue = self.vars.revenue.convert(&self.model)?;
        let violation = self.vars.violation.convert(&self.model)?;
        let timing = self.vars.timing.convert(&self.model)?;
        let spot = self.vars.spot.convert(&self.model)?;
        let a = self.vars.a.convert(&self.model)?;
        let cap_violation = self.vars.cap_violation.convert(&self.model)?;
        let travel_empty = self.vars.travel_empty.convert(&self.model)?;
        let travel_at_cap = self.vars.travel_at_cap.convert(&self.model)?;

        let v = F64Variables {
            w,
            x,
            s,
            l,
            revenue,
            spot,
            violation,
            timing,
            a,
            cap_violation,
            travel_empty,
            travel_at_cap,
        };
        Ok(v)
    }

    /// Fixes the semicont and integer variables and converts the MIP to an LP
    /// Model is updated, but not optimized
    pub fn fix(&mut self) -> grb::Result<()> {
        let fix_var = |var| -> grb::Result<()> {
            let value = self
                .model
                .get_obj_attr(grb::attr::X, var)
                .expect("failed to retrieve variable value");

            self.model.set_obj_attr(grb::attr::LB, var, value)?;
            self.model.set_obj_attr(grb::attr::UB, var, value)?;
            self.model
                .set_obj_attr(grb::attr::VType, var, grb::VarType::Continuous)?;
            Ok(())
        };

        // fix all semicont
        for var in self.vars.x.iter().flatten().flatten().flatten() {
            fix_var(var)?;
        }

        // fix all integer
        for var in self.vars.b.iter().flatten().flatten() {
            fix_var(var)?;
        }

        self.model.update()?;

        Ok(())
    }

    /// Fixes the semicont and integer variables and converts the LP to a MIP
    /// Model is updated, but not optimized
    pub fn unfix(&mut self) -> grb::Result<()> {
        let unfix_var = |var: &grb::Var, vtype: grb::VarType| -> grb::Result<()> {
            self.model.set_obj_attr(grb::attr::LB, var, 0.0)?;
            self.model.set_obj_attr(grb::attr::UB, var, f64::INFINITY)?;
            self.model.set_obj_attr(grb::attr::VType, var, vtype)?;
            Ok(())
        };
        // unfix all semicont
        for var in self.vars.x.iter().flatten().flatten().flatten() {
            unfix_var(var, grb::VarType::SemiCont)?;
        }

        // unfix all integer
        for var in self.vars.b.iter().flatten().flatten() {
            unfix_var(var, grb::VarType::Binary)?;
        }

        self.model.update()?;

        Ok(())
    }
}

/// Struct to hold the solution variables and objective terms
pub struct Solution<'a> {
    pub vars: &'a Variables,
    pub revenue: f64,
    pub spot: f64,
    pub timing: f64,
    pub violation: f64,
}
