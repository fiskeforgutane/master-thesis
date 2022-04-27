use std::{
    collections::{HashMap, HashSet},
    ops::{Range, RangeInclusive},
};

use grb::{add_ctsvar, c, expr::GurobiSum, Model, Var};
use itertools::{iproduct, Itertools};

use crate::{
    models::utils::AddVars,
    problem::{NodeIndex, NodeType, Problem, ProductIndex, TimeIndex, VesselIndex},
    solution::routing::RoutingSolution,
};

pub struct Variables {
    pub w: Vec<Vec<Vec<Var>>>,
    pub x: Vec<Vec<Vec<Vec<Var>>>>,
    pub s: Vec<Vec<Vec<Var>>>,
    pub l: Vec<Vec<Vec<Var>>>,
    pub b: Vec<Vec<Vec<Var>>>,
    pub a: Vec<Vec<Vec<Var>>>,
    pub violation: Var,
    pub spot: Var,
    pub revenue: Var,
    pub timing: Var,
}

pub struct SparseQuantityLp {
    pub model: Model,
    pub vars: Variables,
    pub semicont: bool,
    pub berth: bool,
}

impl SparseQuantityLp {
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

        Ok(SparseQuantityLp {
            model,
            vars: todo!(),
            semicont: false,
            berth: false,
        })
    }

    pub fn active<'s>(
        solution: &'s RoutingSolution,
    ) -> impl Iterator<
        Item = (
            VesselIndex,
            impl Iterator<Item = (NodeIndex, RangeInclusive<usize>)> + 's,
        ),
    > + 's {
        let problem = solution.problem();
        let p = problem.products();

        solution
            .iter_with_terminals()
            .enumerate()
            .map(move |(v, plan)| {
                let vessel = &problem.vessels()[v];

                (
                    v,
                    plan.tuple_windows().map(move |(v1, v2)| {
                        // We must determine when we need to leave `v1.node` in order to make it to `v2.node` in time.
                        // Some additional arithmetic parkour is done to avoid underflow cases (damn usizes).
                        let travel_time = problem.travel_time(v1.node, v2.node, vessel);
                        let departure_time = v2.time - v2.time.min(travel_time);
                        // In addition, we can further restrict the active time periods by looking at the longest possible time
                        // the vessel can spend at the node doing constant loading/unloading.
                        let rate = problem.nodes()[v1.node].min_unloading_amount();
                        let max_loading_time = (vessel.capacity() / rate).ceil() as usize;

                        let period = v1.time..=departure_time.min(v1.time + max_loading_time);

                        (v1.node, period)
                    }),
                )
            })
    }

    /// Set up the model to be ready to solve quantities for `solution`.
    pub fn configure(
        &mut self,
        solution: &RoutingSolution,
        semicont: bool,
        berth: bool,
    ) -> grb::Result<()> {
        self.semicont = semicont;

        self.berth = berth;

        // By default: disable `all` variables
        let model = &mut self.model;
        let problem = solution.problem();

        // The timelines for each node (where stuff happens)
        let mut t_n = vec![HashSet::new(); problem.nodes().len()];
        let mut t_v = vec![Vec::new(); problem.vessels().len()];

        let mut x = HashMap::new();
        let mut l = HashMap::new();
        let mut w = HashMap::new();
        let mut a = HashMap::new();
        let mut s = HashMap::new();

        for (v, nodes) in Self::active(solution) {
            let vessel = problem.vessels()[v];
            let l_cap = vessel.capacity();
            let available = vessel.available_from();
            for (n, times) in nodes {
                let node = problem.nodes()[n];
                let x_cap = node.max_loading_amount();
                let pre = (*times).start().min(1) - 1;
                let post = *times.end() + 1;
                let a_cap = node.spot_market_limit_per_time();
                let a_tot = node.spot_market_limit();

                for t in times {
                    t_v[v].push(t);
                    t_n[n].insert(t);

                    for p in 0..problem.products() {
                        x.insert(
                            (t, n, v, p),
                            add_ctsvar!(model, bounds: 0.0..l_cap.min(x_cap)),
                        );
                        l.entry((t, v, p))
                            .or_insert_with(|| add_ctsvar!(model, bounds: 0.0..l_cap));

                        a.entry((t, n, p))
                            .or_insert_with(|| add_ctsvar!(model, bounds: 0.0..a_cap));

                        // TODO: should probably sum over capacity. However it doesn't matter for MIRPLIB, since there
                        // is only one produ dct.
                        s.entry((t, n, p))
                            .or_insert_with(|| add_ctsvar!(model, bounds: 0.0..x_cap));
                    }
                }

                // The load should be available in the timestep before and after, so that we can link them together
                for p in (0..problem.products()) {
                    l.entry((pre, v, p))
                        .or_insert_with(|| add_ctsvar!(model, name: "", bounds: 0.0..l_cap));
                    l.entry((post, v, p))
                        .or_insert_with(|| add_ctsvar!(model, name: "", bounds: 0.0..l_cap));
                }

                // Violation should be available for enough time before arriving to be able to do something to stop shortage from occuring
                let mut cumulative = 0.0;
                for t in (0..=pre).rev() {
                    if cumulative > a_cap {
                        break;
                    }

                    for p in 0..problem.products() {
                        a.entry((t, n, p))
                            .or_insert_with(|| add_ctsvar!(model, bounds: 0.0..a_cap));
                    }

                    cumulative += node.inventory_changes()[t]
                }
            }
        }

        // Should be defined over `active`
        let x = (t, n, v, p).cont(&mut model, "x")?;
        // B is only needed when berth == true, and should be defined over active (no p though)
        // let b = (t, n, v).cont(&mut model, "b")?;
        // Load should be defined over active, plus the timestep after it leaves
        let l = (t + 1, v, p).cont(&mut model, "l")?;
        // Violation should be defined over active. plus the timestep before + after
        let w = (t + 1, n, p).cont(&mut model, "w")?;
        // Storage should be defined over active, plus the timestep before + after
        let s = (t + 1, n, p).free(&mut model, "s")?;
        // Spot market should be defined over `active`, plus the maximum number of time periods it could supply spot at maximum rate.
        let a = (t, n, p).cont(&mut model, "a")?;

        Ok(())
    }

    /// Solves the model for the current configuration. Returns the variables
    /// Should also include dual variables for the upper bounds on x, but this is not implemented yet
    pub fn solve(&mut self) -> grb::Result<()> {
        self.model.optimize()?;

        Ok(&self.vars)
    }
}
