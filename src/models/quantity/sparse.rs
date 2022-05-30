use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    io::Write,
    ops::RangeInclusive,
};

use grb::{add_ctsvar, c, expr::GurobiSum, Expr, Model, Var};
use itertools::{iproduct, Itertools};
use log::debug;
use slice_group_by::GroupBy;

use crate::{
    problem::{NodeIndex, NodeType, Problem, ProductIndex, TimeIndex, VesselIndex},
    solution::routing::RoutingSolution,
};

#[derive(Debug)]
pub struct Variables {
    pub w: HashMap<(TimeIndex, NodeIndex, ProductIndex), Var>,
    pub x: HashMap<(TimeIndex, NodeIndex, VesselIndex, ProductIndex), Var>,
    pub s: HashMap<(TimeIndex, NodeIndex, ProductIndex), Var>,
    pub l: HashMap<(TimeIndex, VesselIndex, ProductIndex), Var>,
    pub a: HashMap<(TimeIndex, NodeIndex, ProductIndex), Var>,
    pub violation: Var,
    pub spot: Var,
    pub revenue: Var,
    pub timing: Var,
    pub travel_empty: Var,
    pub travel_at_cap: Var,
}

pub struct QuantityLp {
    pub model: Model,
    pub vars: Option<Variables>,
    pub semicont: bool,
    pub berth: bool,
}

#[macro_export]
macro_rules! time {
    ($name: expr, $bl: expr) => {{
        let start = std::time::Instant::now();
        $bl
        let end = std::time::Instant::now();
        let elapsed = (end - start).as_micros();
        println!("{} completed in {}us", $name, elapsed);
    }};
}

impl QuantityLp {
    fn multiplier(kind: NodeType) -> f64 {
        match kind {
            NodeType::Consumption => -1.0,
            NodeType::Production => 1.0,
        }
    }

    pub fn new(_: &Problem) -> grb::Result<Self> {
        let mut model = Model::new(&format!("quantities-sparse"))?;
        // Disable output logging.
        model.set_param(grb::param::OutputFlag, 0)?;
        // Use primal simplex, instead of the default concurrent solver. Reason: we will use multiple concurrent GAs
        model.set_param(grb::param::Method, 0)?;
        // Restrict to one thread. Also due to using concurrent GAs.
        model.set_param(grb::param::Threads, 1)?;

        Ok(QuantityLp {
            model,
            vars: None,
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
                        // let rate = problem.nodes()[v1.node].min_unloading_amount();
                        //let max_loading_time = (vessel.capacity() / rate).ceil() as usize;

                        let period = v1.time..=departure_time;

                        (v1.node, period)
                    }),
                )
            })
    }

    fn clear(&mut self) -> grb::Result<()> {
        /* if let Some(vars) = self.vars.take() {
            let Variables {
                w,
                x,
                s,
                l,
                a,
                violation,
                spot,
                revenue,
                timing,
                travel_empty,
                travel_at_cap,
            } = vars;

            for &var in w
                .values()
                .chain(x.values())
                .chain(s.values())
                .chain(l.values())
                .chain(a.values())
                .chain(
                    [
                        violation,
                        spot,
                        revenue,
                        timing,
                        travel_empty,
                        travel_at_cap,
                    ]
                    .iter(),
                )
            {
                self.model.remove(var)?;
            }
        }

        let constrs = self.model.get_constrs()?.to_vec();

        for constr in constrs {
            self.model.remove(constr)?;
        } */

        self.model = Model::new("blah")?;
        self.vars = None;

        // Disable output logging.
        self.model.set_param(grb::param::OutputFlag, 0)?;
        // Use primal simplex, instead of the default concurrent solver. Reason: we will use multiple concurrent GAs
        self.model.set_param(grb::param::Method, 0)?;
        // Restrict to one thread. Also due to using concurrent GAs.
        self.model.set_param(grb::param::Threads, 1)?;

        Ok(())
    }

    /// Set up the model to be ready to solve quantities for `solution`.
    pub fn configure(
        &mut self,
        solution: &RoutingSolution,
        _semicont: bool,
        _berth: bool,
        _tight: bool,
    ) -> grb::Result<()> {
        // Some stuff that will be useful later
        self.clear()?;
        let model = &mut self.model;
        let problem = solution.problem();
        let p = problem.products();
        let t = problem.timesteps();
        let v = problem.vessels().len();
        let n = problem.nodes().len();

        /// To avoid construction of format strings for variable and constraint names, while
        /// making it easy to remove it (by commenting out this)
        /* macro_rules! format {
            ($x:expr $(, $xs:expr)*) => {
                ""
            };
        } */

        macro_rules! insert {
            ($m: expr, $k: expr, $b: expr) => {{
                let x: grb::Result<&mut Var> = match $m.entry($k) {
                    Entry::Occupied(e) => Ok(e.into_mut()),
                    Entry::Vacant(e) => Ok(e.insert(add_ctsvar!(
                        model,
                        name: &format!("{}_{:?}", stringify!($m), $k),
                        bounds: $b
                    )?)),
                };
                x
            }};
        }

        // The timelines for each node and each vessel (when stuff happens)
        // Note that to ensure inventory is not violated, we will define t_n for [0, t] as well.
        // We also add the `available_from` of each vessel.
        let mut t_n = vec![[0, t].into_iter().collect::<HashSet<_>>(); problem.nodes().len()];
        let mut t_v = problem
            .vessels()
            .iter()
            .map(|v| [v.available_from(), t].into_iter().collect::<HashSet<_>>())
            .collect::<Vec<_>>();

        // The loading/unloading variables
        let mut x = HashMap::new();
        let mut revenue_expr: Expr = 0.0_f64.into();

        for (v, nodes) in Self::active(solution) {
            let l_cap = problem.vessels()[v].capacity();
            for (n, times) in nodes {
                let node = &problem.nodes()[n];
                let unit_revenue = node.revenue();
                let x_cap = node.max_loading_amount();
                // We need to include enough time before arrival to include the possibility of avoiding shortage
                // by using the spot market
                let a_maxt = node.spot_market_limit_per_time();
                let a_max = node.spot_market_limit();
                let spot = (a_max / a_maxt).ceil() as usize;
                let pre = spot.max(*times.start()) - spot;

                for t in pre..*times.start() {
                    t_n[n].insert(t);
                }

                /*                 // Some other code relies on the loading being defined at each visit.
                // Not strictly needed for the sparse model.
                t_v[v].insert(*times.start()); */
                let end = *times.end();

                for t in times {
                    t_v[v].insert(t);
                    t_n[n].insert(t);

                    for p in 0..problem.products() {
                        let var = *insert!(x, (t, n, v, p), 0.0..l_cap.min(x_cap)).unwrap();
                        revenue_expr = revenue_expr + var * unit_revenue;
                    }
                }

                // The vessel an port inventory needs to be defined for one additional timestep, to avoid "cheating" the inventory
                t_v[v].insert(end + 1);
                t_n[n].insert(end + 1);
            }
        }

        // The timeline of each node needs to be ordered by time
        let t_n: Vec<Vec<usize>> = t_n
            .into_iter()
            .map(|xs| xs.into_iter().sorted().collect())
            .collect();

        let t_v: Vec<Vec<usize>> = t_v
            .into_iter()
            .map(|xs| xs.into_iter().sorted().collect())
            .collect();

        // Both l and s are at the `start` of the period, i.e. before any loading/unloading happens.
        let mut l = HashMap::new();
        for (v, timeline) in t_v.iter().enumerate() {
            let cap = problem.vessels()[v].capacity();
            // For every "discontuity", we need to include the timestep after, to prevent the vessel from "cheating", its inventory by loading in the last possible timestep
            // Each group is a non-empty continuous range, by construction
            for group in timeline.linear_group_by(|&a, &b| a + 1 == b) {
                let first = *group.first().unwrap();
                let last = *group.last().unwrap();
                for t in first..=last.min(t) {
                    for p in 0..p {
                        insert!(l, (t, v, p), 0.0..cap)?;
                    }
                }
            }
        }

        let mut s = HashMap::new();
        let mut w = HashMap::new();
        let mut a = HashMap::new();

        for (n, timeline) in t_n.iter().enumerate() {
            let node = &problem.nodes()[n];
            let a_maxt = node.spot_market_limit_per_time();
            let a_max = node.spot_market_limit();
            let mut a_tot = Expr::from(0.0_f64);
            for group in timeline.linear_group_by(|&a, &b| a + 1 == b) {
                for &t in group {
                    for p in 0..p {
                        match node.r#type() {
                            NodeType::Consumption => insert!(s, (t, n, p), ..node.capacity()[p])?,
                            NodeType::Production => insert!(s, (t, n, p), 0.0..)?,
                        };
                        insert!(w, (t, n, p), 0.0..)?;
                        a_tot = a_tot + *insert!(a, (t, n, p), 0.0..a_maxt)?;
                    }
                }

                /* // We need to observe whether violation occurs AFTER day `last` = `end - 1`
                let end = group.last().unwrap() + 1;
                for p in 0..p {
                    match node.r#type() {
                        NodeType::Consumption => insert!(s, (end, n, p), ..node.capacity()[p])?,
                        NodeType::Production => insert!(s, (end, n, p), 0.0..)?,
                    };
                    insert!(w, (end, n, p), 0.0..)?;
                } */
            }

            // Total `a` can not exceed the limit
            model.add_constr(&format!("c_a_{n}"), c!(a_tot <= a_max))?;
        }

        // Initial inventory for vessels
        for (v, p) in iproduct!(0..v, 0..p) {
            let vessel = &problem.vessels()[v];
            let t0 = vessel.available_from();
            let initial = vessel.initial_inventory()[p];
            let name = format!("l_init_{:?}", (v, p));
            model.add_constr(&name, c!(l[&(t0, v, p)] == initial))?;
        }

        // Inventory conservation for vessels
        for (v, timeline) in t_v.iter().enumerate() {
            for (&t1, &t2) in timeline.iter().zip(&timeline[1..]) {
                for p in 0..p {
                    let i = |i: usize| Self::multiplier(problem.nodes()[i].r#type());
                    let name = format!("l_bal_{:?}", (t2, v, p));
                    let lhs = l[&(t2, v, p)];
                    let rhs = l[&(t1, v, p)]
                        + (0..n)
                            .filter_map(|n| {
                                let x = *x.get(&(t1, n, v, p))?;
                                Some(i(n) * x)
                            })
                            .grb_sum();
                    model.add_constr(&name, c!(lhs == rhs))?;
                }
            }
        }

        // Initial inventory for nodes
        for (n, p) in iproduct!(0..n, 0..p) {
            let s0 = problem.nodes()[n].initial_inventory()[p];
            let name = &format!("s_init_{:?}", (n, p));
            model.add_constr(name, c!(s[&(0, n, p)] == s0))?;
        }

        // Soft inventory constraints
        for (&(t, n, p), &s) in &s {
            let node = &problem.nodes()[n];
            let w = w[&(t, n, p)];
            // Whether to enable/disable upper and lower soft constraint, respectively
            let c = match node.r#type() {
                NodeType::Consumption => c!(s >= -w),
                NodeType::Production => c!(s <= node.capacity()[p] + w),
            };

            model.add_constr(&format!("s_soft_{:?}", (t, n, p)), c)?;
        }

        // Inventory conservation for nodes
        for (n, timeline) in t_n.iter().enumerate() {
            let node = &problem.nodes()[n];
            let i = Self::multiplier(node.r#type());

            for (&t1, &t2) in timeline.iter().zip(&timeline[1..]) {
                for p in 0..p {
                    let external = (0..v).filter_map(|v| x.get(&(t1, n, v, p))).grb_sum();
                    let internal = node.inventory_change(t1, t2 - 1, p);
                    let spot = a[&(t1, n, p)];
                    let constr =
                        c!(s[&(t2, n, p)] == s[&(t1, n, p)] + internal - i * (external + spot));

                    model.add_constr(&format!("s_bal_{:?}", (t2, n, p)), constr)?;
                }
            }
        }

        let spot_expr = a
            .iter()
            .map(|(&(t, n, p), var)| {
                let node = &problem.nodes()[n];
                let unit_price = node.spot_market_unit_price();
                let discount = node.spot_market_discount_factor().powi(t as i32);
                (*var) * unit_price * discount
            })
            .grb_sum();

        let revenue = add_ctsvar!(model, name: "revenue", bounds: 0.0..)?;
        let violation = add_ctsvar!(model, name: "violation", bounds: 0.0..)?;
        let spot = add_ctsvar!(model, name: "spot", bounds: 0.0..)?;
        let timing = add_ctsvar!(model, name: "timing", bounds: 0.0..)?;

        model.add_constr("c_revenue", c!(revenue == revenue_expr))?;
        model.add_constr("c_violation", c!(violation == w.values().grb_sum()))?;
        model.add_constr("c_spot", c!(spot == spot_expr))?;
        model.add_constr("c_timing", c!(timing == 0.0_f64))?;

        // Note: these aren't really used here, but we'll just force them to 0 so that other
        // code can rely on there being variables with these names.
        let travel_empty = add_ctsvar!(model, name: "travel_empty", bounds: 0.0..)?;
        let travel_at_cap = add_ctsvar!(model, name: "travel_at_cap", bounds: 0.0..)?;
        model.add_constr("c_travel_empty", c!(travel_empty == 0.0_f64))?;
        model.add_constr("c_travel_at_cap", c!(travel_at_cap == 0.0_f64))?;

        let obj = 1.0e8_f64 * violation + 0.5_f64 * spot - 1e-6_f64 * revenue
            + 1e-12_f64 * timing
            + travel_empty
            + travel_at_cap;

        model.set_objective(obj, grb::ModelSense::Minimize)?;

        self.vars = Some(Variables {
            w,
            x,
            s,
            l,
            a,
            violation,
            spot,
            revenue,
            timing,
            travel_empty,
            travel_at_cap,
        });

        Ok(())
    }

    /// Solves the model for the current configuration. Returns the variables
    /// Should also include dual variables for the upper bounds on x, but this is not implemented yet
    pub fn solve(&mut self) -> grb::Result<&Variables> {
        self.model.optimize()?;
        match self.model.status()? {
            grb::Status::Optimal => (),
            status => {
                println!("Status = {status:?}");

                let m = &mut self.model;
                m.compute_iis()?;
                let constrs = m.get_constrs()?; // all constraints in model
                let iis_constrs = m
                    .get_obj_attr_batch(grb::attr::IISConstr, constrs.iter().copied())?
                    .into_iter()
                    .zip(constrs)
                    // IISConstr is 1 if constraint is in the IIS, 0 otherwise
                    .filter_map(|(is_iis, c)| {
                        if is_iis > 0 {
                            Some(m.get_obj_attr(grb::attr::ConstrName, c).unwrap())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                let mut file = std::fs::File::create(format!("iis.txt")).unwrap();
                write!(file, "{:#?}", iis_constrs).unwrap();
                panic!();
            }
        }

        /* //debug!("{:?}", self.vars);
        let vars = self.vars.as_ref().unwrap();

        for (vs, name) in [(&vars.a, "a"), (&vars.l, "l"), (&vars.s, "s")] {
            let keys = vs.keys().sorted();

            for k in keys {
                let value = self.model.get_obj_attr(grb::attr::X, &vs[k])?;
                println!("{name}_{k:?} = {value}");
            }
        }

        let keys = vars.x.keys().sorted();

        for k in keys {
            let value = self.model.get_obj_attr(grb::attr::X, &vars.x[k])?;
            println!("x_{k:?} = {value}");
        } */

        self.model.write("sparse.lp");

        Ok(self.vars.as_ref().expect("should exist"))
    }
}
