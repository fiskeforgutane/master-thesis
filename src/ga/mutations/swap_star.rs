use std::f64::consts::PI;

use float_ord::FloatOrd;
use itertools::{iproduct, Itertools};
use log::trace;

use crate::{
    ga::Mutation,
    problem::Problem,
    solution::{
        routing::{Plan, RoutingSolution},
        Visit,
    },
    utils::GetPairMut,
};

/// A local search swap procedure that exchanges two visits v and v' from different plans r and r' without and insertion in place.
/// In this process, v can be inserted in any position in r', and v' can likewise be inserted in any position of r.
/// The procedure only evaluates swaps using distance, hence deliveries are ignored.
pub struct SwapStar;

impl SwapStar {
    /// Evaluates the cost of inserting the given `visit` into the given `idx` in the given `plan`
    fn evaluate(idx: usize, visit: &Visit, plan: &Plan, problem: &Problem) -> f64 {
        let dist = |n1, n2| problem.distance(n1, n2);
        if idx == plan.len() {
            let prev = plan[idx - 1];
            dist(prev.node, visit.node)
        } else {
            let (prev, next) = (plan[idx - 1], plan[idx]);
            dist(prev.node, visit.node) + dist(visit.node, next.node) - dist(prev.node, next.node)
        }
    }

    /// cost of inserting a visit between two other visits
    ///
    /// ## Arguments
    ///
    /// * `prev` - the visit prior to the one inserted
    /// * `inserted` - the visit in the middle
    /// * `next` - the next visit, if any
    fn cost_between(visits: Vec<Visit>, problem: &Problem) -> f64 {
        visits
            .iter()
            .fold((0.0, None), |acc: (f64, Option<Visit>), x| match acc.1 {
                Some(v) => (problem.distance(v.node, x.node), Some(*x)),
                None => (0.0, Some(*x)),
            })
            .0
    }

    /// Finds the best entry point among the top three given that does not have an edge into, nor out from the `visit_to_remove`
    ///
    /// ## Arguments
    ///
    /// * `top_three` - The top three (might be fewer if the plan was shorter) best places to insert, sorted from best to worst
    /// * `plan` - The plan where the insertions will be made
    /// * `visit_to_remove` - The visit that will be removed from `plan`
    fn best_without_v(top_three: &Vec<usize>, plan: &Plan, visit_to_remove: &Visit) -> usize {
        trace!("in best without");
        trace!(
            "top three: {:?}, \nplan: {:?}, \nto_remove: {:?}",
            top_three,
            plan.iter().map(|v| (v.node, v.time)).collect::<Vec<_>>(),
            visit_to_remove
        );
        let a = top_three
            .into_iter()
            .filter(|visit_idx| {
                let curr = plan.get(**visit_idx);
                match curr {
                    Some(curr) => {
                        if curr == visit_to_remove {
                            false
                        } else if &plan[*visit_idx - 1] == visit_to_remove {
                            false
                        } else {
                            true
                        }
                    }
                    None => {
                        if &plan[*visit_idx - 1] == visit_to_remove {
                            false
                        } else {
                            true
                        }
                    }
                }
            })
            .next();
        *a.unwrap()
    }

    /// Checks whether it is best to insert the `visit_to_insert` into the current place of `visit_to_remove` or in the best place that does not
    /// have an edge into nor out from the `visit_to_remove`
    ///
    /// ## Arguments
    /// * `pos1` - The best place to insert, that does not have and edge into, nor out from, `visit_to_remove`
    /// * `to_remove_idx` - The index in the `plan` of the `visit_to_remove`
    /// * `visit_to_insert` - The visit to insert
    /// * `visit_to_remove` - The visit to remove
    /// * `plan` - The plan of `visit_to_remove`
    /// * `problem` - The problem
    ///
    /// ## Returns
    /// The best position to insert `visit_to_insert` into `plan`, as well as the cost associated with doing this
    fn _get_best(
        pos1: usize,
        to_remove_idx: usize,
        visit_to_insert: &Visit,
        visit_to_remove: &Visit,
        plan: &Plan,
        problem: &Problem,
    ) -> (usize, f64) {
        trace!("pos1: {}", pos1);
        trace!("to_remove_idx: {:?}", to_remove_idx);
        // the cost of inserting not associated with the visit that is removed
        let cost1 = Self::cost_between(
            vec![plan.get(pos1 - 1), Some(visit_to_insert), plan.get(pos1)]
                .into_iter()
                .filter_map(|v| match v {
                    Some(x) => Some(*x),
                    None => None,
                })
                .collect(),
            problem,
        );
        // the cost of inserting at the place of the visit to remove
        let cost2 = Self::cost_between(
            vec![
                plan.get(to_remove_idx - 1),
                Some(visit_to_insert),
                plan.get(to_remove_idx + 1),
            ]
            .into_iter()
            .filter_map(|v| match v {
                Some(x) => Some(*x),
                None => None,
            })
            .collect(),
            problem,
        );
        // gain from removing the visit to remove
        let gain = Self::cost_between(
            vec![
                plan.get(to_remove_idx - 1),
                Some(visit_to_remove),
                plan.get(to_remove_idx + 1),
            ]
            .into_iter()
            .filter_map(|v| match v {
                Some(x) => Some(*x),
                None => None,
            })
            .collect(),
            problem,
        );
        if cost1 < cost2 {
            (pos1, cost1 - gain)
        } else {
            (to_remove_idx, cost2 - gain)
        }
    }

    /// Identify the two three best places to insert the given visit into the given plan
    /// Returns a vector sorted from best position to worst. The length of the vector might be shorter than 3 if
    /// the plan is shorter than 3
    pub fn find_top_three(visit: &Visit, plan: &Plan, problem: &Problem) -> Vec<usize> {
        trace!(
            "finding top three of inserting {:?}, from plan {:?}",
            visit,
            plan.iter().map(|v| (v.node, v.time)).collect::<Vec<_>>()
        );

        // omits origin as nothing should be inserted prior to origin.
        let mut costs = (1..=plan.len())
            .map(|idx| (idx, Self::evaluate(idx, visit, plan, problem)))
            .collect::<Vec<_>>();

        // if there are less than three elements, return the entire list
        let mut res = Vec::new();
        if costs.len() < 3 {
            res = costs;
        } else {
            // find the three cheapest insertions
            for _ in 0..3 {
                let (index_to_remove, element) = costs
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, e)| FloatOrd(e.1))
                    .unwrap();

                res.push(*element);

                costs.swap_remove(index_to_remove);
            }
        }

        res.into_iter()
            .sorted_by_key(|(_, cost)| FloatOrd(*cost))
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns the best swap to perform between `plan1` and `plan2`
    ///
    /// ## Returns
    /// Two tuples consisting of the visit to remove from the plan and where to insert it in the other plan
    pub fn best_swap(
        plan1: &Plan,
        plan2: &Plan,
        problem: &Problem,
    ) -> Option<((usize, usize), (usize, usize))> {
        trace!("plan1.len():{}", plan1.len());
        trace!("plan2.len():{}", plan2.len());
        if plan1.len() < 3 || plan2.len() < 3 {
            return None;
        }
        // best indices in plan1 to insert visits from plan2
        let best_plan1 = plan2
            .iter()
            .map(|visit| Self::find_top_three(visit, plan1, problem))
            .collect::<Vec<_>>();
        // best indices in plan2 to insert visits from plan1
        let best_plan2 = plan1
            .iter()
            .map(|visit| Self::find_top_three(visit, plan2, problem))
            .collect::<Vec<_>>();

        let mut best = 0.0;
        let mut best_swap = None;
        for (i, visit1) in plan1[1..].iter().enumerate() {
            for (j, visit2) in plan2[1..].iter().enumerate() {
                // the best place to insert visit1 in plan2, that is not right before, nor after visit2
                let k = Self::best_without_v(&best_plan2[i], plan2, visit2);
                // the best place to insert visit2 in plan1, that is not right before, nor after visit1
                let k_marked = Self::best_without_v(&best_plan1[j], plan1, visit1);

                // cost of inserting visit1 into the best place in plan2
                let (pos1, cost1) = Self::_get_best(k, j, visit1, visit2, plan2, problem);
                let (pos2, cost2) = Self::_get_best(k_marked, i, visit2, visit1, plan1, problem);

                if cost1 + cost2 < best {
                    best = cost1 + cost2;
                    best_swap = Some(((i, pos1), (j, pos2)));
                }
            }
        }
        best_swap
    }

    /// Gets the time to insert visit `to_insert` into `plan_idx`
    /// Tries to add it at a time step when the vessel can reach the node,
    /// and if not, the time step between the between the previous visit and next is selected
    fn get_new_time(
        plan_idx: usize,
        into_idx: usize,
        to_insert: Visit,
        solution: &RoutingSolution,
    ) -> usize {
        let plan = &solution[plan_idx];
        let prev = plan[into_idx - 1];
        let next = plan.get(into_idx);
        let vessel = &solution.problem().vessels()[plan_idx];
        let required_diff = solution
            .problem()
            .travel_time(prev.node, to_insert.node, vessel);
        match next {
            Some(v) => {
                if v.time - prev.time > required_diff {
                    prev.time + required_diff
                } else {
                    (v.time - prev.time) / 2
                }
            }
            None => prev.time + required_diff,
        }
    }

    /// Create a new visit to insert into `plan_idx` at `into_idx`
    pub fn new_visit(
        plan_idx: usize,
        into_idx: usize,
        to_insert: Visit,
        solution: &RoutingSolution,
    ) -> Visit {
        Visit {
            node: to_insert.node,
            time: Self::get_new_time(plan_idx, into_idx, to_insert, solution),
        }
    }

    /// Apply the swap
    ///
    /// ## Arguments
    ///
    /// * `solution`- The routing solution
    /// * `plan1_idx` - The index of the first plan
    /// * `plan2_idx` - The index of the second plan
    /// * `into_plan1` - The visit to insert into the first plan
    /// * `into_plan2` - The visit to insert into the second plan
    /// * `remove_from_1` - The index of the visit to remove from the first plan
    /// * `remove_from_2` - The index of the visit to remove from the second plan
    pub fn apply_swap(
        solution: &mut RoutingSolution,
        plan1_idx: usize,
        plan2_idx: usize,
        into_plan1: Visit,
        into_plan2: Visit,
        remove_from_1: usize,
        remove_from_2: usize,
    ) {
        let mutator = &mut solution.mutate();
        let (plan1, plan2) = mutator.get_pair_mut(plan1_idx, plan2_idx);

        // start by removing visit1 from plan1 and visit2 from plan2
        plan1.mutate().remove(remove_from_1);
        plan2.mutate().remove(remove_from_2);

        plan1.mutate().push(into_plan1);
        plan1.mutate().fix();
        plan2.mutate().push(into_plan2);
        plan2.mutate().fix();
    }

    /// Checks if the plans at index `i` and `j` have overlapping polar sectors
    pub fn overlapping(i: usize, j: usize, solution: &RoutingSolution, problem: &Problem) -> bool {
        let sector1 = CircleSector::from_route(&solution[i], problem);
        trace!("sector1: {:?}", sector1);
        let sector2 = CircleSector::from_route(&solution[j], problem);
        trace!("sector2: {:?}", sector2);
        overlap(&sector1, &sector2)
    }

    /// Returns a vector with all pairs of plans that have overlapping polar sectors
    fn swap_star_plans<'s>(
        problem: &'s Problem,
        solution: &'s RoutingSolution,
    ) -> impl Iterator<Item = (usize, usize)> + 's {
        iproduct!(0..solution.len(), 0..solution.len())
            .filter(|(i, j)| i != j)
            .filter(|(i, j)| Self::overlapping(*i, *j, solution, problem))
    }
}

impl Mutation for SwapStar {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        let routes = Self::swap_star_plans(problem, solution).collect::<Vec<_>>();
        for (r1, r2) in routes {
            let swap = Self::best_swap(&solution[r1], &solution[r2], problem);
            if let Some(((v1, p1), (v2, p2))) = swap {
                let into_plan2 = Self::new_visit(r2, p1, solution[r1][v1], &solution);
                let into_plan1 = Self::new_visit(r1, p2, solution[r2][v2], &solution);
                Self::apply_swap(solution, r1, r2, into_plan1, into_plan2, v1, v2);
            }
        }
    }
}

#[derive(Debug)]
/// A circle sector represented by a start angle and an end angle. Angles are given in the discrete range [0,16535]
struct CircleSector {
    pub start: i16,
    pub end: i16,
}

impl CircleSector {
    pub fn new(angle: i16) -> CircleSector {
        let start = angle;
        let end = angle;
        CircleSector { start, end }
    }

    /// Checks if the given `angle` is enclosed in `self`
    pub fn is_enclosed(&self, angle: i16) -> bool {
        modulo(angle - self.start) <= modulo(self.end - self.start)
    }

    /// Extends the interval of `self`
    pub fn extend(&mut self, angle: i16) {
        if !self.is_enclosed(angle) {
            if modulo(angle - self.end) <= modulo(self.start - angle) {
                self.end = angle;
            } else {
                self.start = angle;
            }
        }
    }

    /// creates the circle sector of the given `plan`
    fn from_route(plan: &Plan, problem: &Problem) -> CircleSector {
        let center = problem.center();
        trace!("problem center: {:?}", problem.center());
        let points = plan
            .iter()
            .map(|visit| {
                let n = &problem.nodes()[visit.node];
                let (x, y) = (n.coordinates().0 - center.0, n.coordinates().1 - center.1);
                let polar = Self::cartesian_to_polar(x, y)
                    .expect("failed to convert from cartesian to polar");

                // scale the angle from radians to [0,16535]
                f64::round((polar / (2.0 * PI)) * 16535.0) as i16
            })
            .collect::<Vec<_>>();
        let mut sector = CircleSector::new(points[0]);
        points.into_iter().for_each(|p| sector.extend(p));
        sector
    }

    /// Returns the angle to add to the given cartesian coordinate depending on which quadrant it is in
    fn quadrant(x: f64, y: f64) -> Option<f64> {
        if x >= 0.0 && y >= 0.0 {
            return Some(0.0);
        } else if x < 0.0 && y > 0.0 {
            return Some(PI);
        } else if x < 0.0 && y < 0.0 {
            return Some(PI);
        } else if x > 0.0 && y < 0.0 {
            return Some(2.0 * PI);
        }
        None
    }

    /// Returns the polar angle in radians
    fn cartesian_to_polar(x: f64, y: f64) -> Option<f64> {
        let q = Self::quadrant(x, y);
        match q {
            Some(q) => Some((y / x).atan() + q),
            None => None,
        }
    }
}

/// Positive modulo operation of i
fn modulo(i: i16) -> i16 {
    (i % 16536 + 16536) % 16536
}

/// Checks if two circle sectors overlap
fn overlap(sector1: &CircleSector, sector2: &CircleSector) -> bool {
    (modulo(sector2.start - sector1.start) <= modulo(sector1.end - sector1.start))
        || (modulo(sector1.start - sector2.start) <= modulo(sector2.end - sector2.start))
}
