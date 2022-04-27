use std::f64::consts::PI;

use float_ord::FloatOrd;
use itertools::{iproduct, Itertools};

use crate::{
    ga::Mutation,
    problem::Problem,
    solution::{
        routing::{Plan, RoutingSolution},
        Visit,
    },
    utils::GetPairMut,
};

pub struct SwapStar;

impl SwapStar {
    /// Evaluates the cost of inserting the given visit into the given index in the given plan
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
    fn cost_between(
        prev: &Visit,
        inserted: &Visit,
        next: Option<&Visit>,
        problem: &Problem,
    ) -> f64 {
        match next {
            Some(next) => {
                problem.distance(prev.node, inserted.node)
                    + problem.distance(inserted.node, next.node)
            }
            None => problem.distance(prev.node, inserted.node),
        }
    }

    fn best_without_v(top_three: &Vec<usize>, plan: &Plan, visit_to_remove: &Visit) -> usize {
        let a = top_three
            .into_iter()
            .filter(|visit_idx| {
                let curr = &plan[**visit_idx];
                if curr == visit_to_remove {
                    false
                } else if &plan[*visit_idx - 1] == visit_to_remove {
                    false
                } else {
                    true
                }
            })
            .next();
        *a.unwrap()
    }

    fn _get_best(
        pos1: usize,
        to_remove_idx: usize,
        visit_to_insert: &Visit,
        visit_to_remove: &Visit,
        plan: &Plan,
        problem: &Problem,
    ) -> (usize, f64) {
        // the cost of inserting not associated with the visit that is removed
        let cost1 = Self::cost_between(&plan[pos1 - 1], visit_to_insert, plan.get(pos1), problem);
        // the cost of inserting at the place of the visit to remove
        let cost2 = Self::cost_between(
            &plan[to_remove_idx - 1],
            visit_to_insert,
            plan.get(to_remove_idx + 1),
            problem,
        );
        // gain from removing the visit to remove
        let gain = Self::cost_between(
            &plan[to_remove_idx - 1],
            visit_to_remove,
            plan.get(to_remove_idx + 1),
            problem,
        );
        if cost1 < cost2 {
            (pos1, cost1 - gain)
        } else {
            (to_remove_idx, cost2 - gain)
        }
    }

    pub fn find_top_three(visit: &Visit, plan: &Plan, problem: &Problem) -> Vec<usize> {
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
            let mut res = Vec::new();
            for _ in 0..3 {
                let (index_to_remove, element) = costs
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, e)| FloatOrd(e.1))
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

    fn best_swap(
        plan1: &Plan,
        plan2: &Plan,
        problem: &Problem,
    ) -> Option<((usize, usize), (usize, usize))> {
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

    fn new_visit(
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

    fn apply_swap(
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

    fn overlapping(i: usize, j: usize, solution: &RoutingSolution, problem: &Problem) -> bool {
        let sector1 = CircleSector::from_route(&solution[i], problem);
        let sector2 = CircleSector::from_route(&solution[j], problem);
        overlap(&sector1, &sector2)
    }

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

struct CircleSector {
    pub start: i16,
    pub end: i16,
}

impl CircleSector {
    pub fn new(point: i16) -> CircleSector {
        let start = point;
        let end = point;
        CircleSector { start, end }
    }

    pub fn is_enclosed(&self, point: i16) -> bool {
        modulo(point - self.start) <= modulo(self.end - self.start)
    }

    pub fn extend(&mut self, point: i16) {
        if !self.is_enclosed(point) {
            if modulo(point - self.end) <= modulo(self.start - point) {
                self.end = point;
            } else {
                self.start = point;
            }
        }
    }

    fn from_route(plan: &Plan, problem: &Problem) -> CircleSector {
        let points = plan
            .iter()
            .map(|visit| {
                let n = &problem.nodes()[visit.node];
                let (x, y) = n.coordinates();
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

fn modulo(i: i16) -> i16 {
    (i % 16536 + 16536) % 16536
}

fn overlap(sector1: &CircleSector, sector2: &CircleSector) -> bool {
    (modulo(sector2.start - sector1.start) <= modulo(sector1.end - sector1.start))
        || (modulo(sector1.start - sector2.start) <= modulo(sector2.end - sector2.start))
}
