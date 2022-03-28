use rand::{seq::SliceRandom, Rng};
use std::{
    cmp::{max, min},
    ops::Deref,
    sync::Arc,
};

use crate::{
    problem::{IndexCount, Vessel},
    solution::{routing::Plan, Visit},
};

use super::{Problem, Recombination, RoutingSolution};

pub struct PIX;

impl Recombination for PIX {
    fn apply(
        &mut self,
        problem: &Problem,
        left: &mut RoutingSolution,
        right: &mut RoutingSolution,
    ) {
        // Recombination technique based on looking at each vessel isolated.
        // The process consists of a preprocessing step, where the vessels are splitted into
        // three subsets.
        //     1. The vessels from which the routes will be used directly from the first parent
        //     2. The vessels from which the routes will be used directly from the second parent
        //     3. The remaining vessels that will be mixed from the two parents

        let mut vessel_indices = problem.indices::<Vessel>().collect::<Vec<usize>>();

        let mut rng = rand::thread_rng();

        let n_1 = rng.gen_range(1..=vessel_indices.len());
        let n_2 = rng.gen_range(1..=vessel_indices.len());
        vessel_indices.shuffle(&mut rng);

        let left_vessels = &vessel_indices[..min(n_1, n_2)];
        let right_vessels = &vessel_indices[min(n_1, n_2)..max(n_1, n_2)];
        let mixed_vessels = &vessel_indices[max(n_1, n_2)..];

        let mut routes = left_vessels
            .iter()
            .map(|i| left[*i].to_vec())
            .collect::<Vec<Vec<Visit>>>();

        routes.append(
            &mut right_vessels
                .iter()
                .map(|i| right[*i].to_vec())
                .collect::<Vec<Vec<Visit>>>(),
        );

        routes.append(&mut self.mixed(left, right, &mixed_vessels));

        // let child = RoutingSolution::new(Arc::new(*problem), routes);
    }

    fn with_probability(self, p: f64) -> super::Stochastic<Self>
    where
        Self: Sized,
    {
        super::Stochastic::new(p, self)
    }
}

impl PIX {
    fn mixed(
        &mut self,
        left: &mut RoutingSolution,
        right: &mut RoutingSolution,
        vessel_indices: &[usize],
    ) -> Vec<Vec<Visit>> {
        let mut rng = rand::thread_rng();

        let routes = vessel_indices
            .iter()
            .map(|i| {
                let split_point = rng.gen_range(0..min(left[*i].len(), right[*i].len()));
                [&left[*i][..split_point], &right[*i][split_point..]].concat()
            })
            .collect();

        routes
    }
}
