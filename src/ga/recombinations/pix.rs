use rand::{seq::SliceRandom, Rng};
use std::cmp::{max, min};

use crate::{problem::{Vessel, Problem}, ga::Recombination, solution::routing::RoutingSolution};

#[derive(Debug, Clone)]
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

        let mut indices = problem.indices::<Vessel>().collect::<Vec<usize>>();
        let mut rng = rand::thread_rng();

        let n1 = rng.gen_range(1..=indices.len());
        let n2 = rng.gen_range(1..=indices.len());
        // Defined the three ranges where we will apply: [unchanged, swapped, crossover]
        let (start, end) = (min(n1, n2), max(n1, n2));

        // Shuffle the indices
        indices.shuffle(&mut rng);

        // Note: vessels corresponding to indices[..start] are left as-is.
        // The plans that will be taken from the right parent
        let right_vessels = &indices[start..end];
        // The plans that we will apply one-point crossover to
        let mixed_vessels = &indices[end..];

        let mut left = left.mutate();
        let mut right = right.mutate();

        // Swap over the `right` vessels in their entirety
        for &v in right_vessels {
            std::mem::swap(&mut left[v], &mut right[v]);
        }

        // To a one-point crossover for the `mixed` vessels
        for &v in mixed_vessels {
            let split = rng.gen_range(0..min(left[v].len(), right[v].len()));
            let mut l = left[v].mutate();
            let mut r = right[v].mutate();

            l[..split].swap_with_slice(&mut r[..split]);
            l.fix();
            r.fix();
        }
    }
}
