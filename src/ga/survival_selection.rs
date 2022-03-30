use float_ord::FloatOrd;
use rand::prelude::*;
use std::cell::Cell;

use crate::solution::routing::RoutingSolution;
use crate::utils;

use super::traits::SurvivalSelection;
use super::{parent_selection, ParentSelection};

pub struct Greedy;

impl SurvivalSelection for Greedy {
    fn select_survivors<F>(
        &mut self,
        count: usize,
        objective_fn: F,
        population: &[RoutingSolution],
        _parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut Vec<RoutingSolution>,
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        let mut combined = Vec::new();

        for p in population {
            combined.push(p);
        }

        for c in children {
            combined.push(c);
        }

        combined.sort_by_cached_key(|&x| FloatOrd(objective_fn(x)));

        out.extend(combined.into_iter().cloned().take(count));
    }
}

pub struct Proportionate<F>(pub F)
where
    F: Fn(f64) -> f64;

impl<G> SurvivalSelection for Proportionate<G>
where
    G: Fn(f64) -> f64,
{
    fn select_survivors<F>(
        &mut self,
        count: usize,
        objective_fn: F,
        population: &[RoutingSolution],
        _parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut Vec<RoutingSolution>,
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        let mut proportionate = parent_selection::Proportionate::with_fn(&self.0);
        proportionate.init(
            population
                .iter()
                .chain(children)
                .map(|x| objective_fn(x))
                .collect(),
        );

        while out.len() < count {
            let i = proportionate.sample();

            out.push(if i >= population.len() {
                children[i - population.len()].clone()
            } else {
                population[i].clone()
            });
        }
    }
}

pub struct FeasibleElite<S>(pub usize, pub S);

impl<S> SurvivalSelection for FeasibleElite<S>
where
    S: SurvivalSelection,
{
    fn select_survivors<F>(
        &mut self,
        count: usize,
        objective_fn: F,
        population: &[RoutingSolution],
        parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut Vec<RoutingSolution>,
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        let elite_count = self.0;
        let mut elites = Vec::new();
        // We assume `k` to be small, such that the k in O(kn) is negligible
        for _ in 0..elite_count {
            let elite = population
                .iter()
                .chain(children)
                .enumerate()
                .filter(|(i, _)| !elites.contains(i))
                .min_by_key(|(_, x)| {
                    let cost = objective_fn(x);
                    let feasible = x.warp() == 0 && x.violation() <= utils::EPSILON;
                    (!feasible, FloatOrd(cost))
                })
                .unwrap();

            elites.push(elite.0);
            out.push(elite.1.clone());
        }

        self.1.select_survivors(
            count - elite_count,
            objective_fn,
            population,
            parents,
            children,
            out,
        );
    }
}

pub struct Elite<S>(pub usize, pub S);

impl<S> SurvivalSelection for Elite<S>
where
    S: SurvivalSelection,
{
    fn select_survivors<F>(
        &mut self,
        count: usize,
        objective_fn: F,
        population: &[RoutingSolution],
        parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut Vec<RoutingSolution>,
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        let elite_count = self.0;
        let mut elites = Vec::new();
        // We assume `k` to be small, such that the k in O(kn) is negligible
        for _ in 0..elite_count {
            let elite = population
                .iter()
                .chain(children)
                .enumerate()
                .filter(|(i, _)| !elites.contains(i))
                .min_by_key(|(_, x)| FloatOrd(objective_fn(x)))
                .unwrap();

            elites.push(elite.0);
            out.push(elite.1.clone());
        }

        self.1.select_survivors(
            count - elite_count,
            objective_fn,
            population,
            parents,
            children,
            out,
        );
    }
}

pub struct Generational {
    rng: Cell<StdRng>,
}

impl Generational {
    pub fn new() -> Self {
        Generational {
            rng: Cell::new(StdRng::from_entropy()),
        }
    }
}

impl SurvivalSelection for Generational {
    fn select_survivors<F>(
        &mut self,
        count: usize,
        _: F,
        _: &[RoutingSolution],
        __: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut Vec<RoutingSolution>,
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        let rng = self.rng.get_mut();
        out.extend(children.choose_multiple(rng, count).cloned())
    }
}
