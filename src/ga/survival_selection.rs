use float_ord::FloatOrd;
use pyo3::pyclass;
use rand::prelude::*;
use std::cell::Cell;

use crate::solution::routing::RoutingSolution;
use crate::utils;

use super::traits::SurvivalSelection;
use super::{parent_selection, ParentSelection};

#[pyclass]
pub struct Greedy;

impl SurvivalSelection for Greedy {
    fn select_survivors<F>(
        &mut self,
        objective_fn: F,
        population: &[RoutingSolution],
        _parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut [RoutingSolution],
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        let mut combined = population.iter().chain(children).collect::<Vec<_>>();

        combined.sort_by_cached_key(|&x| FloatOrd(objective_fn(x)));

        for (out, source) in out.iter_mut().zip(&combined) {
            out.clone_from(source);
        }
    }
}

#[derive(Clone)]
pub struct Proportionate<F>(pub F)
where
    F: Fn(f64) -> f64;

impl<G> SurvivalSelection for Proportionate<G>
where
    G: Fn(f64) -> f64,
{
    fn select_survivors<F>(
        &mut self,
        objective_fn: F,
        population: &[RoutingSolution],
        _parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut [RoutingSolution],
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

        for out in out.iter_mut() {
            let i = proportionate.sample();
            let source = population
                .get(i)
                .unwrap_or_else(|| &children[i - population.len()]);

            out.clone_from(source);
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
        objective_fn: F,
        population: &[RoutingSolution],
        parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut [RoutingSolution],
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        let elite_count = self.0;
        let mut elites = Vec::new();
        // We assume `k` to be small, such that the k in O(kn) is negligible
        for i in 0..elite_count {
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
            out[i].clone_from(elite.1);
        }

        self.1.select_survivors(
            objective_fn,
            population,
            parents,
            children,
            &mut out[elite_count..],
        );
    }
}

#[derive(Clone)]
pub struct Elite<S>(pub usize, pub S);

impl<S> SurvivalSelection for Elite<S>
where
    S: SurvivalSelection,
{
    fn select_survivors<F>(
        &mut self,
        objective_fn: F,
        population: &[RoutingSolution],
        parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut [RoutingSolution],
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        let elite_count = self.0;
        let mut elites = Vec::new();
        // We assume `k` to be small, such that the k in O(kn) is negligible
        for i in 0..elite_count {
            let elite = population
                .iter()
                .chain(children)
                .enumerate()
                .filter(|(i, _)| !elites.contains(i))
                .min_by_key(|(_, x)| FloatOrd(objective_fn(x)))
                .unwrap();

            elites.push(elite.0);
            out[i].clone_from(elite.1);
        }

        self.1.select_survivors(
            objective_fn,
            population,
            parents,
            children,
            &mut out[elite_count..],
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
        _: F,
        _: &[RoutingSolution],
        __: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut [RoutingSolution],
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        let rng = self.rng.get_mut();
        let chosen = children.choose_multiple(rng, out.len());
        for (out, source) in out.iter_mut().zip(chosen) {
            out.clone_from(source);
        }
    }
}
