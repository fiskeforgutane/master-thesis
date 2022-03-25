use super::traits::ParentSelection;
use float_ord::FloatOrd;
use rand::{distributions::Uniform, prelude::*};
use std::unreachable;

/// Choose an individual for mating proportional to `f(fitness)` for the `f` provided.
pub struct Proportionate<F>
where
    F: Fn(f64) -> f64,
{
    f: F,
    distribution: Uniform<f64>,
    rng: StdRng,
    weights: Vec<f64>,
}

impl<F> Proportionate<F>
where
    F: Fn(f64) -> f64,
{
    pub fn with_fn(f: F) -> Proportionate<F> {
        Proportionate {
            f,
            distribution: Uniform::new_inclusive(0.0, 1.0),
            rng: StdRng::from_entropy(),
            weights: Vec::new(),
        }
    }
}

impl<F> ParentSelection for Proportionate<F>
where
    F: Fn(f64) -> f64,
{
    fn init(&mut self, fitness_values: Vec<f64>) {
        self.weights.clear();
        self.weights.extend(fitness_values.into_iter().map(&self.f));

        let total: f64 = self.weights.iter().sum();

        self.distribution = Uniform::new_inclusive(0.0, total);
    }

    fn sample(&mut self) -> usize {
        let threshold = self.distribution.sample(&mut self.rng);
        let mut sum = 0.0;

        for (i, w) in self.weights.iter().enumerate() {
            sum += *w;
            if sum >= threshold {
                return i;
            }
        }

        unreachable!()
    }
}

/// Tournament-style parent selection. Select `k` random individuals from the population, and choose the best among them for reproduction.
pub struct Tournament {
    k: usize,
    buffer: Vec<f64>,
    rng: StdRng,
}

impl Tournament {
    pub fn new(k: usize) -> Option<Self> {
        if k > 0 {
            Tournament {
                k,
                buffer: Vec::new(),
                rng: StdRng::from_entropy(),
            }
            .into()
        } else {
            None
        }
    }
}

impl ParentSelection for Tournament {
    fn init(&mut self, fitness_values: Vec<f64>) {
        self.buffer.clear();
        self.buffer.extend(fitness_values);
    }

    fn sample(&mut self) -> usize {
        let drawn = rand::seq::index::sample(&mut self.rng, self.buffer.len(), self.k);

        drawn
            .iter()
            .min_by_key(|&x| FloatOrd(self.buffer[x]))
            .expect("nonempty by construction")
    }
}
