use crate::solution::routing::RoutingSolution;

#[derive(Clone, Debug)]
pub enum Termination {
    /// Terminate after a given number of epochs
    Epochs(u64),
    /// Terminate upon finding a solution with no violation
    NoViolation,
    /// Terminate upon finding a solution with that adheres to travel at capacity / travel empty
    TravelFullEmptyValid,
    /// Terminate if there has been no improvement for the given amount of time
    NoImprovement(std::time::Instant, std::time::Duration, f64),
    /// Maximum running time from `Instant`
    Timeout(std::time::Instant, std::time::Duration),
    /// Run forever
    Never,
    /// Terminate if either of the two termination criteria
    /// tells it to terminate
    Any(Box<Termination>, Box<Termination>),
    /// Terminate when both of the criteria tells it to terminate
    All(Box<Termination>, Box<Termination>),
}

impl Termination {
    pub fn should_terminate(
        &mut self,
        epoch: u64,
        solution: &RoutingSolution,
        fitness: f64,
    ) -> bool {
        match self {
            Termination::Timeout(from, duration) => (std::time::Instant::now() - *from) > *duration,
            Termination::Epochs(e) => *e > epoch,
            Termination::NoViolation => solution.warp() == 0 && solution.violation() < 1e-3,
            Termination::TravelFullEmptyValid => {
                solution.travel_at_cap() < 1e-3 && solution.travel_empty() < 1e-3
            }
            Termination::Never => false,
            Termination::Any(one, two) => {
                one.should_terminate(epoch, solution, fitness)
                    || two.should_terminate(epoch, solution, fitness)
            }
            Termination::All(one, two) => {
                one.should_terminate(epoch, solution, fitness)
                    && two.should_terminate(epoch, solution, fitness)
            }
            Termination::NoImprovement(last, duration, best) => {
                // Replace best fitness and reset time of last improvement if the current solution is better than the incumbent.
                if fitness < *best {
                    *best = fitness;
                    *last = std::time::Instant::now();
                }

                (std::time::Instant::now() - *last) > *duration
            }
        }
    }
}

impl std::fmt::Display for Termination {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Termination::Epochs(e) => write!(f, "{e} epochs"),
            Termination::NoViolation => write!(f, "no-violation"),
            Termination::TravelFullEmptyValid => write!(f, "full-empty-valid"),
            Termination::NoImprovement(_, dur, _) => write!(f, "{} no-improvement", dur.as_secs()),
            Termination::Timeout(_, dur) => write!(f, "{} timeout", dur.as_secs()),
            Termination::Never => write!(f, "never"),
            Termination::Any(lhs, rhs) => write!(f, "({lhs}) | ({rhs})"),
            Termination::All(lhs, rhs) => write!(f, "({lhs}) & ({rhs})"),
        }
    }
}
