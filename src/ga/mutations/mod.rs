pub mod add_random;
pub mod add_smart;
pub mod best_move;
pub mod bounce;
pub mod dedup_policy;
pub mod distance_reduction;
pub mod inter_swap;
pub mod intra_swap;
pub mod red_cost;
pub mod remove_random;
pub mod replace_node;
pub mod rr;
pub mod swap_star;
pub mod time_setter;
pub mod twerk;
pub mod two_opt;
pub mod vessel_swap;

pub use add_random::*;
pub use add_smart::*;
pub use best_move::*;
pub use bounce::*;
pub use dedup_policy::*;
pub use distance_reduction::*;
pub use inter_swap::*;
pub use intra_swap::*;
pub use red_cost::*;
pub use remove_random::*;
pub use replace_node::*;
pub use swap_star::*;
pub use time_setter::*;
pub use twerk::*;
pub use two_opt::*;
pub use vessel_swap::*;

use crate::solution::routing::Improvement;

use super::{Chain, Nop, Stochastic};

impl std::fmt::Display for AddRandom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("add-random")
    }
}

impl std::fmt::Display for RemoveRandom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("remove-random")
    }
}

impl std::fmt::Display for InterSwap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("interswap")
    }
}

impl std::fmt::Display for IntraSwap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("intraswap")
    }
}

impl std::fmt::Display for RedCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_visits = self.max_visits;
        write!(f, "{max_visits} redcost")
    }
}

impl std::fmt::Display for Twerk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "twerk")
    }
}

impl std::fmt::Display for TwoOpt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "2opt")
    }
}

impl std::fmt::Display for TimeSetter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let delay = self.delay();
        write!(f, "{delay} time-setter")
    }
}

impl std::fmt::Display for ReplaceNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let blink_rate = self.blink_rate;
        write!(f, "{blink_rate} replace-node")
    }
}

impl std::fmt::Display for SwapStar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "swap*")
    }
}

impl std::fmt::Display for Dedup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dedup")
    }
}

impl std::fmt::Display for Bounce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let passes = self.passes;
        write!(f, "{passes} bounce")
    }
}

impl std::fmt::Display for AddSmart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "add-smart")
    }
}

impl std::fmt::Display for rr::Period {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.c;
        let max_size = self.max_size;
        let removal_rate = self.removal_rate;
        let blink_rate = self.blink_rate;

        write!(f, "{c} {max_size} {removal_rate} {blink_rate} rr-period")
    }
}

impl std::fmt::Display for rr::Vessel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.c;
        let removal_rate = self.removal_rate;
        let blink_rate = self.blink_rate;

        write!(f, "{c} {removal_rate} {blink_rate} rr-vessel")
    }
}

impl std::fmt::Display for rr::sisr::SlackInductionByStringRemoval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let config = &self.config;
        let epsilon = config.epsilon;
        let first_n = config.first_n;
        let blink_rate = config.blink_rate;
        let alpha = config.alpha;
        let max_cardinality = config.max_cardinality;
        let average_removal = config.average_removal;

        write!(
            f,
            "{epsilon} {first_n} {blink_rate} {alpha} {max_cardinality} {average_removal} sisr"
        )
    }
}

impl<M> std::fmt::Display for Stochastic<M>
where
    M: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = self.0;
        let m = &self.1;

        write!(f, "{m} {p} ?")
    }
}

impl<A, B> std::fmt::Display for Chain<A, B>
where
    A: std::fmt::Display,
    B: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let a = &self.0;
        let b = &self.1;

        write!(f, "{a} {b}")
    }
}

impl std::fmt::Display for Nop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "nop")
    }
}
