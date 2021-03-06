use crate::solution::Visit;

pub trait GetPairMut {
    type Out;
    fn get_pair_mut(self, v1: usize, v2: usize) -> (Self::Out, Self::Out);
}

impl<'a, T> GetPairMut for &'a mut [T] {
    type Out = &'a mut T;

    fn get_pair_mut(self, v1: usize, v2: usize) -> (Self::Out, Self::Out) {
        assert!(v1 != v2);
        let min = v1.min(v2);
        let max = v2.max(v1);

        let (one, rest) = self[min..].split_first_mut().unwrap();
        let two = &mut rest[max - min - 1];

        if v1 < v2 {
            (one, two)
        } else {
            (two, one)
        }
    }
}

pub trait SwapNodes {
    fn swap_nodes(self, visit: &mut Visit);
}

impl SwapNodes for &mut Visit {
    fn swap_nodes(self, visit: &mut Visit) {
        let temp = self.node;

        // perform the swap
        self.node = visit.node;
        visit.node = temp;
    }
}

pub const EPSILON: f64 = 1e-5;
