use rand::prelude::*;

pub fn choose_proportional_by_key<'a, I, T, F, R>(it: I, f: F, mut rng: R) -> T
where
    I: IntoIterator<Item = T> + 'a,
    for<'b> &'b I: IntoIterator<Item = &'b T>,
    T: 'a,
    F: for<'c> Fn(&'c T) -> f64,
    R: Rng,
{
    let total: f64 = (&it).into_iter().map(&f).sum();
    let threshold = rng.gen_range(0.0..=total);

    let mut sum = 0.0;
    for x in it.into_iter() {
        sum += f(&x);
        if sum >= threshold {
            return x;
        }
    }

    unreachable!()
}
