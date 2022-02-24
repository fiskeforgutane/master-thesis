pub mod problem;
pub mod sisrs;
use problem::*;
fn main() {
    let test = Inventory::new(&[1.0]).unwrap();
    println!("{:?}", test);
    println!("Hello, world!");
}
