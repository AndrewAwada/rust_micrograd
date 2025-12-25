use rust_micrograd::Arena;
use rust_micrograd::ValueFactory;
use std::process::Command;
use std::fs;

const DOT_FILE: &str = "graph.dot";
const DOT_PNG: &str = "graph.png";

fn main() {
    let (_arena_life_time, arena_ref) = Arena::build();
    let vf = ValueFactory::new(arena_ref);

    // inputs x1,x2
    let x1 = vf.value(2.0);
    let x2 = vf.value(0.0);
    // weights w1,w2
    let w1 = vf.value(-3.0);
    let w2 = vf.value(1.0);
    // bias of the neuron
    let b = vf.value(6.8813735870195432);
    // x1*w1 + x2*w2 + b
    let x1w1 = &x1*&w1;
    let x2w2 = &x2*&w2;
    let x1w1x2w2 = &x1w1 + &x2w2;
    let n = &x1w1x2w2 + &b;
    let o = n.tanh();

    o.backward();
    let dot = o.draw_dot();

    write_dot(&dot, DOT_FILE).unwrap();
    render_dot(DOT_FILE, DOT_PNG).unwrap();
}

fn write_dot(dot: &str, out_path: &str) -> std::io::Result<()> {
    fs::write(out_path, dot)
}

fn render_dot(dot_path: &str, png_path: &str) -> std::io::Result<()> {
    let status = Command::new("dot")
        .args(["-Tpng", dot_path, "-o", png_path])
        .status()?;

    if !status.success() {
        panic!("dot command failed");
    }

    Ok(())
}




