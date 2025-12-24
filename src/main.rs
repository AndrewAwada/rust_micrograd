use rust_micrograd::Value;
use std::process::Command;
use std::fs;

const DOT_FILE: &str = "graph.dot";
const DOT_PNG: &str = "graph.png";

fn main() {
    println!("Hello, world!");
    // inputs x1,x2
    let x1 = Value::build(2.0);
    let x2 = Value::build(0.0);
    // weights w1,w2
    let w1 = Value::build(-3.0);
    let w2 = Value::build(1.0);
    // bias of the neuron
    let b = Value::build(6.8813735870195432);
    // x1*w1 + x2*w2 + b
    let x1w1 = &x1*&w1;
    let x2w2 = &x2*&w2;
    let x1w1x2w2 = &x1w1 + &x2w2;
    let n = &x1w1x2w2 + &b;
    let o = n.tanh();
    
    o.backward();
    let dot = o.draw_dot();

    // Should panick but doesn't because shadowed variables don't have to drop till end of scope
    // let a = Value::build(2.0);
    // let b = Value::build(-3.0);
    // let c = &a + &b;
    // let a = &c + &b;
    // let dot = a.draw_dot();

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




