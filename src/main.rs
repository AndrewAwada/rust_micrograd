use rust_micrograd::{Arena, Module};
use rust_micrograd::{Value, ValueFactory, MLP};
use std::process::Command;
use std::fs;

const DOT_FILE: &str = "graph.dot";
const DOT_PNG: &str = "graph.png";

fn main() {
    let (_arena_life_time, arena_ref) = Arena::build();
    let vf = ValueFactory::new(arena_ref);

    example_usage(&vf);

    example_expression_with_dot(&vf);
    
    example_training_loop(&vf);
}

fn example_usage(vf: &ValueFactory) {
    let a = vf.value(-4.0);
    let b = vf.value(2.0);
    let mut c = &a + &b;
    let mut d = &(&a * &b) + &b.powi(3);
    c = &(&c + &c) + 1.0;
    c = &(&(&c + 1.0) + &c) + &-&a;
    d = &(&d + &(&d * 2.0)) + &(&b + &a).relu();
    d = &d + &(&(3.0 * &d) + &(&b - &a).relu());
    let e = &c - &d;
    let f = e.powi(2);
    let mut g = &f / 2.0;
    g = &g + &(10.0 / &f);
    println!("{:.4}", g.get_data()); // prints 24.7041, the outcome of this forward pass
    g.backward();
    println!("{:.4}", a.get_grad()); // prints 138.8338, i.e. the numerical value of dg/da
    println!("{:.4}", b.get_grad()); // prints 645.5773, i.e. the numerical value of dg/db
}

fn example_training_loop(vf: &ValueFactory) {
    let n = MLP::new(vf, 3, &vec![4 as usize, 4 as usize, 1 as usize]);

    let xs = vec![
        vec![vf.value(2.0), vf.value(3.0), vf.value(-1.0)],
        vec![vf.value(3.0), vf.value(-1.0), vf.value(0.5)],
        vec![vf.value(0.5), vf.value(1.0), vf.value(1.0)],
        vec![vf.value(1.0), vf.value(1.0), vf.value(-1.0)]
    ];
    let ys = vec![vf.value(1.0), vf.value(-1.0), vf.value(-1.0), vf.value(1.0)];

    let mse_loss = |ys: &Vec<Value>, ypred: &Vec<Value>| {
        ys.iter()
            .zip(ypred.iter())
            .fold(vf.value(0.0), |acc, (ygt, yout)| {
                &acc + &(ygt - yout).powi(2)
            })
    };

    let forward = |xs: &Vec<Vec<Value>>| -> Vec<Value> {
        xs.iter()
            .map(|x| {
                n.call(x).get(0).unwrap().clone()
            })
            .collect()
    };

    let epochs = 500;
    let lr = -0.1;
    println!("Beginning Training Loop");
    for i in 0..epochs {
        let ypred: Vec<Value> = forward(&xs);
        let loss = mse_loss(&ys, &ypred);

        n.zero_grad();
        loss.backward();

        n.parameters().for_each(|p| p.set_data(p.get_data() + lr * p.get_grad()));

        if i % 10 == 0 {
            println!("Loss at step {}: {}", i, loss.get_data());
        }
    }
    let ypred_final = forward(&xs);
    let final_loss = mse_loss(&ys, &ypred_final);
    println!("Final Predictions With Loss {} After {} Epochs: [{}, {}, {}, {}]", final_loss.get_data(), epochs,
        ypred_final[0].get_data(), ypred_final[1].get_data(), ypred_final[2].get_data(), ypred_final[3].get_data());
}

fn example_expression_with_dot(vf: &ValueFactory) {
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




