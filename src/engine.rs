use core::fmt;
use std::{collections::HashSet, ops};
use std::hash::{Hash, Hasher};
use std::rc::Weak;
use std::cell::{RefCell};
use crate::ArenaRef;

// For the convenience of creating Values without having to manually clone the arena
pub struct ValueFactory {
    arena: ArenaRef<ValueData>
}

impl ValueFactory {
    pub fn new(arena: ArenaRef<ValueData>) -> ValueFactory {
        ValueFactory { arena }
    }

    pub fn value(&self, data: f64) -> Value {
        Value::build(self.arena.clone(), data)
    }
}

#[derive(Clone)]
pub struct Value {
    value: Weak<RefCell<ValueData>>,
    arena: ArenaRef<ValueData>
}

impl Value {
    pub fn build(arena: ArenaRef<ValueData>, data: f64) -> Value {
        Value {
            value: arena.alloc_with_mut_borrow(ValueData::new(data, 0.0, Box::new(|| {}), &[], None)),
            arena
        }
    }

    fn new(arena: ArenaRef<ValueData>, data: f64, children: &[Value], op: String) -> Value {
        Value { 
            value: arena.alloc_with_mut_borrow(ValueData::new(data, 0.0, Box::new(|| {}), children, Some(op))),
            arena
        }
    }

    // Always panic if upgrade references a dropped value (autograd graph not DAG)
    fn with_borrow<R>(&self, f: impl FnOnce(&ValueData) -> R) -> R {
        let value_ptr = self.value.upgrade().expect("DAG properties of autograd graph violated");
        let borrow = value_ptr.borrow();
        f(&borrow)
    }

    // Always panic if upgrade references a dropped value (autograd graph not DAG)
    fn with_mut_borrow<R>(&self, f: impl FnOnce(&mut ValueData) -> R) -> R {
        let value_ptr = self.value.upgrade().expect("DAG properties of autograd graph violated");
        let mut borrow = value_ptr.borrow_mut();
        f(&mut borrow)
    }

    pub fn get_data(&self) -> f64 {
        self.with_borrow(|v| v.data)
    }

    pub fn get_grad(&self) -> f64 {
        self.with_borrow(|v| v.grad)
    }

    pub fn set_data(&self, data: f64) {
        self.with_mut_borrow(|v| v.data = data);
    }

    pub fn set_grad(&self, grad: f64) {
        self.with_mut_borrow(|v| v.grad = grad);
    }

    fn add_grad(&self, delta: f64) {
        self.with_mut_borrow(|v| v.grad += delta);
    }

    fn set_backward(&self, backward_fn: impl Fn() + 'static) {
        self.with_mut_borrow(|v| v.backward = Box::new(backward_fn));
    }

    pub fn backward(&self) {
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<Value> = HashSet::new();
        fn build_topo(v: &Value, visited: &mut HashSet<Value>, topo: &mut Vec<Value>) {
            if !visited.contains(v) {
                visited.insert(v.clone());
                v.with_borrow(|node| {
                    node.prev.iter().for_each(|child| {build_topo(child, visited, topo);});
                });
                topo.push(v.clone());
            }
        }
        build_topo(&self, &mut visited, &mut topo);
        
        // go one variable at a time and apply the chain rule to get its gradient
        self.with_mut_borrow(|v| v.grad = 1.0);
        topo.iter().rev().for_each(|node| {
            node.with_borrow(|v| (v.backward)());
        });
    }

    fn trace(&self) -> (HashSet<Value>, HashSet<(Value, Value)>) {
        let mut nodes: HashSet<Value> = HashSet::new();
        let mut edges: HashSet<(Value, Value)> = HashSet::new();
        fn build(v: &Value, nodes: &mut HashSet<Value>, edges: &mut HashSet<(Value, Value)>) {
            if !nodes.contains(v) {
                nodes.insert(v.clone());
                v.with_borrow(|node| {
                    node.prev.iter().for_each(|child| {
                        edges.insert((child.clone(), v.clone()));
                        build(child, nodes, edges);
                    });
                });
            }
        }
        build(&self, &mut nodes, &mut edges);
        (nodes, edges)
    }

    pub fn draw_dot(&self) -> String {
        let mut dot = String::new();

        // Configure the digraph
        dot.push_str("digraph {\n");
        dot.push_str("    rankdir=LR;\n");
        // dot.push_str("    nodesep=0.6;\n");
        // dot.push_str("    ranksep=0.7;\n");
        // ****************************************

        let (nodes, edges) = self.trace();

        let get_node_id = |n: &Value| -> String {
            (n.value.as_ptr() as usize).to_string()
        };
        
        nodes.iter().for_each(|n| {
            let (data, grad, opt_op) = n.with_borrow(
                |value| (value.data, value.grad, value.op.clone())
            );
            let n_id = get_node_id(&n);
            dot.push_str(&Self::add_data_node(&n_id, data, grad));

            if let Some(op) = &opt_op {
                let op_id = format!("\"{}{}\"", n_id, op);
                dot.push_str(&Self::add_op_node(&op_id, op));
                dot.push_str(&Self::add_edge(&op_id, &n_id));
            }
        });

        edges.iter().for_each(|(n1, n2)| {
            let n_id = get_node_id(n1);
            let op = n2.with_borrow(|v| {v.op.clone().unwrap()});
            let op_id = format!("\"{}{}\"", get_node_id(n2), op);
            dot.push_str(&Self::add_edge(&n_id, &op_id));
        });

        // Close the digraph
        dot.push_str("}\n");
        dot
    }

    fn add_data_node(id: &str, data: f64, grad: f64) -> String {
        // a [ shape=record, label = "data 1" ]
        format!("    {} [ shape=record, label = \"{{data {:.4} | grad {:.4}}}\" ]\n", id, data, grad)
    }

    fn add_op_node(id: &str, op: &str) -> String {
        // a [ label = "+" ]
        format!("    {} [ label = \"{}\" ]\n", id, op)
    }

    fn add_edge(id_1: &str, id_2: &str) -> String {
        // a -> b
        format!("    {} -> {}\n", id_1, id_2)
    }

    pub fn relu(&self) -> Value {
        let self_data = self.get_data();
        let out = Value::new(
            self.arena.clone(),
            if self_data < 0.0 {0.0} else {self_data},
            &[self.clone()],
            String::from("ReLU")
        );

        let (out_ref, self_ref) = (out.clone(), self.clone());
        out.set_backward(move || {
            let (out_grad, out_data) = (out_ref.get_grad(), out_ref.get_data());
            self_ref.add_grad(if out_data > 0.0 {out_grad} else {0.0});
        });

        out
    }

    pub fn tanh(&self) -> Value {
        let x = self.get_data();
        let t = ((2.0*x).exp() - 1.0) / ((2.0*x).exp() + 1.0);
        let out = Value::new(
            self.arena.clone(),
            t,
            &[self.clone()],
            String::from("tanh")
        );

        let (out_ref, self_ref) = (out.clone(), self.clone());
        out.set_backward(move || {
            let out_grad = out_ref.get_grad();
            self_ref.add_grad((1.0 - t.powi(2)) * out_grad);
        });

        out
    }

    pub fn exp(&self) -> Value {
        let x = self.get_data();
        let out = Value::new(
            self.arena.clone(),
            x.exp(),
            &[self.clone()],
            String::from("exp")
        );

        let (out_ref, self_ref) = (out.clone(), self.clone());
        out.set_backward(move || {
            let (out_grad, out_data) = (out_ref.get_grad(), out_ref.get_data());
            self_ref.add_grad(out_data * out_grad);
        });

        out
    }

    pub fn powi(&self, other: i32) -> Value {
        let out = Value::new(
            self.arena.clone(),
            self.get_data().powi(other),
            &[self.clone()],
            String::from(format!("powi{}", other))
        );

        let (out_ref, self_ref) = (out.clone(), self.clone());
        out.set_backward(move || {
            let out_grad = out_ref.get_grad();
            self_ref.add_grad(other as f64 * self_ref.get_data().powi(other - 1) * out_grad);
        });

        out
    }

    pub fn powf(&self, other: f64) -> Value {
        let out = Value::new(
            self.arena.clone(),
            self.get_data().powf(other),
            &[self.clone()],
            String::from(format!("powf{}", other))
        );

        let (out_ref, self_ref) = (out.clone(), self.clone());
        out.set_backward(move || {
            let out_grad = out_ref.get_grad();
            self_ref.add_grad(other * self_ref.get_data().powf(other - 1.0) * out_grad);
        });

        out
    }
}

impl fmt::Display for Value {
    // f"Value(data={self.data}, grad={self.grad})"
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.with_borrow(|v| v.fmt(f))
    }
}

impl<'a, 'b> ops::Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, rhs: &'b Value) -> Value {
        let out = Value::new(
            self.arena.clone(),
            self.get_data() + rhs.get_data(),
            &[self.clone(), rhs.clone()],
            String::from("+")
        );
        
        let (out_ref, self_ref, rhs_ref) = (out.clone(), self.clone(), rhs.clone());
        out.set_backward(move || {
            let out_grad = out_ref.get_grad();
            self_ref.add_grad(out_grad);
            rhs_ref.add_grad(out_grad);
        });

        out
    }
}

impl<'a> ops::Add<f64> for &'a Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Value {
        self + &Value::build(self.arena.clone(), rhs)
    }
}

impl<'a> ops::Add<&'a Value> for f64 {
    type Output = Value;

    fn add(self, rhs: &'a Value) -> Value {
        &Value::build(rhs.arena.clone(), self) + rhs
    }
}

impl<'a> ops::Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Value {
        self * &Value::build(self.arena.clone(), -1.0)
    }
}

impl<'a, 'b> ops::Sub<&'b Value> for &'a Value {
    type Output = Value;

    fn sub(self, rhs: &'b Value) -> Value {
        self + &(-rhs)
    }
}

impl<'a> ops::Sub<f64> for &'a Value {
    type Output = Value;

    fn sub(self, rhs: f64) -> Value {
        self - &Value::build(self.arena.clone(), rhs)
    }
}

impl<'a> ops::Sub<&'a Value> for f64 {
    type Output = Value;

    fn sub(self, rhs: &'a Value) -> Value {
        &Value::build(rhs.arena.clone(), self) - rhs
    }
}

impl<'a, 'b> ops::Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, rhs: &'b Value) -> Value {
        let out = Value::new(
            self.arena.clone(),
            self.get_data() * rhs.get_data(),
            &[self.clone(), rhs.clone()],
            String::from("*")
        );

        let (out_ref, self_ref, rhs_ref) = (out.clone(), self.clone(), rhs.clone());
        out.set_backward(move || {
            let out_grad = out_ref.get_grad();
            self_ref.add_grad(rhs_ref.get_data() * out_grad);
            rhs_ref.add_grad(self_ref.get_data() * out_grad);
        });

        out
    }
}

impl<'a> ops::Mul<f64> for &'a Value {
    type Output = Value;

    fn mul(self, rhs: f64) -> Value {
        self * &Value::build(self.arena.clone(), rhs)
    }
}

impl<'a> ops::Mul<&'a Value> for f64 {
    type Output = Value;

    fn mul(self, rhs: &'a Value) -> Value {
        &Value::build(rhs.arena.clone(), self) * rhs
    }
}

impl<'a, 'b> ops::Div<&'b Value> for &'a Value {
    type Output = Value;

    fn div(self, rhs: &'b Value) -> Value {
        self * &rhs.powi(-1)
    }
}

impl<'a> ops::Div<f64> for &'a Value {
    type Output = Value;

    fn div(self, rhs: f64) -> Value {
        self / &Value::build(self.arena.clone(), rhs)
    }
}

impl<'a> ops::Div<&'a Value> for f64 {
    type Output = Value;

    fn div(self, rhs: &'a Value) -> Value {
        &Value::build(rhs.arena.clone(), self) / rhs
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.value.ptr_eq(&other.value)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.as_ptr().hash(state);
    }
}

pub struct ValueData {
    data: f64,
    grad: f64,
    backward: Box<dyn Fn()>,
    prev: HashSet<Value>,
    op: Option<String>,
}

impl ValueData {
    fn new(data: f64, grad: f64, backward: Box<dyn Fn()>, children: &[Value], op: Option<String>) -> ValueData {
        ValueData { data, grad, backward, prev: children.iter().cloned().collect(), op }
    }
}

impl fmt::Display for ValueData {
    // f"Value(data={self.data}, grad={self.grad})"
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data, self.grad)
    }
}

/******************************** unit tests ********************************/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arena;

    #[test]
    fn display() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);

        let v = vf.value(4.2);
        assert_eq!(v.to_string(), "Value(data=4.2, grad=0)");
    }

    #[test]
    fn add() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);

        let a = vf.value(2.5);
        let b = vf.value(3.2);
        let c = &a + &b;
        let d = &(&a + &b) + &c;
        assert_eq!(c.get_data(), 5.7);
        assert_eq!(d.get_data(), 11.4);
        assert_eq!(c.to_string(), "Value(data=5.7, grad=0)");
        assert_eq!(d.to_string(), "Value(data=11.4, grad=0)");
    }

    #[test]
    fn add_grad() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);

        // Value + Value
        let a = vf.value(2.5);
        let b = vf.value(3.2);
        let c = &a + &b;
        assert_eq!(c.get_data(), 5.7);
        assert_eq!(c.to_string(), "Value(data=5.7, grad=0)");

        assert_eq!(a.get_grad(), 0.0);
        assert_eq!(b.get_grad(), 0.0);
        c.backward();
        assert_eq!(a.get_grad(), 1.0);
        assert_eq!(b.get_grad(), 1.0);

        // Value + f64
        let a = vf.value(2.5);
        let b = 3.2;
        let c = &a + b;
        assert_eq!(c.get_data(), 5.7);
        assert_eq!(c.to_string(), "Value(data=5.7, grad=0)");

        assert_eq!(a.get_grad(), 0.0);
        c.backward();
        assert_eq!(a.get_grad(), 1.0);

        // f64 + Value
        let a = 2.5;
        let b = vf.value(3.2);
        let c = a + &b;
        assert_eq!(c.get_data(), 5.7);
        assert_eq!(c.to_string(), "Value(data=5.7, grad=0)");

        assert_eq!(b.get_grad(), 0.0);
        c.backward();
        assert_eq!(b.get_grad(), 1.0);
    }

    #[test]
    fn sub() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);

        // Value - Value
        let a = vf.value(2.5);
        let b = vf.value(5.0);
        let c = &a - &b;
        assert_eq!(c.get_data(), -2.5);
        assert_eq!(c.to_string(), "Value(data=-2.5, grad=0)");

        assert_eq!(a.get_grad(), 0.0);
        assert_eq!(b.get_grad(), 0.0);
        c.backward();
        assert_eq!(a.get_grad(), 1.0);
        assert_eq!(b.get_grad(), -1.0);
        
        // Value - f64
        let a = vf.value(2.5);
        let b = 5.0;
        let c = &a - b;
        assert_eq!(c.get_data(), -2.5);
        assert_eq!(c.to_string(), "Value(data=-2.5, grad=0)");

        assert_eq!(a.get_grad(), 0.0);
        c.backward();
        assert_eq!(a.get_grad(), 1.0);

        // f64 - Value
        let a = 2.5;
        let b = vf.value(5.0);
        let c = a - &b;
        assert_eq!(c.get_data(), -2.5);
        assert_eq!(c.to_string(), "Value(data=-2.5, grad=0)");

        assert_eq!(b.get_grad(), 0.0);
        c.backward();
        assert_eq!(b.get_grad(), -1.0);
    }

    #[test]
    fn mul() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);
        
        // value mul with other values
        let a = vf.value(2.5);
        let b = vf.value(3.0);
        let c = &a * &b;
        let d = &(&a * &b) * &c;
        assert_eq!(c.get_data(), 7.5);
        assert_eq!(d.get_data(), 56.25);
        assert_eq!(c.to_string(), "Value(data=7.5, grad=0)");
        assert_eq!(d.to_string(), "Value(data=56.25, grad=0)");

        // float mul with value
        let e = 2.0 * &d;
        assert_eq!(e.get_data(), 112.5);
        assert_eq!(e.to_string(), "Value(data=112.5, grad=0)");

        // value mul with float
        let f = &e * 2.0;
        assert_eq!(f.get_data(), 225.0);
        assert_eq!(f.to_string(), "Value(data=225, grad=0)");
    }

    #[test]
    fn mul_grad() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);

        // value * value
        let a = vf.value(2.5);
        let b = vf.value(3.0);
        let c = &a * &b;
        assert_eq!(c.get_data(), 7.5);
        assert_eq!(c.to_string(), "Value(data=7.5, grad=0)");

        assert_eq!(a.get_grad(), 0.0);
        assert_eq!(b.get_grad(), 0.0);
        c.backward();
        assert_eq!(a.get_grad(), 3.0);
        assert_eq!(b.get_grad(), 2.5);

        // value * float
        let e = vf.value(2.5);
        let f = &e * 3.0;
        assert_eq!(f.get_data(), 7.5);
        assert_eq!(f.to_string(), "Value(data=7.5, grad=0)");

        assert_eq!(e.get_grad(), 0.0);
        f.backward();
        assert_eq!(e.get_grad(), 3.0);

        // float * value
        let g = vf.value(3.0);
        let h = 2.5 * &g;
        assert_eq!(h.get_data(), 7.5);
        assert_eq!(h.to_string(), "Value(data=7.5, grad=0)");

        assert_eq!(g.get_grad(), 0.0);
        h.backward();
        assert_eq!(h.to_string(), "Value(data=7.5, grad=1)");
        assert_eq!(g.get_grad(), 2.5);
        assert_eq!(g.to_string(), "Value(data=3, grad=2.5)");
    }

    #[test]
    fn pow() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);
        
        // pow with int
        let a = vf.value(2.0);
        let b = a.powi(2);
        assert_eq!(b.get_data(), 4.0);
        assert_eq!(b.to_string(), "Value(data=4, grad=0)");

        // test grad as well
        b.backward();
        assert_eq!(b.get_grad(), 1.0);
        assert_eq!(b.to_string(), "Value(data=4, grad=1)");
        assert_eq!(a.get_grad(), 4.0);
        assert_eq!(a.to_string(), "Value(data=2, grad=4)");

        // pow with float works the same as pow with int
        let a = vf.value(2.0);
        let b = a.powf(2.0);
        assert_eq!(b.get_data(), 4.0);
        assert_eq!(b.to_string(), "Value(data=4, grad=0)");

        // test grad as well
        b.backward();
        assert_eq!(b.get_grad(), 1.0);
        assert_eq!(b.to_string(), "Value(data=4, grad=1)");
        assert_eq!(a.get_grad(), 4.0);
        assert_eq!(a.to_string(), "Value(data=2, grad=4)");
    }

    #[test]
    fn neg() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);
        
        let a = vf.value(2.0);
        let b = -&a;
        assert_eq!(b.get_data(), -2.0);
        assert_eq!(b.to_string(), "Value(data=-2, grad=0)");

        // test grad as well
        b.backward();
        assert_eq!(b.get_grad(), 1.0);
        assert_eq!(b.to_string(), "Value(data=-2, grad=1)");
        assert_eq!(a.get_grad(), -1.0);
        assert_eq!(a.to_string(), "Value(data=2, grad=-1)");
    }

    #[test]
    fn div() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);

        // value / value
        let a = vf.value(2.5);
        let b = vf.value(1.0/3.0);
        let c = &a / &b;
        assert_eq!(c.get_data(), 7.5);
        assert_eq!(c.to_string(), "Value(data=7.5, grad=0)");

        assert_eq!(a.get_grad(), 0.0);
        assert_eq!(b.get_grad(), 0.0);
        c.backward();
        assert_eq!(a.get_grad(), 3.0);
        assert_eq!(b.get_grad(), -22.5);

        // value / float
        let e = vf.value(2.5);
        let f = &e / (1.0/3.0);
        assert_eq!(f.get_data(), 7.5);
        assert_eq!(f.to_string(), "Value(data=7.5, grad=0)");

        assert_eq!(e.get_grad(), 0.0);
        f.backward();
        assert_eq!(e.get_grad(), 3.0);

        // float / value
        let g = vf.value(1.0/3.0);
        let h = 2.5 / &g;
        assert_eq!(h.get_data(), 7.5);
        assert_eq!(h.to_string(), "Value(data=7.5, grad=0)");

        assert_eq!(g.get_grad(), 0.0);
        h.backward();
        assert_eq!(h.to_string(), "Value(data=7.5, grad=1)");
        assert_eq!(g.get_grad(), -22.5);
        assert_eq!(g.to_string(), "Value(data=0.3333333333333333, grad=-22.5)");
    }

    #[test]
    fn relu() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);
        
        let a = vf.value(2.0);
        let b = a.relu();
        assert_eq!(b.get_data(), 2.0);
        assert_eq!(b.to_string(), "Value(data=2, grad=0)");

        // test grad as well
        b.backward();
        assert_eq!(b.get_grad(), 1.0);
        assert_eq!(b.to_string(), "Value(data=2, grad=1)");
        assert_eq!(a.get_grad(), 1.0);
        assert_eq!(a.to_string(), "Value(data=2, grad=1)");

        // less than 0 case
        let c = vf.value(-2.0);
        let d = c.relu();
        assert_eq!(d.get_data(), 0.0);
        assert_eq!(d.to_string(), "Value(data=0, grad=0)");

        // test grad as well
        d.backward();
        assert_eq!(d.get_grad(), 1.0);
        assert_eq!(d.to_string(), "Value(data=0, grad=1)");
        assert_eq!(c.get_grad(), 0.0);
        assert_eq!(c.to_string(), "Value(data=-2, grad=0)");
    }

    #[test]
    fn tanh() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);

        let a = vf.value(2.0);
        let b = a.tanh();

        let expected_data = 2.0_f64.tanh();
        assert!((b.get_data() - expected_data).abs() < 1e-12);
        assert_eq!(b.to_string(), format!("Value(data={}, grad=0)", b.get_data()));

        // test grad as well
        b.backward();

        let expected_grad = 1.0 - expected_data * expected_data;
        assert!((a.get_grad() - expected_grad).abs() < 1e-12);

        assert_eq!(b.get_grad(), 1.0);
    }

    #[test]
    fn exp() {
        let (_arena_life_time, arena_ref) = Arena::build();
        let vf = ValueFactory::new(arena_ref);

        let a = vf.value(2.0);
        let b = a.exp();

        let expected_data = 2.0_f64.exp();
        assert!((b.get_data() - expected_data).abs() < 1e-12);
        assert_eq!(b.to_string(), format!("Value(data={}, grad=0)", b.get_data()));

        // test grad as well
        b.backward();

        let expected_grad = expected_data;
        assert!((a.get_grad() - expected_grad).abs() < 1e-12);

        assert_eq!(b.get_grad(), 1.0);
    }
}

