use core::fmt;
use std::{collections::HashSet, ops};
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};
use std::cell::{RefCell};

#[derive(Clone)]
pub struct Value {
    value: Rc<RefCell<ValueData>>
}

impl Value {
    pub fn build(data: f64) -> Value {
        Value { value: Rc::new(RefCell::new(ValueData::new(data, 0.0, Box::new(|| {}), &[], None))) }
    }

    fn new(data: f64, children: &[ValueRef], op: String) -> Value {
        Value { value: Rc::new(RefCell::new(ValueData::new(data, 0.0, Box::new(|| {}), children, Some(op)))) }
    }

    pub fn get_data(&self) -> f64 {
        self.value.borrow().data
    }

    pub fn get_grad(&self) -> f64 {
        self.value.borrow().grad
    }

    pub fn backward(&self) {
        let mut topo: Vec<ValueRef> = Vec::new();
        let mut visited: HashSet<ValueRef> = HashSet::new();
        let self_ref = ValueRef::new(&self.value);
        fn build_topo(v: &ValueRef, visited: &mut HashSet<ValueRef>, topo: &mut Vec<ValueRef>) {
            if !visited.contains(v) {
                visited.insert(v.clone());
                v.with_borrow(|node| {
                    node.prev.iter().for_each(|child| {build_topo(child, visited, topo);});
                });
                topo.push(v.clone());
            }
        }
        build_topo(&self_ref, &mut visited, &mut topo);
        
        // go one variable at a time and apply the chain rule to get its gradient
        self_ref.with_mut_borrow(|v| v.grad = 1.0);
        topo.iter().rev().for_each(|node| {
            node.with_borrow(|v| (v.backward)());
        });
    }

    fn trace(&self) -> (HashSet<ValueRef>, HashSet<(ValueRef, ValueRef)>) {
        let mut nodes: HashSet<ValueRef> = HashSet::new();
        let mut edges: HashSet<(ValueRef, ValueRef)> = HashSet::new();
        fn build(v: &ValueRef, nodes: &mut HashSet<ValueRef>, edges: &mut HashSet<(ValueRef, ValueRef)>) {
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
        build(&ValueRef::new(&self.value), &mut nodes, &mut edges);
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

        let get_node_id = |n: &ValueRef| -> String {
            (n.0.as_ptr() as usize).to_string()
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

    pub fn tanh(&self) -> Value {
        let self_ref = ValueRef::new(&self.value);
        let x = self.get_data();
        let t = ((2.0*x).exp() - 1.0) / ((2.0*x).exp() + 1.0);
        let out = Value::new(
            t,
            &[self_ref.clone()],
            String::from("tanh")
        );
        let out_ref = ValueRef::new(&out.value);
        out.value.borrow_mut().backward = Box::new(move || {
            let out_grad = out_ref.with_borrow(|v| {v.grad});
            self_ref.with_mut_borrow(|v| {v.grad += (1.0 - t.powi(2)) * out_grad});
        });

        out
    }

    pub fn exp(&self) -> Value {
        let self_ref = ValueRef::new(&self.value);
        let x = self.get_data();
        let out = Value::new(
            x.exp(),
            &[self_ref.clone()],
            String::from("exp")
        );
        let out_ref = ValueRef::new(&out.value);
        out.value.borrow_mut().backward = Box::new(move || {
            let (out_grad, out_data) = out_ref.with_borrow(|v| {(v.grad, v.data)});
            self_ref.with_mut_borrow(|v| {v.grad += out_data * out_grad});
        });

        out
    }

    pub fn powi(&self, other: i32) -> Value {
        let self_ref = ValueRef::new(&self.value);
        let out = Value::new(
            self.get_data().powi(other),
            &[self_ref.clone()],
            String::from(format!("powi{}", other))
        );
        let out_ref = ValueRef::new(&out.value);
        out.value.borrow_mut().backward = Box::new(move || {
            let out_grad = out_ref.with_borrow(|v| {v.grad});
            let self_data = self_ref.with_borrow(|v| {v.data});
            self_ref.with_mut_borrow(|v| {v.grad += other as f64 * self_data.powi(other - 1) * out_grad});
        });

        out
    }

    pub fn powf(&self, other: f64) -> Value {
        let self_ref = ValueRef::new(&self.value);
        let out = Value::new(
            self.get_data().powf(other),
            &[self_ref.clone()],
            String::from(format!("powf{}", other))
        );
        let out_ref = ValueRef::new(&out.value);
        out.value.borrow_mut().backward = Box::new(move || {
            let out_grad = out_ref.with_borrow(|v| {v.grad});
            let self_data = self_ref.with_borrow(|v| {v.data});
            self_ref.with_mut_borrow(|v| {v.grad += other * self_data.powf(other - 1.0) * out_grad});
        });

        out
    }
}

impl fmt::Display for Value {
    // f"Value(data={self.data}, grad={self.grad})"
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.borrow().fmt(f)
    }
}

impl<'a, 'b> ops::Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, rhs: &'b Value) -> Value {
        let self_ref = ValueRef::new(&self.value);
        let rhs_ref = ValueRef::new(&rhs.value);
        let out = Value::new(
            self.get_data() + rhs.get_data(),
            &[self_ref.clone(), rhs_ref.clone()],
            String::from("+")
        );
        let out_ref = ValueRef::new(&out.value);
        out.value.borrow_mut().backward = Box::new(move || {
            let out_grad = out_ref.with_borrow(|v| {v.grad});
            self_ref.with_mut_borrow(|v| {v.grad += out_grad});
            rhs_ref.with_mut_borrow(|v| {v.grad += out_grad});
        });

        out
    }
}

// TODO: value immediately dropped
impl<'a> ops::Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Value {
        self * &Value::build(-1.0)
    }
}

impl<'a, 'b> ops::Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, rhs: &'b Value) -> Value {
        let self_ref = ValueRef::new(&self.value);
        let rhs_ref = ValueRef::new(&rhs.value);
        let out = Value::new(
            self.get_data() * rhs.get_data(),
            &[self_ref.clone(), rhs_ref.clone()],
            String::from("*")
        );
        let out_ref = ValueRef::new(&out.value);
        out.value.borrow_mut().backward = Box::new(move || {
            let out_grad = out_ref.with_borrow(|v| {v.grad});
            self_ref.with_mut_borrow(|v| {v.grad += rhs_ref.with_borrow(|v| {v.data}) * out_grad});
            rhs_ref.with_mut_borrow(|v| {v.grad += self_ref.with_borrow(|v| {v.data}) * out_grad});
        });

        out
    }
}

impl<'a, 'b> ops::Div<&'b Value> for &'a Value {
    type Output = Value;

    fn div(self, rhs: &'b Value) -> Value {
        self * &rhs.powi(-1)
    }
}

//#[derive(Debug)]
struct ValueData {
    data: f64,
    grad: f64,
    backward: Box<dyn Fn()>,
    prev: HashSet<ValueRef>,
    op: Option<String>,
}

impl ValueData {
    fn new(data: f64, grad: f64, backward: Box<dyn Fn()>, children: &[ValueRef], op: Option<String>) -> ValueData {
        ValueData { data, grad, backward, prev: children.iter().cloned().collect(), op }
    }
}

impl fmt::Display for ValueData {
    // f"Value(data={self.data}, grad={self.grad})"
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}

#[derive(Debug, Clone)]
struct ValueRef(Weak<RefCell<ValueData>>);

impl ValueRef {
    fn new(value: &Rc<RefCell<ValueData>>) -> ValueRef {
        ValueRef(Rc::downgrade(value))
    }

    // Always panic if upgrade references a dropped value (autograd graph not DAG)
    fn with_borrow<R>(&self, f: impl FnOnce(&ValueData) -> R) -> R {
        let value_ptr = self.0.upgrade().expect("DAG properties of autograd graph violated");
        let borrow = value_ptr.borrow();
        f(&borrow)
    }

    // Always panic if upgrade references a dropped value (autograd graph not DAG)
    fn with_mut_borrow<R>(&self, f: impl FnOnce(&mut ValueData) -> R) -> R {
        let value_ptr = self.0.upgrade().expect("DAG properties of autograd graph violated");
        let mut borrow = value_ptr.borrow_mut();
        f(&mut borrow)
    }
}

impl PartialEq for ValueRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}

impl Eq for ValueRef {}

impl Hash for ValueRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

/******************************** unit tests ********************************/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let v = Value::build(4.2);
        assert_eq!(v.to_string(), "Value(data=4.2)");
    }

    #[test]
    fn add() {
        let a = Value::build(2.5);
        let b = Value::build(3.2);
        let c = &a + &b;
        let d = &(&a + &b) + &c;
        assert_eq!(c.get_data(), 5.7);
        assert_eq!(d.get_data(), 11.4);
        assert_eq!(c.to_string(), "Value(data=5.7)");
        assert_eq!(d.to_string(), "Value(data=11.4)");
    }

    #[test]
    fn add_grad() {
        let a = Value::build(2.5);
        let b = Value::build(3.2);
        let c = &a + &b;
        assert_eq!(c.get_data(), 5.7);
        assert_eq!(c.to_string(), "Value(data=5.7)");

        assert_eq!(a.get_grad(), 0.0);
        assert_eq!(b.get_grad(), 0.0);
        c.backward();
        assert_eq!(a.get_grad(), 1.0);
        assert_eq!(b.get_grad(), 1.0);
    }

    #[test]
    fn mul() {
        let a = Value::build(2.5);
        let b = Value::build(3.0);
        let c = &a * &b;
        let d = &(&a * &b) * &c;
        assert_eq!(c.get_data(), 7.5);
        assert_eq!(d.get_data(), 56.25);
        assert_eq!(c.to_string(), "Value(data=7.5)");
        assert_eq!(d.to_string(), "Value(data=56.25)");
    }

    #[test]
    fn mul_grad() {
        let a = Value::build(2.5);
        let b = Value::build(3.0);
        let c = &a * &b;
        assert_eq!(c.get_data(), 7.5);
        assert_eq!(c.to_string(), "Value(data=7.5)");

        assert_eq!(a.get_grad(), 0.0);
        assert_eq!(b.get_grad(), 0.0);
        c.backward();
        assert_eq!(a.get_grad(), 3.0);
        assert_eq!(b.get_grad(), 2.5);
    }
}

