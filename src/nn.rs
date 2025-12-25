use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::{Value, ValueFactory};
use std::iter::{zip, once};

pub trait Module {
    fn zero_grad(&self) {
        self.parameters()
            .for_each(|v| v.set_data(0.0));
    }

    fn parameters(&self) -> impl Iterator<Item = &Value>;
}

struct Neuron {
    w: Vec<Value>,
    b: Value
}

impl Neuron {
    fn new(vf: &ValueFactory, nin: usize) -> Neuron {
        Neuron {
            w: (0..nin).map(|_| vf.value(rand::random::<f64>() * 2.0 - 1.0)).collect(),
            b: vf.value(rand::random::<f64>() * 2.0 - 1.0)
        }
    }

    fn new_with_seed(vf: &ValueFactory, nin: usize, seed: u64) -> Neuron {
        let mut rng = StdRng::seed_from_u64(seed);
        Neuron {
            w: (0..nin).map(|_| vf.value(rng.random_range(-1.0..1.0))).collect(),
            b: vf.value(rand::random::<f64>() * 2.0 - 1.0)
        }
    }

    fn call(&self, x: &Vec<Value>) -> Value {
        let out = zip(&self.w, x)
            .map(|(wi, xi)| wi * xi)
            .fold(self.b.clone(), |acc, v| &acc + &v);
        out.tanh()
    }
}

impl Module for Neuron {
    fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.w.iter().chain(once(&self.b))
    }
}

struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    fn new(vf: &ValueFactory, nin: usize, nout: usize) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(vf, nin)).collect()
        }
    }

    fn new_with_seed(vf: &ValueFactory, nin: usize, nout: usize, seed: u64) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new_with_seed(vf, nin, seed)).collect()
        }
    }

    fn call(&self, x: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|n| n.parameters())
    }
}

pub struct MLP {
    layers: Vec<Layer>
}

impl MLP {
    pub fn new(vf: &ValueFactory, nin: usize, nout: &Vec<usize>) -> MLP {
        let sz: Vec<usize> = once(nin)
            .chain(nout.iter().copied())
            .collect();
        MLP {
            layers: (0..nout.len()).map(|i| Layer::new(vf, sz[i], sz[i + 1])).collect()
        }
    }

    pub fn new_with_seed(vf: &ValueFactory, nin: usize, nout: &Vec<usize>, seed: u64) -> MLP {
        let sz: Vec<usize> = std::iter::once(nin)
            .chain(nout.iter().copied())
            .collect();
        MLP {
            layers: (0..nout.len()).map(|i| Layer::new_with_seed(vf, sz[i], sz[i + 1], seed)).collect()
        }
    }

    pub fn call(&self, x: &Vec<Value>) -> Vec<Value> {
        self.layers
            .iter()
            .fold(x.to_vec(), |acc, layer: &Layer| layer.call(&acc))
    }
}

impl Module for MLP {
    fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.layers.iter().flat_map(|l| l.parameters())
    }
}
