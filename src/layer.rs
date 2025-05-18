use crate::neuron::{Neuron, NeuronType};

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neuron_type: NeuronType, size: usize) -> Self {
        let neurons = match neuron_type {
            NeuronType::Input => (0..size).map(|_| Neuron::new_input()).collect(),
            _ => (0..size).map(|_| Neuron::new(neuron_type.clone(), vec![], vec![])).collect(),
        };
        Layer { neurons }
    }

    // connect current layer to next layer
    pub fn connect_to(&mut self, next_layer: &Layer) {
        for neuron in &mut self.neurons {
            neuron.output_refs = (0..next_layer.neurons.len()).collect();
            neuron.weights = (0..next_layer.neurons.len())
                .map(|_| rand::random::<f32>())
                .collect();
        }
    }

    // forward values to next layer
    pub fn forward(&mut self, input_values: Option<&[f32]>, next_layer: &mut Layer) {
        match self.neurons.first().map(|n| &n.neuron_type) {
            Some(NeuronType::Input) => {
                for (i, &val) in input_values.unwrap().iter().enumerate() {
                    self.neurons[i].value = val;
                    self.neurons[i].calculate_forward_outputs();
                }
            },
            _ => {
                for neuron in &mut self.neurons {
                    let mut sum = 0.0;
                    for (j, weight) in neuron.output_refs.iter().zip(&neuron.weights) {
                        sum += next_layer.neurons[*j].value * weight;
                    }
                    neuron.value = neuron.activate(sum);
                    neuron.calculate_forward_outputs();
                }
            }
        }
    }
}