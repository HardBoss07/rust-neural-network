#[derive(Clone, Debug)]
pub enum NeuronType {
    Input,
    Middle,
    Output,
}

#[derive(Clone, Debug)]
pub struct Neuron {
    pub neuron_type: NeuronType,
    pub value: f32,
    pub weights: Vec<f32>,
    pub output_refs: Vec<usize>,
    pub forward_outputs: Vec<(usize, f32)>,
}

impl Neuron {
    // constructor for input neuron
    pub fn new_input() -> Self {
        Neuron {
            neuron_type: NeuronType::Input,
            value: 0.0,
            weights: vec![],
            output_refs: vec![],
            forward_outputs: vec![],
        }
    }

    // constructor for hidden & output neurons
    pub fn new(neuron_type: NeuronType, output_refs: Vec<usize>, weights: Vec<f32>) -> Self {
        assert_eq!(output_refs.len(), weights.len(), "Mismatched weights and references");
        Neuron {
            neuron_type,
            value: 0.0,
            weights,
            output_refs,
            forward_outputs: vec![],
        }
    }

    pub fn activate(&self, input_sum: f32) -> f32 {
        1.0 / (1.0 + (-input_sum).exp()) // sigmoid activation function
    }

    // calculate forward outputs based off value
    pub fn calculate_forward_outputs(&mut self) {
        self.forward_outputs = self
            .output_refs
            .iter()
            .zip(&self.weights)
            .map(|(&index, &weight)| (index, self.value * weight))
            .collect();
    }
}