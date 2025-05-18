use crate::layer::Layer;
use crate::neuron::NeuronType;

pub struct Network {
    pub input_layer: Layer,
    pub hidden_layer: Layer,
    pub output_layer: Layer,
}

impl Network {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut input_layer = Layer::new(NeuronType::Input, input_size);
        let mut hidden_layer = Layer::new(NeuronType::Middle, hidden_size);
        let output_layer = Layer::new(NeuronType::Output, output_size);

        input_layer.connect_to(&hidden_layer);
        hidden_layer.connect_to(&output_layer);

        Network {
            input_layer,
            hidden_layer,
            output_layer,
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        // forward values from input to hidden and calculates its values
        self.input_layer.forward(Some(input), &mut self.hidden_layer);

        // forward values from hidden to output
        self.hidden_layer.forward(None, &mut self.output_layer);

        // collect values from output layer
        self.output_layer.neurons.iter().map(|n| n.value).collect()
    }
}
