use rand::Rng;

pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weight1: Vec<Vec<f32>>,
    bias1: Vec<f32>,
    weight2: Vec<Vec<f32>>,
    bias2: Vec<f32>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weight1 = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let bias1 = vec![0.0; hidden_size];

        let weight2 = (0..output_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let bias2 = vec![0.0; output_size];

        Self {
            input_size,
            hidden_size,
            output_size,
            weight1,
            bias1,
            weight2,
            bias2,
        }
    }

    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    fn softmax(input: Vec<f32>) -> Vec<f32> {
        let max_input = input
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        
        let mut exponent_vals = Vec::with_capacity(input.len());
        for &x in input.iter() {
            let shifted = x - max_input;
            let exponent_x = shifted.exp();
            exponent_vals.push(exponent_x);
        }

        let sum_exponents: f32 = exponent_vals.iter().sum();

        let mut softmax_output = Vec::with_capacity(input.len());
        for &exponent_x in exponent_vals.iter() {
            softmax_output.push(exponent_x / sum_exponents);
        }

        softmax_output
    }

    pub fn feedforward(&self, input: Vec<f32>) -> Vec<f32> {
        assert_eq!(input.len(), self.input_size);

        let mut hidden = vec![0.0; self.hidden_size];

        for hidden_neuron in 0..self.hidden_size {
            let mut sum = 0.0;

            for input_neuron in 0..self.input_size {
                let weight = self.weight1[hidden_neuron][input_neuron];
                sum += weight * input[input_neuron];
            }

            sum += self.bias1[hidden_neuron];

            hidden[hidden_neuron] = Self::relu(sum);
        }

        let mut output = vec![0.0; self.output_size];

        for output_neuron in 0..self.output_size {
            let mut sum = 0.0;

            for hidden_neuron in 0..self.hidden_size {
                let weight = self.weight2[output_neuron][hidden_neuron];
                sum += weight * hidden[hidden_neuron];
            }

            sum += self.bias2[output_neuron];

            output[output_neuron] = sum;
        }

        Self::softmax(output)
    }
}