use rand::Rng;

use std::fs::OpenOptions;
use std::io::Write;

use std::fs::File;
use std::io::{Read};

pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weight1: Vec<Vec<f32>>,
    bias1: Vec<f32>,
    weight2: Vec<Vec<f32>>,
    bias2: Vec<f32>,
    read_weights: bool,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, read_weights: bool) -> Self {
        let weight1 = if read_weights {
            Self::load_weights("weights1.glg")
        } else {
            Self::generate_random_weights(hidden_size, input_size)
        };

        let bias1 = vec![0.0; hidden_size];

        let weight2 = if read_weights {
            Self::load_weights("weights2.glg")
        } else {
            Self::generate_random_weights(output_size, hidden_size)
        };

        let bias2 = vec![0.0; output_size];

        Self {
            input_size,
            hidden_size,
            output_size,
            weight1,
            bias1,
            weight2,
            bias2,
            read_weights,
        }
    }

    fn generate_random_weights(rows: usize, cols: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect()
    }

    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    fn softmax(input: Vec<f32>) -> Vec<f32> {
        let max_input = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exponent_vals: Vec<f32> = input.iter().map(|&x| (x - max_input).exp()).collect();
        let sum_exponents: f32 = exponent_vals.iter().sum();
        exponent_vals.iter().map(|&x| x / sum_exponents).collect()
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

    fn load_weights(path: &str) -> Vec<Vec<f32>> {
        let mut file = File::open(path).expect("Failed to open weight file:");
        let mut contents = String::new();
        file.read_to_string(&mut contents).expect("Failed to read file");

        let clean = contents
            .replace("\n", "")
            .replace("\r", "")
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .to_string();

        let rows: Vec<Vec<f32>> = clean
            .split("],")
            .map(|row| {
                row.replace('[', "")
                    .replace(']', "")
                    .split(',')
                    .filter_map(|s| {
                        let token = s.trim();
                        if token.is_empty() {
                            None
                        } else {
                            match token.parse::<f32>() {
                                Ok(n) => Some(n),
                                Err(e) => {
                                    println!("Failed to parse '{}': {:?}", token, e);
                                    None
                                }
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        rows
    }

    pub fn write_weights(&self) {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("weights1.glg")
            .expect("Unable to open weights1.glg for writing");
        writeln!(file, "{:?}", self.weight1).expect("Failed to write to weights1.glg");

        let mut file2 = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("weights2.glg")
            .expect("Unable to open weights2.glg for writing");
        writeln!(file2, "{:?}", self.weight2).expect("Failed to write to weights2.glg");
    }
}