mod neuralnetwork;
use neuralnetwork::NeuralNetwork;

use eframe::{egui, egui::Vec2, egui::SidePanel};
use egui::{Color32, Pos2};

const BUFFER_SIZE: usize = 64;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: {
            egui::ViewportBuilder::default()
            .with_inner_size([800.0, 450.0])
            .with_title("Shape Recognizer")
        },
        ..Default::default()
    };
    eframe::run_native(
        "Shape Recognizer",
        options,
        Box::new(|_cc| Ok(Box::<MyApp>::default())),
    )
}

struct MyApp {
    drawing: Vec<Pos2>,
    is_drawing: bool,
    buffer: [[u8; BUFFER_SIZE]; BUFFER_SIZE],
    nn: NeuralNetwork,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            drawing: Vec::new(),
            is_drawing: false,
            buffer: [[0; BUFFER_SIZE]; BUFFER_SIZE],
            nn: NeuralNetwork::new(BUFFER_SIZE * BUFFER_SIZE, 64, 3),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Draw your shape below:");

            let available_size = ui.available_size();
            let (response, painter) =
                ui.allocate_painter(available_size, egui::Sense::drag());

            let pointer_pos = response.interact_pointer_pos();

            if response.drag_started() {
                self.is_drawing = true;
                self.drawing.clear();
                self.buffer = [[0; BUFFER_SIZE]; BUFFER_SIZE];
            }

            if response.drag_stopped() {
                self.is_drawing = false;
            }

            if self.is_drawing {
                if let Some(pos) = pointer_pos {
                    self.drawing.push(pos);

                    let canvas_rect = response.rect;
                    let rel_x = (pos.x - canvas_rect.left()) / canvas_rect.width();
                    let rel_y = (pos.y - canvas_rect.top()) / canvas_rect.height();

                    let x = (rel_x * BUFFER_SIZE as f32) as usize;
                    let y = (rel_y * BUFFER_SIZE as f32) as usize;

                    if x < BUFFER_SIZE && y < BUFFER_SIZE {
                        self.buffer[y][x] = 255;
                    }
                }
            }

            for window in self.drawing.windows(2) {
                let [p1, p2] = [window[0], window[1]];
                painter.line_segment([p1, p2], egui::Stroke::new(2.0, Color32::WHITE));
            }

            SidePanel::right("buffer_panel").show(ctx, |ui| {
                ui.heading("Neural Net Output");

                let input = self.buffer.iter().flat_map(|row| {
                    row.iter().map(|&val| val as f32 / 255.0)
                }).collect::<Vec<_>>();

                let output = self.nn.feedforward(input);

                let class_names = ["Circle", "Square", "Triangle"];

                for (i, &prob) in output.iter().enumerate() {
                    let label = class_names.get(i).map(|&s| s).unwrap_or_else(|| {
                        Box::leak(format!("Class {}", i).into_boxed_str())
                    });
                    ui.label(format!("{label}: {:.2}%", prob * 100.0));
                    ui.add(egui::widgets::ProgressBar::new(prob).show_percentage());
                }

                ui.separator();
                ui.heading("Input Buffer Preview");

                let pixel_size = 4.0;
                let (response, painter) = ui.allocate_painter(
                    Vec2::new(pixel_size * BUFFER_SIZE as f32, pixel_size * BUFFER_SIZE as f32),
                    egui::Sense::hover(),
                );

                for y in 0..BUFFER_SIZE {
                    for x in 0..BUFFER_SIZE {
                        let brightness = self.buffer[y][x];
                        let color = Color32::from_gray(brightness);

                        let top_left = response.rect.left_top()
                            + egui::vec2(x as f32 * pixel_size, y as f32 * pixel_size);
                        let rect = egui::Rect::from_min_size(top_left, egui::vec2(pixel_size, pixel_size));

                        painter.rect_filled(rect, 0.0, color);
                    }
                }
            });

            ctx.request_repaint();
        });
    }
}
