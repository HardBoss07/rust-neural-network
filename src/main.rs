use eframe::{egui, egui::Vec2};
use egui::{Color32, Pos2};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Shape Recognizer",
        options,
        Box::new(|_cc| Ok(Box::<MyApp>::default())),
    )
}

#[derive(Default)]
struct MyApp {
    drawing: Vec<Pos2>,
    is_drawing: bool,
    buffer: [[u8; 28]; 28],
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
            }

            if response.drag_released() {
                self.is_drawing = false;
            }

            if self.is_drawing {
                if let Some(pos) = pointer_pos {
                    self.drawing.push(pos);
                
                    let canvas_rect = response.rect;
                    let rel_x = (pos.x - canvas_rect.left()) / canvas_rect.width();
                    let rel_y = (pos.y - canvas_rect.top()) / canvas_rect.height();
                
                    let x = (rel_x * 28.0) as usize;
                    let y = (rel_y * 28.0) as usize;
                
                    if x < 28 && y < 28 {
                        self.buffer[y][x] = 255;
                    }
                }
            }

            for window in self.drawing.windows(2) {
                let [p1, p2] = [window[0], window[1]];
                painter.line_segment([p1, p2], egui::Stroke::new(2.0, Color32::WHITE));
            }

            ctx.request_repaint();
        });
    }
}
