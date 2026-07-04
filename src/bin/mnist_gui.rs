use std::{
    error::Error,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread,
    time::Duration,
};

use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
use knok::{prelude::*, Engine};
use knok_mnist_training::{
    argmax, batch_with_labels, data, evaluate, forward_graphs, grad_graphs, run_logits, sgd, Model,
    BATCH, CLASSES, IMAGE_PIXELS,
};

type GuiResult<T> = std::result::Result<T, Box<dyn Error>>;

const IMAGE_SIDE: usize = 28;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([920.0, 640.0])
            .with_min_inner_size([780.0, 520.0]),
        ..Default::default()
    };

    eframe::run_native(
        "knok MNIST",
        options,
        Box::new(|_| Ok(Box::<MnistGuiApp>::default())),
    )
}

#[derive(Clone)]
struct TrainConfig {
    data_dir: String,
    epochs: usize,
    learning_rate: f32,
    train_batches: usize,
    eval_batches: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".to_owned(),
            epochs: 2,
            learning_rate: 0.1,
            train_batches: 100,
            eval_batches: 20,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Draw,
    Erase,
}

struct TrainingHandle {
    receiver: Receiver<TrainingEvent>,
    cancel: Arc<AtomicBool>,
}

enum TrainingEvent {
    Started {
        train_len: usize,
        test_len: usize,
    },
    Epoch {
        epoch: usize,
        epochs: usize,
        loss: f32,
        accuracy: f32,
    },
    Finished(Model),
    Cancelled(Model),
    Failed(String),
}

struct Prediction {
    digit: u8,
    probabilities: [f32; CLASSES],
}

struct MnistGuiApp {
    config: TrainConfig,
    model: Option<Model>,
    training: Option<TrainingHandle>,
    status: String,
    canvas: [f32; IMAGE_PIXELS],
    brush_radius: f32,
    tool: Tool,
    last_paint_cell: Option<(f32, f32)>,
    prediction: Option<Prediction>,
    forward_engine: Option<Engine>,
}

impl Default for MnistGuiApp {
    fn default() -> Self {
        Self {
            config: TrainConfig::default(),
            model: None,
            training: None,
            status: "Idle".to_owned(),
            canvas: [0.0; IMAGE_PIXELS],
            brush_radius: 1.7,
            tool: Tool::Draw,
            last_paint_cell: None,
            prediction: None,
            forward_engine: None,
        }
    }
}

impl eframe::App for MnistGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_training(ctx);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("knok MNIST");
                ui.separator();
                ui.label(&self.status);
            });
        });

        egui::SidePanel::left("training")
            .resizable(false)
            .default_width(270.0)
            .show(ctx, |ui| {
                self.training_ui(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |columns| {
                self.canvas_ui(&mut columns[0]);
                self.prediction_ui(&mut columns[1]);
            });
        });
    }
}

impl MnistGuiApp {
    fn training_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Training");
        ui.add_space(8.0);

        ui.label("Data");
        ui.text_edit_singleline(&mut self.config.data_dir);

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Epochs");
            ui.add(egui::DragValue::new(&mut self.config.epochs).range(1..=200));
        });
        ui.horizontal(|ui| {
            ui.label("Batches");
            ui.add(egui::DragValue::new(&mut self.config.train_batches).range(1..=10_000));
        });
        ui.horizontal(|ui| {
            ui.label("Eval");
            ui.add(egui::DragValue::new(&mut self.config.eval_batches).range(1..=1_000));
        });
        ui.add(
            egui::Slider::new(&mut self.config.learning_rate, 0.001..=1.0)
                .logarithmic(true)
                .text("LR"),
        );

        ui.add_space(12.0);
        let is_training = self.training.is_some();
        if ui
            .add_enabled(!is_training, egui::Button::new("Train"))
            .clicked()
        {
            self.start_training();
        }

        if ui
            .add_enabled(is_training, egui::Button::new("Stop"))
            .clicked()
        {
            if let Some(training) = &self.training {
                training.cancel.store(true, Ordering::Relaxed);
                self.status = "Stopping after current batch".to_owned();
            }
        }

        if ui
            .add_enabled(!is_training, egui::Button::new("Reset model"))
            .clicked()
        {
            self.model = None;
            self.prediction = None;
            self.status = "Model reset".to_owned();
        }
    }

    fn canvas_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Canvas");
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.tool == Tool::Draw, "Draw")
                .clicked()
            {
                self.tool = Tool::Draw;
            }
            if ui
                .selectable_label(self.tool == Tool::Erase, "Erase")
                .clicked()
            {
                self.tool = Tool::Erase;
            }
            if ui.button("Clear").clicked() {
                self.canvas = [0.0; IMAGE_PIXELS];
                self.prediction = None;
                self.last_paint_cell = None;
            }
            if ui.button("Predict").clicked() {
                self.predict();
            }
        });

        ui.add(egui::Slider::new(&mut self.brush_radius, 0.5..=3.5).text("Brush"));
        ui.add_space(8.0);

        let side = ui.available_width().clamp(280.0, 420.0);
        let (response, painter) =
            ui.allocate_painter(egui::Vec2::splat(side), egui::Sense::click_and_drag());
        let rect = response.rect;
        painter.rect_filled(rect, 4.0, Color32::BLACK);

        let cell = rect.width() / IMAGE_SIDE as f32;
        for row in 0..IMAGE_SIDE {
            for col in 0..IMAGE_SIDE {
                let value = self.canvas[row * IMAGE_SIDE + col];
                if value <= 0.0 {
                    continue;
                }
                let shade = (value.clamp(0.0, 1.0) * 255.0) as u8;
                let min = Pos2::new(
                    rect.left() + col as f32 * cell,
                    rect.top() + row as f32 * cell,
                );
                let max = Pos2::new(min.x + cell + 0.5, min.y + cell + 0.5);
                painter.rect_filled(Rect::from_min_max(min, max), 0.0, Color32::from_gray(shade));
            }
        }
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_gray(85)));

        let pointer_down =
            ui.input(|input| input.pointer.primary_down() || input.pointer.secondary_down());
        if pointer_down && (response.hovered() || response.dragged()) {
            if let Some(pos) = response.interact_pointer_pos() {
                if rect.contains(pos) {
                    let erase = self.tool == Tool::Erase
                        || ui.input(|input| input.pointer.secondary_down());
                    self.paint_pointer(pos, rect, erase);
                    self.prediction = None;
                }
            }
        } else {
            self.last_paint_cell = None;
        }
    }

    fn prediction_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Prediction");
        ui.add_space(16.0);

        if let Some(prediction) = &self.prediction {
            ui.label(format!("Digit {}", prediction.digit));
            ui.add_space(12.0);

            for digit in 0..CLASSES {
                let probability = prediction.probabilities[digit];
                ui.horizontal(|ui| {
                    ui.label(format!("{digit}"));
                    ui.add(
                        egui::ProgressBar::new(probability)
                            .desired_width(220.0)
                            .text(format!("{:.1}%", probability * 100.0)),
                    );
                });
            }
        } else {
            ui.label("No prediction");
        }
    }

    fn start_training(&mut self) {
        let config = self.config.clone();
        let initial_model = self.model.clone();
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);

        thread::spawn(move || {
            if let Err(error) = run_training(config, initial_model, worker_cancel, &tx) {
                let _ = tx.send(TrainingEvent::Failed(error.to_string()));
            }
        });

        self.training = Some(TrainingHandle {
            receiver: rx,
            cancel,
        });
        self.status = "Training started".to_owned();
    }

    fn poll_training(&mut self, ctx: &egui::Context) {
        let events = self
            .training
            .as_ref()
            .map(|training| training.receiver.try_iter().collect::<Vec<_>>())
            .unwrap_or_default();

        let mut finished = false;
        for event in events {
            match event {
                TrainingEvent::Started {
                    train_len,
                    test_len,
                } => {
                    self.status = format!("Loaded {train_len} train / {test_len} test examples");
                }
                TrainingEvent::Epoch {
                    epoch,
                    epochs,
                    loss,
                    accuracy,
                } => {
                    self.status =
                        format!("Epoch {epoch}/{epochs}: loss={loss:.4}, accuracy={accuracy:.2}%");
                }
                TrainingEvent::Finished(model) => {
                    self.model = Some(model);
                    self.prediction = None;
                    self.status = "Training complete".to_owned();
                    finished = true;
                }
                TrainingEvent::Cancelled(model) => {
                    self.model = Some(model);
                    self.prediction = None;
                    self.status = "Training stopped".to_owned();
                    finished = true;
                }
                TrainingEvent::Failed(error) => {
                    self.status = format!("Training failed: {error}");
                    finished = true;
                }
            }
        }

        if finished {
            self.training = None;
        } else if self.training.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn predict(&mut self) {
        let Some(model) = self.model.clone() else {
            self.status = "Train a model first".to_owned();
            return;
        };

        let result = (|| -> GuiResult<Prediction> {
            if self.forward_engine.is_none() {
                self.forward_engine = Some(Engine::for_artifact(
                    forward_graphs::mnist_logits::artifact(),
                )?);
            }

            let engine = self
                .forward_engine
                .as_ref()
                .expect("forward engine exists after initialization");
            let mut batch = vec![0.0_f32; BATCH * IMAGE_PIXELS];
            batch[..IMAGE_PIXELS].copy_from_slice(&self.canvas);
            let logits = run_logits(engine, &model, Tensor2::from_vec(batch)?)?;
            let row = &logits.as_slice()[..CLASSES];

            Ok(Prediction {
                digit: argmax(row),
                probabilities: softmax(row),
            })
        })();

        match result {
            Ok(prediction) => {
                self.status = format!("Predicted {}", prediction.digit);
                self.prediction = Some(prediction);
            }
            Err(error) => {
                self.status = format!("Prediction failed: {error}");
                self.prediction = None;
            }
        }
    }

    fn paint_pointer(&mut self, pos: Pos2, rect: Rect, erase: bool) {
        let cell = rect.width() / IMAGE_SIDE as f32;
        let col = ((pos.x - rect.left()) / cell).clamp(0.0, (IMAGE_SIDE - 1) as f32);
        let row = ((pos.y - rect.top()) / cell).clamp(0.0, (IMAGE_SIDE - 1) as f32);

        if let Some((last_row, last_col)) = self.last_paint_cell {
            let distance = ((row - last_row).powi(2) + (col - last_col).powi(2)).sqrt();
            let steps = distance.ceil().max(1.0) as usize;
            for step in 0..=steps {
                let t = step as f32 / steps as f32;
                self.paint_cell(
                    last_row + (row - last_row) * t,
                    last_col + (col - last_col) * t,
                    erase,
                );
            }
        } else {
            self.paint_cell(row, col, erase);
        }

        self.last_paint_cell = Some((row, col));
    }

    fn paint_cell(&mut self, row: f32, col: f32, erase: bool) {
        let radius = self.brush_radius;
        let min_row = (row - radius).floor().max(0.0) as usize;
        let max_row = (row + radius).ceil().min((IMAGE_SIDE - 1) as f32) as usize;
        let min_col = (col - radius).floor().max(0.0) as usize;
        let max_col = (col + radius).ceil().min((IMAGE_SIDE - 1) as f32) as usize;

        for y in min_row..=max_row {
            for x in min_col..=max_col {
                let dy = y as f32 - row;
                let dx = x as f32 - col;
                let distance = (dx * dx + dy * dy).sqrt();
                let strength = (1.0 - distance / (radius + 0.5)).clamp(0.0, 1.0);
                if strength == 0.0 {
                    continue;
                }

                let pixel = &mut self.canvas[y * IMAGE_SIDE + x];
                if erase {
                    *pixel = (*pixel - strength).max(0.0);
                } else {
                    *pixel = (*pixel + strength).min(1.0);
                }
            }
        }
    }
}

fn run_training(
    config: TrainConfig,
    initial_model: Option<Model>,
    cancel: Arc<AtomicBool>,
    tx: &Sender<TrainingEvent>,
) -> GuiResult<()> {
    let (train, test) = data::load_or_download(PathBuf::from(&config.data_dir))?;
    let _ = tx.send(TrainingEvent::Started {
        train_len: train.len(),
        test_len: test.len(),
    });

    let mut rng = sgd::Rng::new(0x5eed);
    let mut model = match initial_model {
        Some(model) => model,
        None => Model::new()?,
    };
    let mut order = (0..train.len()).collect::<Vec<_>>();

    let grad_engine = Engine::for_artifact(grad_graphs::mnist_loss_value_and_grad::artifact())?;
    let forward_engine = Engine::for_artifact(forward_graphs::mnist_logits::artifact())?;

    for epoch in 1..=config.epochs {
        if cancel.load(Ordering::Relaxed) {
            let _ = tx.send(TrainingEvent::Cancelled(model));
            return Ok(());
        }

        rng.shuffle(&mut order);
        let max_batches = config.train_batches.min(order.len() / BATCH);
        let mut loss_sum = 0.0_f32;

        for batch in 0..max_batches {
            if cancel.load(Ordering::Relaxed) {
                let _ = tx.send(TrainingEvent::Cancelled(model));
                return Ok(());
            }

            let batch_indices = &order[batch * BATCH..(batch + 1) * BATCH];
            let (images, labels) = batch_with_labels(&train, batch_indices)?;
            let (w1, b1, w2, b2, w3, b3) = model.tensors()?;

            let (loss, _grad_images, grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3) =
                grad_graphs::mnist_loss_value_and_grad::run(
                    &grad_engine,
                    images,
                    labels,
                    w1,
                    b1,
                    w2,
                    b2,
                    w3,
                    b3,
                )?;

            sgd::step(&mut model.w1, grad_w1.as_slice(), config.learning_rate);
            sgd::step(&mut model.b1, grad_b1.as_slice(), config.learning_rate);
            sgd::step(&mut model.w2, grad_w2.as_slice(), config.learning_rate);
            sgd::step(&mut model.b2, grad_b2.as_slice(), config.learning_rate);
            sgd::step(&mut model.w3, grad_w3.as_slice(), config.learning_rate);
            sgd::step(&mut model.b3, grad_b3.as_slice(), config.learning_rate);

            loss_sum += loss.as_slice()[0];
        }

        let avg_loss = loss_sum / max_batches.max(1) as f32;
        let accuracy = evaluate(&forward_engine, &model, &test, config.eval_batches)?;
        let _ = tx.send(TrainingEvent::Epoch {
            epoch,
            epochs: config.epochs,
            loss: avg_loss,
            accuracy,
        });
    }

    let _ = tx.send(TrainingEvent::Finished(model));
    Ok(())
}

fn softmax(logits: &[f32]) -> [f32; CLASSES] {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |lhs, rhs| lhs.max(rhs));
    let mut sum = 0.0_f32;
    let mut probabilities = [0.0_f32; CLASSES];

    for (probability, logit) in probabilities.iter_mut().zip(logits) {
        *probability = (*logit - max).exp();
        sum += *probability;
    }

    for probability in &mut probabilities {
        *probability /= sum.max(f32::MIN_POSITIVE);
    }

    probabilities
}
