mod data;
mod sgd;

use std::{env, error::Error, path::PathBuf};

use knok::{prelude::*, Engine};

const BATCH: usize = 64;
const IMAGE_PIXELS: usize = 28 * 28;
const HIDDEN: usize = 128;
const CLASSES: usize = 10;

knok::generated_graphs!(pub mod forward_graphs, "knok_forward_graphs.rs");
knok::generated_graphs!(pub mod grad_graphs, "knok_grad_graphs.rs");

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Config {
    data_dir: PathBuf,
    epochs: usize,
    learning_rate: f32,
    train_batches: usize,
    eval_batches: usize,
}

struct Model {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

fn main() -> Result<()> {
    let config = Config::from_env();
    let (train, test) = data::load_or_download(&config.data_dir)?;
    let mut rng = sgd::Rng::new(0x5eed);
    let mut model = Model::new(&mut rng);
    let mut order = (0..train.len()).collect::<Vec<_>>();

    let grad_engine = Engine::for_artifact(grad_graphs::mnist_loss_value_and_grad::artifact())?;
    let forward_engine = Engine::for_artifact(forward_graphs::mnist_logits::artifact())?;

    println!(
        "training on {} examples, testing on {}, batch={BATCH}, hidden={HIDDEN}",
        train.len(),
        test.len()
    );

    for epoch in 1..=config.epochs {
        rng.shuffle(&mut order);
        let max_batches = config.train_batches.min(order.len() / BATCH);
        let mut loss_sum = 0.0_f32;

        for batch in 0..max_batches {
            let batch_indices = &order[batch * BATCH..(batch + 1) * BATCH];
            let (images, labels) = batch_with_labels(&train, batch_indices)?;
            let (w1, b1, w2, b2) = model.tensors()?;

            let (loss, _grad_images, grad_w1, grad_b1, grad_w2, grad_b2) =
                grad_graphs::mnist_loss_value_and_grad::run(
                    &grad_engine,
                    images,
                    labels,
                    w1,
                    b1,
                    w2,
                    b2,
                )?;

            sgd::step(&mut model.w1, grad_w1.as_slice(), config.learning_rate);
            sgd::step(&mut model.b1, grad_b1.as_slice(), config.learning_rate);
            sgd::step(&mut model.w2, grad_w2.as_slice(), config.learning_rate);
            sgd::step(&mut model.b2, grad_b2.as_slice(), config.learning_rate);

            loss_sum += loss.as_slice()[0];
        }

        let avg_loss = loss_sum / max_batches.max(1) as f32;
        let accuracy = evaluate(&forward_engine, &model, &test, config.eval_batches)?;
        println!("epoch {epoch}: loss={avg_loss:.4} eval_accuracy={accuracy:.2}%");
    }

    Ok(())
}

impl Config {
    fn from_env() -> Self {
        Self {
            data_dir: env::var_os("MNIST_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data")),
            epochs: env_usize("EPOCHS", 2),
            learning_rate: env_f32("LR", 0.1),
            train_batches: env_usize("TRAIN_BATCHES", 100),
            eval_batches: env_usize("EVAL_BATCHES", 20),
        }
    }
}

impl Model {
    fn new(rng: &mut sgd::Rng) -> Self {
        Self {
            w1: init_matrix(rng, IMAGE_PIXELS, HIDDEN),
            b1: vec![0.0; HIDDEN],
            w2: init_matrix(rng, HIDDEN, CLASSES),
            b2: vec![0.0; CLASSES],
        }
    }

    fn tensors(
        &self,
    ) -> Result<(
        Tensor2<f32, IMAGE_PIXELS, HIDDEN>,
        Tensor1<f32, HIDDEN>,
        Tensor2<f32, HIDDEN, CLASSES>,
        Tensor1<f32, CLASSES>,
    )> {
        Ok((
            Tensor2::from_vec(self.w1.clone())?,
            Tensor1::from_vec(self.b1.clone())?,
            Tensor2::from_vec(self.w2.clone())?,
            Tensor1::from_vec(self.b2.clone())?,
        ))
    }
}

fn init_matrix(rng: &mut sgd::Rng, rows: usize, cols: usize) -> Vec<f32> {
    let scale = (6.0_f32 / (rows + cols) as f32).sqrt();
    (0..rows * cols)
        .map(|_| rng.uniform(-scale, scale))
        .collect()
}

fn batch_with_labels(
    dataset: &data::Mnist,
    indices: &[usize],
) -> Result<(Tensor2<f32, BATCH, IMAGE_PIXELS>, Tensor2<i64, BATCH, 1>)> {
    let mut images = Vec::with_capacity(BATCH * IMAGE_PIXELS);
    let mut labels = Vec::with_capacity(BATCH);
    for &index in indices {
        images.extend_from_slice(dataset.image(index));
        labels.push(i64::from(dataset.label(index)));
    }
    Ok((Tensor2::from_vec(images)?, Tensor2::from_vec(labels)?))
}

fn batch_images(dataset: &data::Mnist, start: usize) -> Result<Tensor2<f32, BATCH, IMAGE_PIXELS>> {
    let mut images = Vec::with_capacity(BATCH * IMAGE_PIXELS);
    for index in start..start + BATCH {
        images.extend_from_slice(dataset.image(index));
    }
    Ok(Tensor2::from_vec(images)?)
}

fn evaluate(
    engine: &Engine,
    model: &Model,
    dataset: &data::Mnist,
    max_batches: usize,
) -> Result<f32> {
    let batches = max_batches.min(dataset.len() / BATCH);
    let mut correct = 0_usize;
    let mut total = 0_usize;

    for batch in 0..batches {
        let images = batch_images(dataset, batch * BATCH)?;
        let (w1, b1, w2, b2) = model.tensors()?;
        let logits = forward_graphs::mnist_logits::run(engine, images, w1, b1, w2, b2)?;

        for (row, label_index) in logits.as_slice().chunks_exact(CLASSES).zip(0..) {
            let predicted = argmax(row);
            let expected = dataset.label(batch * BATCH + label_index);
            correct += usize::from(predicted == expected);
            total += 1;
        }
    }

    Ok(100.0 * correct as f32 / total.max(1) as f32)
}

fn argmax(values: &[f32]) -> u8 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index as u8)
        .expect("class logits are non-empty")
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
