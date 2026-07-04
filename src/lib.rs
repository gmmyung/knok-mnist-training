pub mod data;
pub mod sgd;

use std::error::Error;

use knok::{prelude::*, Engine};

pub const BATCH: usize = 64;
pub const IMAGE_PIXELS: usize = 28 * 28;
pub const HIDDEN: usize = 256;
pub const CLASSES: usize = 10;

knok::generated_graphs!(pub mod forward_graphs, "knok_forward_graphs.rs");
knok::generated_graphs!(pub mod grad_graphs, "knok_grad_graphs.rs");

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone)]
pub struct Model {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub w3: Vec<f32>,
    pub b3: Vec<f32>,
}

impl Model {
    pub fn new() -> Result<Self> {
        let (w1, w2, w3) = forward_graphs::mnist_initial_weights::call()?;
        Ok(Self {
            w1: w1.as_slice().to_vec(),
            b1: vec![0.0; HIDDEN],
            w2: w2.as_slice().to_vec(),
            b2: vec![0.0; HIDDEN],
            w3: w3.as_slice().to_vec(),
            b3: vec![0.0; CLASSES],
        })
    }

    pub fn tensors(
        &self,
    ) -> Result<(
        Tensor2<f32, IMAGE_PIXELS, HIDDEN>,
        Tensor1<f32, HIDDEN>,
        Tensor2<f32, HIDDEN, HIDDEN>,
        Tensor1<f32, HIDDEN>,
        Tensor2<f32, HIDDEN, CLASSES>,
        Tensor1<f32, CLASSES>,
    )> {
        Ok((
            Tensor2::from_vec(self.w1.clone())?,
            Tensor1::from_vec(self.b1.clone())?,
            Tensor2::from_vec(self.w2.clone())?,
            Tensor1::from_vec(self.b2.clone())?,
            Tensor2::from_vec(self.w3.clone())?,
            Tensor1::from_vec(self.b3.clone())?,
        ))
    }

    pub fn ensure_finite(&self) -> Result<()> {
        ensure_finite_slice("w1", &self.w1)?;
        ensure_finite_slice("b1", &self.b1)?;
        ensure_finite_slice("w2", &self.w2)?;
        ensure_finite_slice("b2", &self.b2)?;
        ensure_finite_slice("w3", &self.w3)?;
        ensure_finite_slice("b3", &self.b3)?;
        Ok(())
    }
}

pub fn ensure_finite_slice(name: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("{name} contains non-finite value at index {index}: {value}").into());
    }
    Ok(())
}

pub fn batch_with_labels(
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

pub fn batch_images(
    dataset: &data::Mnist,
    start: usize,
) -> Result<Tensor2<f32, BATCH, IMAGE_PIXELS>> {
    let mut images = Vec::with_capacity(BATCH * IMAGE_PIXELS);
    for index in start..start + BATCH {
        images.extend_from_slice(dataset.image(index));
    }
    Ok(Tensor2::from_vec(images)?)
}

pub fn run_logits(
    engine: &Engine,
    model: &Model,
    images: Tensor2<f32, BATCH, IMAGE_PIXELS>,
) -> Result<Tensor2<f32, BATCH, CLASSES>> {
    let (w1, b1, w2, b2, w3, b3) = model.tensors()?;
    Ok(forward_graphs::mnist_logits::run(
        engine, images, w1, b1, w2, b2, w3, b3,
    )?)
}

pub fn evaluate(
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
        let logits = run_logits(engine, model, images)?;

        for (row, label_index) in logits.as_slice().chunks_exact(CLASSES).zip(0..) {
            let predicted = argmax(row);
            let expected = dataset.label(batch * BATCH + label_index);
            correct += usize::from(predicted == expected);
            total += 1;
        }
    }

    Ok(100.0 * correct as f32 / total.max(1) as f32)
}

pub fn argmax(values: &[f32]) -> u8 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index as u8)
        .expect("class logits are non-empty")
}
