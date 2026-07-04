use std::{env, error::Error, path::PathBuf};

use knok::Engine;
use knok_mnist_training::{
    batch_with_labels, data, evaluate, forward_graphs, grad_graphs, sgd, Model, BATCH, HIDDEN,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Config {
    data_dir: PathBuf,
    epochs: usize,
    learning_rate: f32,
    train_batches: usize,
    eval_batches: usize,
}

fn main() -> Result<()> {
    let config = Config::from_env();
    let (train, test) = data::load_or_download(&config.data_dir)?;
    let mut rng = sgd::Rng::new(0x5eed);
    let mut model = Model::new()?;
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
