use knok_build::prelude::*;

const BATCH: usize = 64;
const IMAGE_PIXELS: usize = 28 * 28;
const HIDDEN: usize = 256;
const CLASSES: usize = 10;

fn xavier_bound(fan_in: usize, fan_out: usize) -> f32 {
    (6.0_f32 / (fan_in + fan_out) as f32).sqrt()
}

fn mlp_logits(
    images: T2<f32, BATCH, IMAGE_PIXELS>,
    w1: T2<f32, IMAGE_PIXELS, HIDDEN>,
    b1: T1<f32, HIDDEN>,
    w2: T2<f32, HIDDEN, HIDDEN>,
    b2: T1<f32, HIDDEN>,
    w3: T2<f32, HIDDEN, CLASSES>,
    b3: T1<f32, CLASSES>,
) -> T2<f32, BATCH, CLASSES> {
    let hidden1: T2<f32, BATCH, HIDDEN> = relu(matmul(images, w1) + b1);
    let hidden2: T2<f32, BATCH, HIDDEN> = relu(matmul(hidden1, w2) + b2);
    matmul(hidden2, w3) + b3
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn mnist_initial_weights() -> (
    T2<f32, IMAGE_PIXELS, HIDDEN>,
    T2<f32, HIDDEN, HIDDEN>,
    T2<f32, HIDDEN, CLASSES>,
) {
    let w1_bound = xavier_bound(IMAGE_PIXELS, HIDDEN);
    let w2_bound = xavier_bound(HIDDEN, HIDDEN);
    let w3_bound = xavier_bound(HIDDEN, CLASSES);
    (
        uniform_static(0x5eed_0001, -w1_bound, w1_bound),
        uniform_static(0x5eed_0002, -w2_bound, w2_bound),
        uniform_static(0x5eed_0003, -w3_bound, w3_bound),
    )
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn mnist_logits(
    images: T2<f32, BATCH, IMAGE_PIXELS>,
    w1: T2<f32, IMAGE_PIXELS, HIDDEN>,
    b1: T1<f32, HIDDEN>,
    w2: T2<f32, HIDDEN, HIDDEN>,
    b2: T1<f32, HIDDEN>,
    w3: T2<f32, HIDDEN, CLASSES>,
    b3: T1<f32, CLASSES>,
) -> T2<f32, BATCH, CLASSES> {
    mlp_logits(images, w1, b1, w2, b2, w3, b3)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn mnist_loss(
    images: T2<f32, BATCH, IMAGE_PIXELS>,
    labels: T2<i64, BATCH, 1>,
    w1: T2<f32, IMAGE_PIXELS, HIDDEN>,
    b1: T1<f32, HIDDEN>,
    w2: T2<f32, HIDDEN, HIDDEN>,
    b2: T1<f32, HIDDEN>,
    w3: T2<f32, HIDDEN, CLASSES>,
    b3: T1<f32, CLASSES>,
) -> T0<f32> {
    let logits = mlp_logits(detach(images), w1, b1, w2, b2, w3, b3);
    let probabilities: T2<f32, BATCH, CLASSES> = softmax_axis(logits, 1);
    let log_probabilities = log(probabilities);
    let picked: T2<f32, BATCH, 1> = take_along_axis(log_probabilities, labels, 1);
    let negative_log_likelihood: T0<f32> = mean(picked);
    -negative_log_likelihood
}

fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::default().output_file("knok_forward_graphs.rs");
        mnist_logits,
        mnist_initial_weights
    );

    knok_build::grad_graphs_with_options!(
        BuildOptions::default().output_file("knok_grad_graphs.rs");
        mnist_loss
    );
}
