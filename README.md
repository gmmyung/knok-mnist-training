# knok MNIST Training

This is a small dogfood example for `knok` static autograd. It trains a
fixed-shape MNIST MLP by compiling two graphs during `cargo build`:

- `mnist_logits`: forward inference for evaluation.
- `mnist_loss_value_and_grad`: scalar loss plus gradients for the model
  parameters.

The training loop is normal Rust. `knok` runs the compiled graphs, while SGD
updates, shuffling, and MNIST IDX parsing stay on the host.

## Run

```sh
nix develop
cargo run --release
```

The first run downloads MNIST into `data/` and compiles the static graphs with
IREE. The default run trains two short epochs over 100 batches each.

Tunable environment variables:

```sh
EPOCHS=5 TRAIN_BATCHES=900 EVAL_BATCHES=100 LR=0.05 cargo run --release
```

Use `MNIST_DIR=/path/to/data` to reuse an existing MNIST IDX cache.

## Notes

`Cargo.toml` depends on `knok` and `knok-build` from GitHub `main`, so this
example follows the merged autograd implementation directly. The `flake.nix`
reuses the upstream `knok` development shell, including the IREE compiler and
MLIR runtime link settings.

The graph uses static `BATCH=64`, `IMAGE_PIXELS=784`, `HIDDEN=128`, and
`CLASSES=10`. Last partial batches are intentionally skipped.
