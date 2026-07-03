{
  description = "MNIST training example for knok static autograd";

  inputs = {
    knok.url = "github:gmmyung/knok";
  };

  outputs = { knok, ... }: {
    devShells = knok.devShells;
  };
}
