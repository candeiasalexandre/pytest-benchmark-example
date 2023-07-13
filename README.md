# pytest-benchmark-example

This repo contains an example using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/) to measure the latency of a simple torch model. 

It was created as a companion for my blogpost [Benchmarking Model Latency](https://candeiasalexandre.github.io/posts/benchmarking-model-latency/).

## instructions

1. `make setup-env`
2. `make simple-benchmark` to run the first example discussed in the blogpost
3. `make parametrized-benchmark` to run the parametrized example and generate a .svg histogram

