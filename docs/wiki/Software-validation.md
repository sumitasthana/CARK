# Software validation

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

These are software checks, not measurements of successful unlearning.

## Commit 4c08f54 review

The conversation records 17 passing core checks and eight passing diagnostic
tests, with one CUDA-only test skipped. Additional CPU checks on CNN and
ResNet18 confirmed matching losses and model states with gradient measurement
enabled and disabled, with and without protected tasks. Consuming the global
random stream between runs did not change the tested forgetting result.

## Notebook utilities at e3087f0

The conversation records 11 passing notebook-helper tests, including execution
of all five notebook cells on a small CPU checkpoint with Colab services mocked.
Before publishing, 17 core checks, seven task checks, and 18 experiment checks
passed. One task check was skipped. The earlier diagnostic run passed eight
tests with one CUDA-only skip. The guide checker had no file at its default
path, so it did not execute guide blocks in that validation pass.

These records describe separate validation sessions. Earlier logs report guide
checks using an explicitly supplied guide path; that does not mean the default
guide check ran during the e3087f0 publish. Full-scale GPU memory behavior was
not validated by the CPU tests.

Source: the recorded review and publication results in the project conversation;
earlier validation history remains in [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md).
