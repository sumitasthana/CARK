# Component-level Attribution of Residual Knowledge after Task Unlearning (CARK)

Working title. An independent reproduction of **UnCLe**, the method introduced
in *An Unlearning Framework for Continual Learning* (Adhikari et al., 2025,
[arXiv:2509.17530](https://arxiv.org/abs/2509.17530)), and a study of which
parts of the model still hold a task after that task has been unlearned.

The wiki has a [concepts FAQ](https://github.com/sumitasthana/CARK/wiki/Concepts-and-processes)
and [run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained).
The structured data and provenance are in [docs/experiments](docs/experiments/README.md).
The [model diagnostics guide](docs/MODEL_DIAGNOSTICS.md) explains how to inspect
weights, gradients, optimizer updates, and diagonal Fisher scores. The
[research diagnostic](docs/RESEARCH_DIAGNOSTIC.md) screens saved forgetting
runs. The proposed recovery measurements still need an experiment runner.
The [notebook guide](notebooks/README.md) separates current experiments, CPU
reports, and historical notebooks.

A note on names, since the two get conflated. *An Unlearning Framework for
Continual Learning* is the paper. **UnCLe** is the method it introduces, and
what this repository implements. There is no paper called UnCLe.

One hypernetwork produces the weights of a target network from a short code,
one code per task. Learning a task trains the hypernetwork to classify it.
Forgetting a task trains the hypernetwork to turn that task code into noise,
which needs no data at all.

## Run it

`uncle-run` runs the Tiny ImageNet request sequences. Its defaults use ResNet50,
200 chunks, and sequence 1. Install the package to add the command.

```bash
pip install -e .                   # or: pip install -r requirements.txt
uncle-run --download               # Tiny ImageNet, sequence 1
uncle-run --sequence 2 --download  # sequence 2
```

For a quick check without a GPU, swap in the small stand-in network:

```bash
uncle-run --backbone cnn --chunks 4 --epochs 1 --limit-requests 2 --max-images 32 --download
python tests/test_uncle.py         # seventeen checks, seconds, no download
python tests/test_tasks.py         # checks the partition, data reader, and notebook syntax
python tests/test_experiments.py   # eighteen checks, a couple of minutes, needs the images
python tests/test_diagnostics.py   # nine checks on the checkpoint diagnostic
python -m unittest tests.test_artifacts     # checks saved-run JSON reading
```

The `--download` flag fetches Tiny ImageNet if it is missing. Full ResNet50
runs need a GPU. The sequences contain 30 requests over 20 tasks, with beta
0.01:

```bash
uncle-run --backbone resnet50 --download
uncle-run --sequence 3 --download
```

The paper's 96.87% retain accuracy and 10.00% forget accuracy are for its
Permuted MNIST setting. They are not targets for the Tiny ImageNet command above.

## What is where

| Path | What it holds |
| --- | --- |
| `uncle/config.py` | Every knob, plus validation of the request list |
| `uncle/data.py` | Permuted MNIST, and the route to Tiny ImageNet |
| `uncle/tinyimagenet.py` | The Tiny ImageNet reader, which never moves a file |
| `uncle/tasks.py` | The saved 20 x 10 class partition and task-local labels |
| `uncle/streams.py` | Those tasks, in the shape the trainer wants |
| `uncle/hypernet.py` | The target network and the network that generates its weights |
| `uncle/trainer.py` | `learn`, `forget`, and the regularizer they share |
| `uncle/metrics.py` | Retain accuracy, forget accuracy, spill, relapse |
| `uncle/experiment.py` | Works through a request sequence, returns records |
| `uncle/experiments.py` | The callable front door: one run, or a sweep of them |
| `uncle/telemetry.py` | What a run cost: time, peak GPU memory, sizes |
| `uncle/baseline.py` | One task, ordinary backprop, no hypernetwork |
| `uncle-run` | Installed command for experiment sequences |
| `main.py` | Older command line kept for compatibility |
| `scripts/` | Setup, dataset exploration, diagnostics, archive export, and compatibility commands |
| `uncle/artifacts.py` | Reads and checks saved run JSON for CPU reports |
| `notebooks/` | [Current workflows and historical notebook guide](notebooks/README.md) |
| `docs/PLAN.md` | Reproduction plan and status |
| `reference/uncle_minimal.py` | The same method in one flat file, for reading |
| `tests/` | `test_uncle.py`, `test_tasks.py`, `test_experiments.py`, `test_diagnostics.py` |

`reference/uncle_minimal.py` is not imported by anything. It exists so the method can be
read top to bottom in one sitting before meeting the package.

## Running it from Python

`main.py` is the older command line and still works. The callable front door,
which is what Colab wants, is this:

```python
from uncle.experiments import run_experiment, run_sequences

result = run_experiment(sequence=1, backbone="resnet50", epochs=5)
print(result["numbers"])     # retain, forget, spill, relapse
print(result["totals"])      # seconds per action, peak GPU memory
```

Any `Config` field passes through as a keyword argument, so a sweep is a loop
over calls. All three of the paper's sequences over three seeds, with a
progress bar and a comparison table at the end:

```python
results = run_sequences(sequences=(1, 2, 3), seeds=(0, 1, 2),
                        backbone="resnet50", epochs=5, output="results")
```

Or from the command line:

```bash
uncle-run --sequence 1 2 3 --seed 0 1 2 --backbone resnet50
```

Each run writes four JSON files named after the sequence, backbone and seed:
the history, the four numbers, the per-request costs, and the environment it
ran in, down to the git commit. The history and the costs are rewritten after
every request, so a long run can be watched and survives a crash.

It also writes a checkpoint after every request, holding the hypernetwork, the
per-task batch-norm statistics, the loop's bookkeeping and the random number
generator's state. Start the same run again and it continues from the last
finished request, making the same draws it would have made had it never
stopped. Pass `checkpoint=False` to skip it, or `resume=False` to start over.

The [notebook guide](notebooks/README.md) lists the current Colab workflows
and separates them from historical notebooks.

## The four numbers

Accuracy alone hides what goes wrong when unlearning happens inside continual
learning, so the paper reports four.

- **RA**, retain accuracy. Average over tasks still held at the end. Higher is better.
- **FA**, forget accuracy. Average over forgotten tasks. Should sit at chance, 10%.
- **Spill**, how much a forget request disturbed every *other* task. Lower is better.
- **Relapse**, how much a forgotten task crept back as later tasks were learned. Lower is better.

Spill and relapse are the two failure modes the paper identifies in existing
unlearning methods. Neither shows up in accuracy at the end of a run.

## What the noise objective actually does

Equation 3 asks the hypernetwork to make a forgotten task's generated weights
match a Gaussian noise sample, averaged over several fresh draws each step.
The paper describes the result as returning the task to a random
initialization.

The averaging does something different. For zero-mean noise `z`, the expected
value of `||x - z||^2` is `||x||^2 + d`, which is smallest at `x = 0`. So
averaging over fresh draws drives the generated weights toward **zero**, not
toward noise. Measured over 3,000 steps on a 2,000-value vector:

| Noise strategy | Final spread of the weights |
| --- | --- |
| Averaged over 10 fresh draws each step (the paper) | 0.04 |
| One fresh draw each step | 0.07 |
| One fixed draw, reused every step | 0.98 |

Only the fixed draw lands on something noise-shaped. That variant is the
"Fixed-noise Alignment" the paper compares against in Appendix E and reports
as worse.

This does not break unlearning. A collapsed generator cannot classify, so
forget accuracy still lands at chance, which is what the metric asks for. But
it is worth knowing that the mechanism is weight collapse rather than
randomization, because the two differ in one way that could matter: a
collapsed network is identical for every forgotten task, while a randomized
one is not.

## Citation

Adhikari, Sayanta, et al. "An Unlearning Framework for Continual Learning."
*arXiv*, 2025.
[doi:10.48550/arXiv.2509.17530](https://doi.org/10.48550/arXiv.2509.17530).
