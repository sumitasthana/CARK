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

## The four numbers

Accuracy alone hides what goes wrong when unlearning happens inside continual
learning, so the paper reports four.

- **RA**, retain accuracy. Average over tasks still held at the end. Higher is better.
- **FA**, forget accuracy. Average over forgotten tasks. Should sit at chance, 10%.
- **Spill**, how much a forget request disturbed every *other* task. Lower is better.
- **Relapse**, how much a forgotten task crept back as later tasks were learned. Lower is better.

Spill and relapse are the two failure modes the paper identifies in existing
unlearning methods. Neither shows up in accuracy at the end of a run.

## Citation

Adhikari, Sayanta, et al. "An Unlearning Framework for Continual Learning."
*arXiv*, 2025.
[doi:10.48550/arXiv.2509.17530](https://doi.org/10.48550/arXiv.2509.17530).
