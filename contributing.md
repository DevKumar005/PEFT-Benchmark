# Contributing to PEFT-Benchmark

Thank you for your interest in contributing to **PEFT-Benchmark**!

This repository benchmarks **Full Fine-Tuning, LoRA, and QLoRA** on DistilBERT usingthe IMDB sentiment-classification task. Contributions that improve correctness, reproducibility, code quality, documentation, or benchmark coverage are welcome.

## Table of Contents

-   How to Contribute

-   Getting Started

-   Pull Requests

-   Coding Standards

-   Reporting Bugs

-   Suggesting Enhancements

-   Areas for Contribution

-   Project Structure

-   Questions

## How to Contribute

You can contribute through:

-   **Bug fixes** — Training, evaluation, metrics, or visualization issues

-   **Code improvements** — Refactoring, testing, memory tracking, or configuration

-   **New methods** — DoRA, AdaLoRA, IA³, Prefix-Tuning, etc.

-   **New models or datasets** — Additional architectures or classification datasets

-   **Reproducibility** — Multi-seed experiments, environment files, Docker, or CI

-   **Documentation** — README, docstrings, reports, and usage examples

-   **Visualizations** — Improved plots or comparison dashboards

For benchmark changes, document the model, dataset, method, hyperparameters, random seed, hardware, software versions, and evaluation metrics.

## Getting Started

### 1\. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/DevKumar005/PEFT-Benchmark.git
cd PEFT-Benchmark
```

### 2\. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bat
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3\. Prepare the Dataset

```bash
python data.py
```

This prepares the IMDB dataset and creates the local cache used by the project.

Do not commit datasets, caches, checkpoints, or other generated artifacts unlessexplicitly required.

### 4\. Create a Branch

Create a descriptive branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

Examples:

```bash
git checkout -b feature/add-dora
git checkout -b fix/qlora-memory
git checkout -b docs/improve-readme
```

## Pull Requests

Before opening a pull request:

1.  Test all affected functionality.

2.  Run the relevant training, evaluation, or test scripts.

3.  Update documentation when necessary.

4.  Keep the PR focused on one logical change.

5.  Clearly describe what changed and why.

6.  Include reproducibility details for new experiments.

7.  Reference related issues when applicable, for example, `Closes #123`.

### PR Title Format

Use a clear prefix:

```plaintext
[FEATURE] Add DoRA support
[FIX] Correct peak VRAM measurement
[DOCS] Improve README
[REFACTOR] Simplify training configuration
[TEST] Add parameter-count tests
[VIS] Improve benchmark plots
```

### Benchmark Changes

When adding or modifying benchmark results, report:

-   Model

-   Dataset

-   Fine-tuning method

-   Hyperparameters

-   Random seed(s)

-   Hardware

-   Relevant package versions

-   Evaluation metrics

Use the same evaluation methodology as existing experiments unless the PRexplicitly changes the benchmark methodology.

## Coding Standards

-   Follow PEP 8.

-   Use clear and descriptive names.

-   Keep functions focused and avoid unnecessary duplication.

-   Add comments or docstrings for non-obvious logic.

-   Keep training scripts consistent with the existing project structure.

-   Avoid unnecessary dependencies.

-   Add tests for new or modified utility functions where practical.

## Reporting Bugs

Search existing issues before opening a new bug report.

Include:

-   Operating system

-   Python version

-   GPU and available VRAM

-   Relevant package versions

-   Steps to reproduce

-   Expected behavior

-   Actual behavior

-   Full traceback, if applicable

### Bug Report Template

```markdown
## Environment

- OS:
- Python:
- GPU:
- torch:
- transformers:
- peft:
- bitsandbytes:

## Steps to Reproduce

1.
2.
3.

## Expected Behavior

Describe the expected result.

## Actual Behavior

Describe what happened.

## Error Output

Paste the full traceback here, if applicable.

## Additional Context

Add any other relevant information.
```

## Suggesting Enhancements

For feature requests or benchmark improvements, open a GitHub Issue and include:

-   **Summary** — What should be added or changed?

-   **Motivation** — Why is it useful?

-   **Proposed Approach** — How could it be implemented?

-   **Expected Impact** — How would it affect the project or benchmark?

-   **References** — Relevant papers, documentation, or related issues.

### Feature Request Template

```markdown
## Summary

Briefly describe the proposed change.

## Motivation

Explain why it would improve the project.

## Proposed Approach

Describe the proposed implementation.

## Expected Impact

Describe the expected effect on the benchmark or workflow.

## References

- Paper:
- Documentation:
- Related issue/PR:
```

## Areas for Contribution

Some useful areas include:

-   Additional PEFT methods such as DoRA, AdaLoRA, IA³, and Prefix-Tuning

-   Additional models and datasets

-   Multi-seed experiments and confidence intervals

-   Inference latency and throughput measurements

-   YAML/Hydra-based configuration

-   Unit and integration tests

-   Docker-based reproducible environments

-   Improved experiment tracking

-   Better visualizations and comparison dashboards

-   Improved README and documentation

## Project Structure

```plaintext
PEFT-Benchmark/
├── data.py                 # Dataset preparation
├── train_full.py           # Full fine-tuning
├── train_lora.py           # LoRA fine-tuning
├── train_qlora.py          # QLoRA fine-tuning
├── evaluate.py             # Evaluation
├── utils.py                # Shared utilities
├── visualize.py            # Visualization
├── results/                # Experimental results
├── figures/                # Generated figures
├── report/                 # Benchmark report
├── notebooks/              # Jupyter notebooks
└── requirements.txt        # Python dependencies
```

Keep new files consistent with this structure. Avoid committing generated files or large artifacts unless they are intentionally tracked by the project.

## Questions

For general questions, open a GitHub Issue and use the `question` label when available.