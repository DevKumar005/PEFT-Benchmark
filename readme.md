# PEFT-Benchmark

**Empirical comparison of Full Fine-Tuning, LoRA, and 4-bit QLoRA on DistilBERT**

PEFT-Benchmark is a controlled experimental study comparing three fine-tuning strategies for **DistilBERT** on the **IMDB binary sentiment-classification task**:

-   **Full Fine-Tuning**

-   **LoRA** (Low-Rank Adaptation)

-   **QLoRA** (4-bit Quantized LoRA)

The benchmark measures:

-   Trainable parameters

-   Peak GPU memory

-   Training time

-   Test Accuracy

-   Test F1 score

> **Note:** The results are specific to the experimental configuration documented in this repository and should not be interpreted as general performance guarantees.

* * *

## Key Results

| Method | Trainable Params | Trainable % | Training Time (s) | Peak VRAM (MB) | Test Accuracy | Test F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Full Fine-Tuning | 66,955,010 | 100.00% | 105.76 | 1,720.34 | 0.886 | 0.886 |
| LoRA | 739,586 | 1.09% | 61.40 | 995.70 | 0.872 | 0.872 |
| QLoRA (4-bit) | 739,586 | 1.59% | 85.26 | 966.32 | 0.867 | 0.867 |

### Observations

Under this experimental configuration:

-   LoRA and QLoRA train approximately **98.9% fewer parameters** than full fine-tuning.

-   LoRA uses approximately **42% less peak GPU memory** than full fine-tuning.

-   QLoRA uses approximately **44% less peak GPU memory** than full fine-tuning.

-   LoRA has the lowest measured training time.

-   LoRA and QLoRA achieve test performance close to full fine-tuning.

See the technical report for detailed methodology, analysis, figures, and discussion.

* * *

## Repository Structure

```plaintext
PEFT-Benchmark/
├── data.py                 # IMDB data preparation and caching
├── train_full.py           # Full fine-tuning
├── train_lora.py           # LoRA fine-tuning
├── train_qlora.py          # 4-bit QLoRA fine-tuning
├── evaluate.py             # Evaluation utilities
├── utils.py                # Metrics, parameter counting, and tracking
├── visualize.py            # Generate comparison figures
├── requirements.txt        # Python dependencies
├── results/
│   ├── full_ft/
│   ├── lora/
│   ├── qlora/
│   ├── comparison_table.csv
│   └── comparison_table.md
├── figures/
│   ├── fig1_parameter_comparison.png
│   ├── fig2_loss_curves.png
│   ├── fig3_hardware_efficiency.png
│   └── fig4_accuracy_vs_memory.png
├── report/
│   └── PEFT Benchmark Report.pdf
└── notebooks/
    └── peft_benchmark.ipynb
```

* * *

## Installation

### Requirements

-   Python **3.9 or later**

-   CUDA-capable GPU recommended

-   NVIDIA GPU tested: **Tesla T4**

-   CUDA-compatible PyTorch and dependencies listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/DevKumar005/PEFT-Benchmark.git
cd PEFT-Benchmark

python -m venv venv
```

Activate the environment:

**Linux/macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```bat
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

* * *

## Quick Start

### 1\. Prepare the Dataset

```bash
python data.py
```

This prepares the IMDB dataset using `seed=42`, creates the experiment splits, tokenizes the data, and caches the processed dataset under `data_cache/`.

The benchmark currently uses:

-   **Train:** 4,000 samples

-   **Validation:** 1,000 samples

-   **Test:** 1,000 samples

### 2\. Run the Experiments

**Full Fine-Tuning:**

```bash
python train_full.py
```

**LoRA:**

```bash
python train_lora.py
```

**QLoRA:**

```bash
python train_qlora.py
```

Each training script:

-   Trains for 3 epochs

-   Tracks training time

-   Tracks peak GPU memory

-   Evaluates on the test set

-   Saves metrics under `results/<method>/`

-   Saves adapters for LoRA and QLoRA when applicable

### 3\. Generate Figures and Comparison Tables

```bash
python visualize.py
```

* * *

## Experimental Configuration

| Setting | Value |
| --- | --- |
| Base Model | `distilbert-base-uncased` |
| Dataset | IMDB |
| Task | Binary sentiment classification |
| Training Samples | 4,000 |
| Validation Samples | 1,000 |
| Test Samples | 1,000 |
| Max Sequence Length | 256 |
| Epochs | 3 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate (Full FT) | `2e-5` |
| Learning Rate (LoRA/QLoRA) | `5e-4` |
| LoRA Rank (`r`) | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.1 |
| Target Modules | `q_lin`, `v_lin` |
| QLoRA Quantization | 4-bit NF4 + double quantization |
| Compute Precision | FP16 |
| Random Seed | 42 |
| Hardware | NVIDIA Tesla T4 |

> LoRA and QLoRA use the same adapter configuration. QLoRA additionally quantizes the base model to 4-bit precision.

* * *

## Reproducing the Figures

Run:

```bash
python visualize.py
```

This regenerates:

1.  Trainable parameter comparison

2.  Training loss curves

3.  Training time and GPU memory comparison

4.  Accuracy vs. peak GPU memory

Generated figures are stored in `figures/`.

* * *

## Limitations

The current benchmark:

-   Uses only 4,000 training samples from IMDB.

-   Uses a single random seed (`42`).

-   Evaluates only DistilBERT.

-   Does not include extensive hyperparameter tuning.

-   Does not measure inference latency or throughput.

-   Reports results from a single hardware configuration.

These limitations should be considered when interpreting or comparing the results.

* * *

## Citation

If you use this benchmark in your work, please cite:

```bibtex
@misc{peft-benchmark-2026,
  author       = {Dev Kumar},
  title        = {PEFT-Benchmark: Empirical Comparison of Full Fine-Tuning, LoRA, and QLoRA on DistilBERT},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/DevKumar005/PEFT-Benchmark}}
}
```

* * *

## Contributing

Contributions are welcome!

Please read `CONTRIBUTING.md` before submitting an issue or pull request.

Useful contribution areas include:

-   Additional PEFT methods such as DoRA, AdaLoRA, and IA³

-   Additional models and datasets

-   Multi-seed experiments

-   Confidence intervals and statistical analysis

-   Inference latency and throughput measurements

-   YAML/Hydra-based configuration

-   Automated testing

-   Reproducible Docker environments

-   Improved visualizations and documentation

* * *

## License

This project is licensed under the **MIT License**.