# Contributing to ECG Fairness-Utility Audit

Thank you for your interest in contributing! This project aims to make healthcare AI safer and fairer — every contribution counts. 🎉

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Areas That Need Help](#areas-that-need-help)

---

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the project and the community

---

## 💡 How Can I Contribute?

There are several ways to contribute, regardless of your experience level:

| Type | Examples |
|---|---|
| 🐛 Bug Reports | Notebook errors, incorrect metric calculations, broken visualizations |
| 📁 New Datasets | Extending analysis other datasets |
| 🏗️ Model Improvements | Better architectures, hyperparameter tuning, transfer learning |
| 📝 Documentation | Fixing typos, improving explanations, adding examples |
| 📈 Visualizations | Better plots, interactive dashboards, clearer figures |

---

## 🚀 Getting Started

### 1. Fork & Clone

```bash
# Fork the repo on GitHub first, then:
git clone https://github.com/DevKumar005/PEFT-Benchmark.git
cd PEFT-Benchmark
```

### 2. Set Up Your Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Create a Branch

Always work on a new branch — never directly on `main`:

```bash
git checkout -b feature/your-feature-name
# Examples:
# git checkout -b feature/
# git checkout -b docs/improve-readme
```

---

## 🔄 Pull Request Process

1. **Ensure your changes work** — run the full notebook end-to-end before submitting
2. **Update documentation** — update the README or docstrings if your change affects usage
3. **Keep PRs focused** — one feature or fix per pull request
4. **Write a clear PR description** — explain *what* you changed and *why*
5. **Reference any related issues** — use `Closes #issue_number` in your PR description

### PR Title Format

```
[TYPE] Short description of change
```

Examples:

```
[FEAT] Add equalized odds fairness metric
[FIX] Correct FNR calculation for female subgroup
[DOCS] Improve dataset setup instructions
[REFACTOR] Clean up preprocessing pipeline
```

---

## 🧹 Coding Standards

To keep the codebase consistent:

- **Python style**: Follow [PEP 8](https://pep8.org/)
- **Notebook cells**: Keep cells short and focused; add markdown explanations between code blocks
- **Variable names**: Use descriptive names (`false_negative_rate` not `fnr_val`)
- **Comments**: Explain *why*, not just *what*

---

## 🐛 Reporting Bugs

Before opening a bug report, please check if the issue already exists.

When filing a bug, include:

1. **Environment details** — Python version, OS, package versions (`pip freeze`)
2. **Steps to reproduce** — exact cell or code that fails
3. **Expected behavior** — what should have happened
4. **Actual behavior** — what actually happened, including the full error traceback

Use this template when opening a bug issue:

```
Environment:
  OS:
  Python version:
  TensorFlow version:
  Key package versions (pip freeze):

Steps to Reproduce:
  1.
  2.
  3.

Expected Behavior:

Actual Behavior (include full traceback):

Dataset Version:
```

---

## ✨ Suggesting Enhancements

For feature requests or new ideas, open a GitHub Issue with:

1. **Clear title** — e.g., *"Add threshold optimization as a mitigation technique"*
2. **Motivation** — why would this be useful for fairness in ECG analysis?
3. **Proposed approach** — how would you implement it?
4. **References** — any papers or resources that support the idea

Use this template when opening a feature request:

```
Summary:

Motivation:

Proposed Approach:

References / Related Work:
```

---

## 🆘 Areas That Need Help

If you're looking for a place to start, these are the most impactful open areas:

- [ ] Add **equalized odds** and **demographic parity** as additional fairness metrics
- [ ] Implement **threshold adjustment** as an alternative mitigation technique
- [ ] Add **confidence intervals** to all reported metrics
- [ ] Build an **interactive dashboard** (e.g., using Plotly or Streamlit) for the audit results
- [ ] Improve **model architecture** (e.g., ResNet-1D or Transformer-based ECG model)
- [ ] Add a `requirements-dev.txt` for development dependencies
- [ ] Write **unit tests** for metric calculation functions

---

## 📁 Project Structure Reminder

When contributing, keep files in their correct locations:

---

## ❓ Questions?

If you're unsure about anything, open a GitHub Issue with the label `question` — let's have a chat.

---