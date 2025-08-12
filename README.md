<h1 align="center">
  <a href="https://github.com/iPelo/Linear-Regression-from-Scratch">
    LINEAR-REGRESSION-FROM-SCRATCH
  </a>
</h1>

<p align="center">Master Data Insights with Simple, Powerful Regression</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/iPelo/Linear-Regression-from-Scratch?style=for-the-badge" alt="Last Commit">
  <img src="https://img.shields.io/github/languages/top/iPelo/Linear-Regression-from-Scratch?style=for-the-badge" alt="Top Language">
  <img src="https://img.shields.io/github/languages/count/iPelo/Linear-Regression-from-Scratch?style=for-the-badge" alt="Language Count">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Markdown-000000?logo=markdown&logoColor=white&style=for-the-badge" alt="Markdown">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python">
</p>

---

## 📑 Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)

---

## 📜 Overview

**Linear-Regression-from-Scratch** is an educational developer tool demonstrating the fundamentals of **linear regression** via a hands-on **gradient descent** implementation. Train, evaluate, and visualize a regression model without high-level ML libraries to build intuition for core concepts.

### Why Linear-Regression-from-Scratch?
- ✨ **Gradient Descent Optimization** — see how parameters learn.
- 📊 **Visualization** — plot data points & fitted line.
- 📈 **Metrics** — loss and R² for performance checks.
- 📝 **Pandas Workflow** — simple data manipulation.
- 🎯 **Extendable** — add multivariate regression, regularization later.

---

## 🚀 Getting Started

### 📦 Prerequisites
- **Python** (version per your `conda.yml`)
- **Conda** (Miniconda/Anaconda)

---

### ⚙ Installation
```bash
# Clone the repository
git clone https://github.com/iPelo/Linear-Regression-from-Scratch

# Enter the project
cd Linear-Regression-from-Scratch

# Create the environment
conda env create -f conda.yml

# Activate environment
conda activate {venv}


conda activate {venv}
python {entrypoint}    # e.g., python src/train.py


conda activate {venv}
pytest            # or: pytest -q
