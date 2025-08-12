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
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?logo=plotly&logoColor=white&style=for-the-badge" alt="Matplotlib">
  <a href="outputs/"><img src="https://img.shields.io/badge/JSON%20Metrics-000000?logo=json&logoColor=white&style=for-the-badge" alt="JSON Metrics"></a>
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

**Linear-Regression-from-Scratch** is an educational developer tool demonstrating the fundamentals of **linear regression** via a hands-on implementation now refactored into an object-oriented design. The core model is encapsulated in a `LinearRegressor` class located in `src/models.py`. Both **Gradient Descent** and the **Normal Equation** solver methods are available to fit the model.

Outputs include data and regression plots, loss curves, and performance metrics saved as JSON files in the `outputs/` directory, enabling comprehensive evaluation and visualization without relying on high-level ML libraries.

### Why Linear-Regression-from-Scratch?
- ✨ **Gradient Descent Optimization** — see how parameters learn.
- 📊 **Visualization** — plot data points & fitted line.
- 📈 **Metrics** — loss and R² for performance checks.
- 📝 **Pandas Workflow** — simple data manipulation.
- 🎯 **Extendable** — add multivariate regression, regularization later.
- ✔️ **Normal Equation Sanity Check** — closed-form solution for comparison.
- 📉 **Loss Curve Visualization** — track training progress over epochs.
- 📂 **JSON Metrics Logging** — save evaluation results for further analysis.

---

## 🚀 Getting Started

### 📦 Prerequisites
- **Python 3.10+**
- **pip** (optional: **Conda**)

---

### ⚙ Installation
```bash
# Clone the repository
git clone https://github.com/iPelo/Linear-Regression-from-Scratch

# Enter the project directory
cd Linear-Regression-from-Scratch

# Install dependencies
pip install -r requirements.txt

# Run the main script
python src/main.py
```

---

### 🛠 Usage

Edit the dataset path or parameters directly in `src/main.py` before running the script:

```python
# Example snippet from src/main.py
dataset_path = "data/your_dataset.csv"
```

Then execute:

```bash
python src/main.py
```

This will train the model, generate plots, loss curves, and save metrics to the `outputs/` folder.

---

### 🧪 Testing

```bash
pytest            # or: pytest -q
```
