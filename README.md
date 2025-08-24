<h1 align="center">
  <a href="https://github.com/iPelo/Linear-Regression-from-Scratch">
    LINEAR-REGRESSION-FROM-SCRATCH
  </a>
</h1>

<p align="center">Master Data Insights with Linear Regression – Built from Scratch</p>

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
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge" alt="Pandas">
</p>

---

## 📑 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)

---

## 📜 Overview

**Linear-Regression-from-Scratch** is an educational project implementing **linear regression from scratch** to predict a continuous target \(y\) from feature(s) \(X\).  

The core model is defined in `src/models.py` (class `LinearRegressor`) and trained with **gradient descent**; a **Normal Equation** closed-form solution is also provided for comparison.  

Outputs include:  
- 📊 **Regression Plot** (data + fitted line)  
- 📉 **Loss Curve**  
- ✅ Performance metrics (R², MAE, MSE)  

This demonstrates how to build a regressor without relying on high-level ML libraries like scikit-learn.  

---

## 🧬 Dataset

CSV input expected for supervised regression.  

- **Features:** numeric attributes.  
- **Target:** continuous \(y\).  

> 📌 Place your CSV under the `data/` directory (e.g., `data/your_dataset.csv`) and update the path in `src/main.py`.

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
