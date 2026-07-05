# 02 - Linear Regression

Notes, lab material, and code from the **Linear Regression** module of
Andrew Ng's *Machine Learning Specialization* (Coursera).

This module covers the first real ML model in the course: predicting a
continuous number (like house price) from input features using a straight
line — the foundation for everything that comes after it (cost function,
gradient descent, multiple features, and eventually neural networks).

## 📁 Contents

| File | Description |
|---|---|
| `LINEAR REGRESSION WITH ONE VARIABLE(AND SO...)` | Handwritten notes covering univariate linear regression, model notation (x, y, w, b, m), and terminology (training set, features, targets). |
| `Linear regression(MODEL REPRESENTATION LAB).pdf` | The official Coursera lab notebook (as PDF) — model representation, plotting data, and building the `f(x) = wx + b` function. |
| `lab_code.py` | Clean, runnable Python version of the lab code — imports, data setup, model function, plotting, the "fit the line" challenge, and a price prediction. |
| `linear_regression_lab_EXPLAINED.md` | Line-by-line breakdown of every line in `lab_code.py` — what each line does and why it's there. Written for reviewing the lab without needing to re-watch the videos. |
| `ML INTRODUCTION (SUPERVISE...)` | Notes from the intro section — supervised vs. unsupervised learning, what ML is. |
| `README.md` | This file. |

## 🧠 Key concepts covered

- **Univariate linear regression** — predicting `y` from a single input feature `x`
- **Model function**: `f(x) = wx + b`, where `w` = weight/slope, `b` = bias/intercept
- **Notation**: `(x⁽ⁱ⁾, y⁽ⁱ⁾)` for the i-th training example, `m` = number of training examples
- **Regression vs. classification**: regression predicts continuous numbers (infinite outputs), classification predicts categories (finite outputs)
- Using NumPy (`np.array`, `.shape`, `np.zeros`) to store and process training data
- Using Matplotlib (`scatter`, `plot`) to visualize data and model predictions
- Manually fitting `w` and `b` to a small dataset before learning the automated approach (gradient descent, covered in the next module)

## ▶️ Running the lab code

```bash
pip install numpy matplotlib
python lab_code.py
```

> Note: `plt.style.use('./deeplearning.mplstyle')` in the code refers to
> Coursera's custom style file. If you don't have it, comment out that line
> — it only affects plot colors/fonts, not the ML logic.

## 🔗 Course reference

Machine Learning Specialization — Andrew Ng (Coursera / DeepLearning.AI)
Course 1: Supervised Machine Learning — Regression and Classification
