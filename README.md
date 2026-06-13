# 🌲 Ensemble Methods — From Scratch to Production

> **Why a panel of mediocre predictors often beats a single expert.**

A complete ML learning resource covering **Bagging**, **Random Forest**, and **Boosting** — from mathematical foundations to production-grade implementations.

Each subdirectory contains a standalone README covering one topic end-to-end, plus a Jupyter notebook with from-scratch and library code.

---

## 📚 Topics

| # | Topic | Description | Come Here For… |
|---|-------|-------------|-----------------|
| 01 | [**Bagging**](01-bagging/) | Bootstrap Aggregating — training many base learners in parallel on bootstrapped data and averaging their outputs. Variance reduction, OOB error, independent learners. | Understanding why averaging reduces variance. When and why parallel ensembles work. |
| 02 | [**Random Forest**](02-random-forest/) | Bagging + random feature subspace — decorrelated trees with built-in feature importance. The go-to structured-data baseline. | The ensemble variance formula. When to prefer RF over Boosting. Tree correlation, MDI bias. |
| 03 | [**Boosting**](03-boosting/) | Sequential error correction — AdaBoost, Gradient Boosting, XGBoost. Weak learners → strong learner via functional gradient descent. | Bias reduction. Sample reweighting vs. gradient fitting. Interview prep. |

---

## 🎯 Who Is This For

| Audience | What You Get |
|----------|-------------|
| **First-year B.Tech / beginner** | Intuitive explanations first, math second. Step-by-step walkthroughs with real-world analogies. Code examples with line-by-line explanations. |
| **ML interview candidate** | Structured interview questions at 3 difficulty levels (beginner / intermediate / advanced). Common mistakes that candidates make. Scenario-based problems. |
| **ML practitioner / engineer** | Production-ready code with cross-validation, early stopping, and parallelisation. Hyperparameter tuning tables. Model selection guides. Deployment considerations. |
| **Professor / trainer** | Complete mathematical formulations with symbol tables. References to original papers. Clear bias-variance analysis. Architecturally correct and notationally consistent. |

---

## 🧠 Prerequisites

Before diving into any topic, you should be comfortable with:

- **Decision Trees** — CART, Gini impurity, entropy, tree splits
- **Bias–Variance Tradeoff** — what underfitting and overfitting mean
- **Classification Metrics** — accuracy, AUC-ROC, precision, recall, F1
- **Python + NumPy** — array operations, vectorisation
- **Basic Scikit-Learn API** — `.fit()`, `.predict()`, `.transform()`

---

## 🎯 Learning Path

```
Decision Trees
      │
      ├─── 01-Bagging ─── 02-Random Forest
      │                          │
      │                  (combines bagging +
      │                   random feature subspace)
      │
      └─── 03-Boosting (sequential; different paradigm)
```

**Two fundamentally different approaches to ensembling:**
- **Bagging/RF** → parallel, reduces **variance**, independent learners
- **Boosting** → sequential, reduces **bias**, error-correcting learners

Understanding *why* they differ is the single most important insight for ML interviews.

---

## 📊 Comparison at a Glance

| Property | Bagging | Random Forest | Boosting |
|----------|---------|---------------|----------|
| **Training** | Parallel | Parallel | Sequential |
| **Bias/Variance Effect** | Reduces variance | Reduces variance | Reduces bias |
| **Learner Type** | Any base learner (usually deep trees) | Decorrelated decision trees | Shallow trees (stumps to depth~8) |
| **Key Mechanism** | Bootstrap sampling | Bootstrap + feature subsampling | Error correction (weights or gradients) |
| **Overfitting Risk** | Low | Low | Moderate–high (needs regularization) |
| **Interpretability** | Low | Moderate (MDI, SHAP) | Moderate (SHAP) |
| **Typical Use** | High-variance base learners | Strong baseline for tabular data | Maximum accuracy on clean data |

---

## 📁 Repository Structure

```
ensemble-methods/
├── 01-bagging/
│   ├── bagging.ipynb      # Notebook: from-scratch + sklearn Bagging
│   └── README.md          # Bagging guide
├── 02-random-forest/
│   ├── README.md          # Random Forest guide
├── 03-boosting/
│   ├── boosting.ipynb     # Notebook: AdaBoost + GBM + XGBoost
│   └── README.md          # Boosting guide
└── README.md              # ← You are here
```

---

## 📖 How to Use This Resource

1. **Start with the README** for a topic — it gives the full picture: intuition, math, assumptions, interview prep.
2. **Open the Jupyter notebook** to see working code — from-scratch implementations build understanding; library code shows production practice.
3. **Run the cells** to verify your understanding and experiment with hyperparameters.
4. **Review the interview questions** at the bottom of each README to test yourself.

---

## 📚 General References

| Resource | Why |
|----------|-----|
| [⭐️ StatQuest: Machine Learning](https://www.youtube.com/playlist?list=PLblh5JKOoLUIcdlgu78MnlAT7hGLvjGbd) | Best visual intuition for all ensemble methods |
| [ISLR: Introduction to Statistical Learning](https://www.statlearning.com/) | Clear, math-light textbook covering all three topics |
| [ESL: Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) | The authoritative reference (more mathematical) |
| [Scikit-Learn Ensemble Docs](https://scikit-learn.org/stable/modules/ensemble.html) | Official library documentation |

---

*Prepared for MIT ML curriculum and FAANG interview preparation. All equations are final-form. Code is production-oriented with proper train/test splits.*
