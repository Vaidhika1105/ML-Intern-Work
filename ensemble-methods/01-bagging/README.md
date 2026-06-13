# 🎒 Bagging — Bootstrap Aggregating

> **"Diversity beats genius — train many, average them all."**

You will learn how bootstrapping creates diverse training sets and why averaging their predictions systematically reduces model variance without touching bias. You will also understand the math behind ensemble error reduction, the mechanics of Out-of-Bag evaluation, and how to apply bagging in both a from-scratch and production-ready workflow.

---

## 📌 What Is Bagging?

Bagging, short for **Bootstrap AGGregating**, is an ensemble learning technique introduced by Leo Breiman in 1996. The core idea is disarmingly simple: instead of training one model on your full dataset, you train many models — each on a slightly different version of the data — and then combine their outputs. The "different versions" are created by **bootstrap sampling**: repeatedly drawing *n* samples *with replacement* from your original dataset of size *n*. This means some rows appear multiple times in a training bag, while others are left out entirely.

Think of it like asking 100 doctors to diagnose the same patient, but each doctor only sees a random 63% of the patient's medical records (a different 63% for each doctor). No single doctor has the full picture, so their individual diagnoses will vary. But when you tally all 100 opinions, random individual errors cancel out, and the collective verdict is far more reliable than any one doctor's guess. That cancellation of random noise is exactly what bagging exploits mathematically.

The reason bagging works is rooted in statistics: the **variance** of an average of *n* independent values is 1/n times the variance of a single value. In practice, the models aren't truly independent (they're all trained on the same original dataset), so the reduction is less than 1/n. But it's still substantial — especially for models like deep decision trees that are inherently high-variance. Bagging does *not* reduce bias; if your base learner is systematically wrong, averaging wrong predictions just gives you the same wrong answer more confidently.

---

## 📐 Mathematical Formulation

### Core Equation — Variance of an Averaged Ensemble

$$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{(1 - \rho)\,\sigma^2}{T}$$

**Symbol glossary:**

| Symbol | Meaning |
|--------|---------|
| $\text{Var}(\bar{f})$ | Variance of the ensemble's prediction |
| $\rho$ | Average pairwise correlation between any two base models |
| $\sigma^2$ | Variance of a single base model's prediction |
| $T$ | Number of base learners (trees, estimators) in the ensemble |

**What this equation tells us:**

- When $T \to \infty$, the second term $(1-\rho)\sigma^2/T \to 0$, but the first term $\rho\sigma^2$ **never disappears**. The irreducible floor on ensemble variance is $\rho\sigma^2$.
- If $\rho = 0$ (perfectly uncorrelated models), variance drops to $\sigma^2/T$ — a clean T-fold reduction.
- If $\rho = 1$ (perfectly identical models), variance stays at $\sigma^2$ — bagging does nothing.
- **Practical implication:** Adding more estimators helps, but decorrelating them (e.g., via feature subsampling in Random Forest) moves the needle far more once T is large.

### Bootstrap Probability

$$P(\text{sample not selected}) = \left(1 - \frac{1}{n}\right)^n \xrightarrow{n \to \infty} e^{-1} \approx 0.368$$

Each bootstrap bag contains roughly **63.2%** of unique samples. The remaining ~36.8% — never seen by that particular model — form the **Out-of-Bag (OOB)** set, a free held-out evaluation set requiring no additional data split.

---

## ⚙️ How It Works — Step by Step

```
Original Dataset (n samples)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Bootstrap Sampling (with replacement) × T times      │
│                                                       │
│  Bag 1 → [s3, s1, s7, s3, s2, ...]  → Model 1       │
│  Bag 2 → [s5, s2, s9, s1, s5, ...]  → Model 2       │
│  Bag 3 → [s1, s6, s2, s8, s4, ...]  → Model 3       │
│  ...                                  ...             │
│  Bag T → [s4, s7, s1, s3, s9, ...]  → Model T       │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Aggregation                                          │
│  Classification → Majority Vote                       │
│  Regression     → Mean of Predictions                 │
└───────────────────────────────────────────────────────┘
        │
        ▼
   Final Prediction
```

**Step 1 — Bootstrap Sampling**
Draw *n* samples with replacement from your training set. Some rows appear 2–3 times; others are missing. Think of pulling numbered balls from a bag, recording, then putting each back before the next draw. Repeat this T times to create T distinct training bags.

**Step 2 — Train Independent Base Learners**
Fit one base model (typically a fully-grown, unpruned decision tree) on each bag. Because the bags differ, the trees learn slightly different decision boundaries. This step is trivially parallelizable — all T models are independent of one another.

**Step 3 — Out-of-Bag Evaluation**
For each training sample, identify which models never saw it (its OOB models). Collect those models' predictions on that sample. Aggregating OOB predictions across all samples gives an unbiased estimate of generalization error — no separate validation set needed.

**Step 4 — Aggregate at Inference**
Given a new input, pass it through all T models. For **classification**, take the class that receives the most votes. For **regression**, take the arithmetic mean of all T outputs. Analogous to asking your 100 doctors: "What's your diagnosis?" and going with the majority answer.

---

## 🔑 Key Assumptions

| Assumption | What Happens If Violated |
|-----------|--------------------------|
| **Base learner is high-variance** | Bagging a low-variance model (e.g., linear regression) yields marginal gains — variance was already small |
| **Base learner has low bias** | Averaging biased models preserves the bias; the ensemble is still systematically wrong |
| **Training samples are i.i.d.** | Temporal or spatial autocorrelation means bootstrap bags aren't truly independent, reducing diversity gains |
| **Dataset is large enough** | On very small datasets, OOB sets are tiny and unreliable; bootstrap bags heavily overlap, reducing diversity |
| **Labels are not excessively noisy** | Heavy label noise gets memorized across all bags; bagging cannot filter it out |

---

## ✅ When to Use / When Not to Use

| **Use Bagging When...** | **Avoid Bagging When...** |
|------------------------|--------------------------|
| Your base model overfits badly (high variance) | Your model already underfits (high bias, e.g., shallow tree) |
| You can afford to train T models in parallel | Compute or memory budget is very tight |
| You want a free OOB-based generalization estimate | You need the model to be fully interpretable/explainable |
| Your dataset has enough samples for diverse bags | Dataset is tiny (< a few hundred rows) |
| Prediction stability across runs matters | A single well-tuned model already meets performance targets |
| You're building a Random Forest pipeline | You need sequential error correction → use Boosting instead |

---

## 🛠️ Implementation Overview

### From Scratch (NumPy)

The conceptual architecture:
- **Bootstrap loop:** For each of T iterations, sample row indices with replacement using `np.random.choice(n, size=n, replace=True)`.
- **Fit:** Train a base estimator (e.g., a `DecisionTreeClassifier` with `max_depth=None`) on the bootstrapped subset.
- **Store:** Append the fitted model to a list.
- **Predict:** For each new sample, call `.predict()` on every stored model. Aggregate via `scipy.stats.mode` (classification) or `np.mean` (regression).

Key design choice: use `replace=True` in the sampling call — that single flag is what makes it bagging versus pasting.

### Library Implementation (Scikit-Learn)

```python
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vanilla Bagging
bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=200,
    max_samples=1.0,       # 100% of training set per bag
    bootstrap=True,        # sampling WITH replacement
    oob_score=True,        # free OOB generalization estimate
    n_jobs=-1,             # use all CPU cores
    random_state=42
)
bag_clf.fit(X_train, y_train)

print(f"OOB Score  : {bag_clf.oob_score_:.4f}")
print(f"Test Acc   : {accuracy_score(y_test, bag_clf.predict(X_test)):.4f}")

# Random Forest (Bagging + feature subsampling)
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',   # decorrelate trees further
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train, y_train)

print(f"RF OOB     : {rf_clf.oob_score_:.4f}")
print(f"RF Test Acc: {accuracy_score(y_test, rf_clf.predict(X_test)):.4f}")
```

**Conceptual comparison:**

| Dimension | From Scratch | Scikit-Learn |
|-----------|-------------|-------------|
| Transparency | Full control over every step | Abstracted behind a clean API |
| Speed | Slow (pure Python loops) | Optimized C extensions, parallel |
| OOB logic | Must implement manually | `oob_score=True` flag |
| Feature subsampling | Must add manually | `max_features` parameter |
| Best for | Learning the mechanics | Production systems |

---

## 🎯 Top 5 Interview Questions

---

### Q1. Why does bagging reduce variance but not bias?

**Beginner:** If you ask 10 doctors the same question and they all make the same mistake, averaging their answers won't fix the mistake. Bagging works when models make *different* random errors — those cancel out. But systematic errors (bias) don't cancel.

**Intermediate:** Bias is the expected difference between prediction and truth. Averaging doesn't change the expectation: $\mathbb{E}[\bar{f}] = \mathbb{E}[f]$, so bias is unchanged. Variance, however, shrinks: $\text{Var}(\bar{f}) = \rho\sigma^2 + (1-\rho)\sigma^2/T$, where the second term vanishes as $T \to \infty$.

**Advanced:** Bias is a property of the hypothesis class — the set of functions the base learner can represent. Ensembling does not expand this class; it only stabilizes predictions across training sets. Bagging is most effective when the base learner has low bias but high variance (e.g., deep unpruned trees).

**Common mistake:** Claiming bagging reduces both bias and variance. It only addresses variance. If your base models underfit, bagging will still underfit.

---

### Q2. How does Random Forest improve on vanilla bagging, and what does tuning `max_features` do?

**Beginner:** Bagging uses all features — trees end up similar. RF uses random subsets of features at each split, making trees more different from each other.

**Intermediate:** RF adds random feature subsampling, which lowers $\rho$ in the ensemble variance formula. Lower $\rho$ reduces the irreducible floor $\rho\sigma^2$. Decreasing `max_features` reduces correlation (lower $\rho$, lower variance) but makes individual trees weaker (higher bias) — the optimal value balances this.

**Advanced:** With vanilla bagging, $\rho \approx 0.5\text{–}0.9$; with $m \approx \sqrt{p}$, $\rho$ drops to $0.1\text{–}0.4$. The default `max_features="sqrt"` for classification and `"log2"` for regression come from Breiman's empirical analysis showing these maximize the strength-correlation tradeoff.

**Common mistake:** Setting `max_features=1.0` (all features) — that's just bagging, not RF. You lose the decorrelation benefit.

---

### Q3. You have 10M rows and need a Random Forest in production. What are your strategies?

**Beginner:** Use `n_jobs=-1` to parallelize across CPU cores. Subsample your data — 10% of 10M may be plenty.

**Intermediate:** Use `max_samples=0.1` or `0.2` to limit each tree's training set. Profile memory — $T \times \text{depth} \times \text{nodes}$ can reach GBs; cap with `max_depth` or `min_samples_leaf`. Use histograms for split finding (LightGBM/XGBoost `tree_method='hist'`).

**Advanced:** Consider dataset reduction techniques: train on a stratified sample, use RF for feature selection on a pilot run, then train final model on top-$k$ features. Evaluate whether gradient boosting (which requires fewer trees for comparable accuracy) is more memory-efficient.

**Common mistake:** Training the full 10M rows × 1000 trees without subsampling — the memory cost is prohibitive and performance gain beyond ~200 trees on subsampled data is negligible.

---

### Q4. Compare bagging vs. boosting across bias-variance, noise sensitivity, and parallelism.

**Beginner:** Bagging trains models in parallel and averages them. Boosting trains models one after another, each fixing the previous one's mistakes.

**Intermediate:**
- **Bagging:** Reduces variance, fully parallelizable, robust to noisy labels
- **Boosting:** Reduces bias, sequential (cannot parallelize natively), amplifies outliers/noise

**Advanced:** From a bias-variance decomposition perspective, bagging leaves bias unchanged while reducing the variance component to $\rho\sigma^2$. Boosting performs functional gradient descent in hypothesis space, reducing approximation error (bias) at the cost of potentially increasing variance through iterative fitting.

**Common mistake:** Saying "boosting is always better." Boosting can overfit to noisy data; bagging is the safer default when labels are noisy or data is high-variance.

---

### Q5. Your OOB score is significantly higher than your test score. What do you investigate?

**Beginner:** The model might be cheating — maybe information from the test set leaked into training. Or the test data might be different from training data.

**Intermediate:**
- **Data leakage:** Target information or derived features leaked into training
- **Distribution shift:** Train and test from different populations or time periods
- **Class imbalance:** OOB sets may be well-balanced while test set reflects real imbalance
- **Preprocessing leakage:** Scaler/imputer fit on full data before splitting

**Advanced:** OOB is an internal estimate computed during training. When it is significantly higher than test performance, suspect a structural difference between the OOB distribution and the test distribution. Run a two-sample KS test on feature distributions. Use stratified k-fold CV for a more reliable estimate in small datasets or under imbalance.

**Common mistake:** Trusting the OOB score blindly. OOB is an unbiased estimator of *training* generalization, not a substitute for a held-out test set.

---

## Production Considerations

### Failure Modes

| Failure Mode | Why It Happens | Mitigation |
|---|---|---|
| **No variance reduction** | Models are highly correlated (ρ ≈ 1) — happens with non-tree base learners or full feature usage | Use Random Forest's feature subsampling to decorrelate |
| **OOB score is misleading** | OOB works only for bagging with bootstrap sampling | Do not use OOB with `bootstrap=False` |
| **High prediction latency** | T deep trees must all be evaluated | Reduce T, prune depth, use `n_jobs=-1` |
| **Memory blowup** | T trees × depth stores in RAM | Use `max_samples=0.5` to shrink each tree, limit depth |
| **Class imbalance in bags** | Bootstrap sampling may miss minority class entirely | Use stratified bootstrap (`bootstrap=True`) |
| **Slow training on large data** | Each tree sees n samples; 10M × 1000 trees is expensive | Subsample rows per bag (`max_samples`), reduce T |

---

## Key Terms Glossary

| Term | Simple Explanation |
|---|---|
| **Bootstrap sample** | A random sample drawn *with replacement* — same size as original, some rows repeated, some missing |
| **Ensemble** | A collection of models whose predictions are combined (by voting or averaging) |
| **Bagging (Bootstrap Aggregating)** | Training many models on different bootstrap samples, then averaging their predictions |
| **Variance** | How much a model's prediction fluctuates when trained on different datasets |
| **Bias** | How far off a model's predictions are on average — systematic error |
| **OOB (Out-of-Bag)** | The ~37% of rows NOT selected in a given bootstrap sample, used as free validation |
| **OOB score** | The model's accuracy on samples that were out-of-bag — approximates test performance |
| **Base learner** | The individual model type being bagged (e.g., decision tree, SVM) |
| **Correlation (ρ)** | How similarly two models predict — lower ρ means more diverse ensemble |
| **Variance floor (ρσ²)** | The minimum variance an ensemble can achieve even with infinite models |
| **Parallel training** | Training multiple models at the same time on different CPU cores |
| **Bias–variance tradeoff** | Models that are too simple underfit (high bias), too complex overfit (high variance) |

---

## 📊 Quick Reference Table

| Property | Detail |
|----------|--------|
| **Algorithm Type** | Ensemble (parallel, model averaging) |
| **Primary Effect** | Variance reduction |
| **Base Learner** | Any (typically decision trees) |
| **Training Time Complexity** | O(T × n log n) for T trees on n samples |
| **Inference Time Complexity** | O(T × depth) per prediction |
| **Space Complexity** | O(T × tree_size) — grows linearly with T |
| **Key Hyperparameters** | `n_estimators`, `max_samples`, `max_features`, `bootstrap`, `max_depth` |
| **Parallelizable?** | Yes — fully (each model is independent) |
| **Evaluation Metrics** | Accuracy, F1, AUC-ROC (classification); RMSE, R² (regression); OOB Score |
| **Handles Missing Values?** | No (sklearn) — impute first |
| **Interpretability** | Low — ensemble of hundreds of trees |
| **Sklearn Classes** | `BaggingClassifier`, `BaggingRegressor`, `RandomForestClassifier` |

---

## 📚 References & Further Reading

| Resource | Type | Link |
|----------|------|------|
| Breiman (1996) — Original Paper | Research Paper | [https://link.springer.com/article/10.1007/BF00058655](https://link.springer.com/article/10.1007/BF00058655) |
| Scikit-Learn Ensemble Guide | Official Docs | [https://scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html) |
| StatQuest: Random Forests | Video Tutorial | [https://www.youtube.com/watch?v=J4Wdy0Wc_xQ](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) |
| Kaggle: Ensemble Methods Notebook | Hands-on | [https://www.kaggle.com/code/msjgriffiths/ensemble-methods-explore-understand-implement](https://www.kaggle.com/code/msjgriffiths/ensemble-methods-explore-understand-implement) |
| ESL Chapter 8 & 15 (Hastie et al.) | Textbook | [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/) |

---

*Generated as part of the ML Study Series · Topic: Ensemble Methods → Bagging*
*Author perspective: Senior ML Engineer (FAANG) + MIT Educator*
