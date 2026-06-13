# 🌲 Random Forest

> **"Many imperfect trees, one powerful forest."**

You will learn how Random Forest builds an ensemble of decision trees using bootstrapped data and random feature selection to produce a model that generalizes far better than any single tree. You will understand the underlying math, the mechanics of training and prediction, when to use it in production, and how to confidently answer it in a FAANG interview.

---

## 1. What Is Random Forest?

A Random Forest is an ensemble learning algorithm that trains a large collection of decision trees and combines their predictions — by majority vote for classification, or by averaging for regression. The central idea is simple: a crowd of diverse, moderately-skilled forecasters will outperform a single expert who might be overconfident or overfit to noise.

Think of it like a panel of doctors diagnosing a patient. Each doctor reviews a slightly different subset of the patient's test results (bootstrap sample) and focuses on a different set of symptoms (random feature subset). No single doctor has the full picture, but when you aggregate all their opinions, the diagnosis becomes far more reliable than any individual judgment.

The two mechanisms that make this work are **bagging** — training each tree on a random sample drawn *with replacement* from the training data — and **random feature subspace selection** — at every node split, only a random subset of `m` features (not all `p`) is considered. Bagging creates diversity through different data; random features create diversity through different perspectives. Together, they produce trees that are individually imperfect but collectively powerful.

---

## 2. Mathematical Formulation

### 2a. Ensemble Variance Formula

$$\text{Var}\left(\bar{T}\right) = \rho \sigma^2 + \frac{(1 - \rho)\,\sigma^2}{n}$$

| Symbol | Meaning |
|--------|---------|
| $\text{Var}(\bar{T})$ | Variance of the ensemble's averaged prediction |
| $\rho$ | Average pairwise correlation between any two trees |
| $\sigma^2$ | Variance of a single tree's prediction |
| $n$ | Number of trees in the forest |

**What this tells us:** As $n \to \infty$, the second term vanishes — adding more trees always helps, but with diminishing returns. The first term, $\rho\sigma^2$, is the *irreducible floor*: no matter how many trees you add, variance cannot drop below this. This is why **reducing tree correlation $\rho$** — the job of random feature selection — matters more than simply adding more trees.

---

### 2b. Gini Impurity (Split Criterion)

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

| Symbol | Meaning |
|--------|---------|
| $G$ | Gini impurity of a node |
| $K$ | Number of classes |
| $p_k$ | Proportion of samples belonging to class $k$ |

**What this tells us:** A perfectly pure node (one class only) has $G = 0$. A maximally impure node (all classes equally probable) has $G$ close to 1. Each tree greedily picks the feature and threshold that minimizes the weighted Gini of the resulting child nodes.

---

### 2c. Feature Importance (Mean Decrease in Impurity)

$$\text{Importance}(f) = \frac{1}{|T|} \sum_{t \in T} \sum_{\text{node } v \in t,\; \text{split on } f} \Delta G_v \cdot \frac{n_v}{N}$$

| Symbol | Meaning |
|--------|---------|
| $T$ | Set of all trees |
| $\Delta G_v$ | Impurity decrease at node $v$ |
| $n_v$ | Number of samples reaching node $v$ |
| $N$ | Total training samples |

**What this tells us:** Features that consistently produce large impurity reductions across many trees and many nodes rank as most important. However, this metric is **biased toward high-cardinality features** — use permutation importance or SHAP for reliable production diagnostics.

---

## 3. How It Works — Step by Step

**Step 1 — Bootstrap sampling.** For each of the $B$ trees, draw $n$ samples *with replacement* from the training set. On average, each tree sees ~63.2% of unique training examples; the remaining ~36.8% become its Out-of-Bag (OOB) validation set. *Analogy: each doctor studies a slightly different patient file, with some records duplicated and others missing.*

**Step 2 — Random feature selection.** At every node split in the tree, randomly select $m$ features from all $p$ available features. Only these $m$ features are candidates for the best split. For classification, $m \approx \sqrt{p}$; for regression, $m \approx p/3$. *Analogy: each doctor may only order 5 out of 20 possible tests — they must make the best diagnosis they can from their subset.*

**Step 3 — Grow trees fully.** Each tree is grown deep — typically until each leaf contains only 1–5 samples. No pruning. A deep tree has low bias but high variance; that's intentional. The ensemble's averaging will handle the variance.

**Step 4 — Aggregate predictions.** For classification, each tree casts a vote; the class with the most votes wins. For regression, predictions are averaged across all trees. *Analogy: the panel of doctors votes on the diagnosis; majority rules.*

**Step 5 — OOB error estimation.** Each training sample was OOB for roughly 36.8% of trees. For each sample, aggregate only the predictions from trees that did *not* see it during training. This gives a free, approximately unbiased estimate of generalization error — no separate validation set needed.

---

## 4. Key Assumptions

| Assumption | What happens if violated |
|------------|--------------------------|
| Training samples are independently drawn | Temporal/grouped data causes OOB error to be overly optimistic; use time-series CV instead |
| Features are informative enough that random subsets remain useful | If most features are noise, random subsets may miss the few signal features; increase `max_features` |
| Target distribution is stationary | RF cannot extrapolate outside the range of training labels; fails on forecasting tasks |
| Sufficient class representation in bootstrap samples | With extreme imbalance (e.g., 1:100), minority class is underrepresented; use `class_weight='balanced'` |
| Base learners (trees) have manageable variance | If individual trees underfit (too shallow), averaging them will still underfit; RF does not reduce bias |

---

## 5. When to Use / When Not to Use

| ✅ Use Random Forest When… | ❌ Avoid Random Forest When… |
|---------------------------|------------------------------|
| Tabular/structured data with mixed feature types | You need to extrapolate beyond the training range (time series forecasting) |
| You need a strong baseline fast, with minimal preprocessing | Your problem is dominated by high bias (underfitting) — use Gradient Boosting |
| The dataset is noisy and overfitting is a risk | You need a lightweight model for real-time inference at very low latency |
| You want built-in feature importance estimates | You're working with unstructured data (images, text, audio) |
| Class imbalance is moderate and correctable | Memory is severely constrained — RF models can be large |
| You want OOB error without a separate validation set | Full model interpretability is required (use logistic regression or a single tree) |

---

## 6. Implementation Overview

### From Scratch (NumPy)

Building RF from scratch requires:
1. Implementing a CART decision tree with Gini/MSE split logic
2. Adding bootstrap sampling (NumPy `np.random.choice` with `replace=True`)
3. Implementing random feature subsampling at each node split
4. Aggregating predictions via `np.bincount` (classification) or `np.mean` (regression)

The from-scratch version reveals the internals — what "a split" actually computes and why growing trees fully is intentional — but it will be 10–100× slower than sklearn due to lack of Cython optimization.

### Library Implementation (Scikit-Learn)

Scikit-learn's `RandomForestClassifier` and `RandomForestRegressor` expose all key hyperparameters cleanly:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate with key hyperparameters
rf = RandomForestClassifier(
    n_estimators=200,        # Number of trees — more is better up to a point
    max_features="sqrt",     # Random features per split (≈√p for classification)
    max_depth=None,          # Fully grown trees — let them overfit individually
    min_samples_leaf=1,      # Minimum samples per leaf
    oob_score=True,          # Enable free OOB validation
    n_jobs=-1,               # Parallelize across all CPU cores
    random_state=42
)

rf.fit(X_train, y_train)

# OOB score — no separate val set needed
print(f"OOB Score: {rf.oob_score_:.4f}")

# Test set evaluation
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importances (MDI — use with caution on high-cardinality features)
importances = rf.feature_importances_
```

**Key difference:** The scratch version builds intuition about *why* RF works. The sklearn version is what you use in production — optimized, parallelized, and battle-tested.

---

## 7. Top 5 Interview Questions

---

**Q1. Why does Random Forest reduce variance but not bias? When would you prefer Gradient Boosting?**

Ideal answer structure:
- Averaging $n$ predictions reduces variance by factor $1/n$ *only if trees are uncorrelated*
- RF achieves decorrelation via random feature subsets → reduces $\rho$ → reduces variance floor
- RF does not affect the bias of individual trees; if each tree underfits, the average will too
- Gradient Boosting adds trees sequentially to correct residual errors → reduces bias
- **Choose RF** when data is noisy and high variance is the problem
- **Choose GBT** when individual trees underfit and bias is the bottleneck

---

**Q2. Derive the intuition behind the ensemble variance formula and explain the role of tree correlation.**

Ideal answer structure:
- Var(avg of $n$ trees) = $\rho\sigma^2 + (1-\rho)\sigma^2/n$
- As $n \to \infty$, second term vanishes; irreducible floor is $\rho\sigma^2$
- Reducing $\rho$ (via random feature subsets) more impactful than adding trees indefinitely
- Plain bagging uses all features → high $\rho$ → limited variance reduction
- RF's key innovation: random feature subspace lowers $\rho$ dramatically

---

**Q3. Why is MDI feature importance biased, and what alternatives would you use in production?**

Ideal answer structure:
- MDI sums impurity reductions over *all splits on a feature*, across all trees
- Features with more unique values have more possible split points → biased to appear important
- High-cardinality features (zip codes, user IDs) rank artificially high under MDI
- **Permutation Importance**: shuffle feature $j$, measure OOB accuracy drop → model-agnostic, unbiased
- **SHAP (TreeSHAP)**: Shapley value-based, locally accurate, handles feature interactions

---

**Q4. How does OOB error work, and how does it compare to k-fold cross-validation?**

Ideal answer structure:
- Each bootstrap sample leaves out ~36.8% of data → these are OOB samples for that tree
- For each training point, aggregate predictions only from trees that didn't train on it
- OOB error ≈ leave-one-out CV error in expectation — nearly unbiased
- Cheaper than k-fold: no retraining needed, computed during normal training
- Slightly noisier than 5/10-fold CV because each point is evaluated by fewer trees
- Preferred when data is large and compute is a constraint

---

**Q5. A dataset has 500 features, 1M rows, severe class imbalance (1:100), and several high-cardinality categoricals. How do you design the RF pipeline?**

Ideal answer structure:
- **Encoding**: use ordinal or target encoding for high-cardinality features (avoid one-hot explosion)
- **Imbalance**: set `class_weight='balanced'` or use stratified bootstrap sampling
- **Feature importance**: use permutation importance (not MDI) due to high-cardinality bias
- **Compute**: use `n_jobs=-1` for parallelism; consider subsampling rows per tree (`max_samples`)
- **Threshold tuning**: default 0.5 decision threshold is wrong under imbalance; tune via PR-AUC
- **Validation**: use stratified k-fold (not OOB alone) to reliably estimate minority class performance

---

## 8. Quick Reference Table

| Property | Details |
|----------|---------|
| **Algorithm type** | Ensemble (Bagging), non-parametric, supervised |
| **Training time complexity** | $O(B \cdot n \cdot m \cdot \log n)$ — $B$ trees, $n$ samples, $m$ features per split |
| **Inference time complexity** | $O(B \cdot \log n)$ — traverse $B$ trees |
| **Space complexity** | $O(B \cdot \text{tree size})$ — grows linearly with number of trees |
| **Key hyperparameters** | `n_estimators`, `max_features`, `max_depth`, `min_samples_leaf`, `min_samples_split` |
| **Evaluation metrics** | Accuracy, F1-score, ROC-AUC, PR-AUC (imbalance), OOB score, RMSE (regression) |
| **Feature scaling required?** | No — tree splits are invariant to monotonic transformations |
| **Handles missing values?** | Not natively in sklearn; requires imputation or use of LightGBM/XGBoost |
| **Interpretability** | Moderate — feature importance available; full explanation requires SHAP |
| **Parallelizable?** | Yes — trees are independent; use `n_jobs=-1` |

---

## 9. Architecture / Flow Diagram

```
Training Data (n samples, p features)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              BOOTSTRAP + FEATURE SAMPLING               │
│                                                         │
│  Bootstrap₁     Bootstrap₂     …     Bootstrapᴮ        │
│  (n samples,    (n samples,          (n samples,        │
│   with repl.)    with repl.)          with repl.)       │
└────────┬──────────────┬──────────────────┬──────────────┘
         │              │                  │
         ▼              ▼                  ▼
      Tree₁          Tree₂    …         Treeᴮ
   (m features     (m features        (m features
    per split)      per split)         per split)
         │              │                  │
         └──────────────┴──────────────────┘
                         │
                         ▼
              AGGREGATION LAYER
         ┌───────────────────────────┐
         │  Classification: Majority │
         │  vote across B trees      │
         │                           │
         │  Regression: Mean of B    │
         │  predicted values         │
         └───────────────────────────┘
                         │
                         ▼
                  Final Prediction
```

---

## 10. References & Further Reading

| Resource | Type | Link |
|----------|------|-------|
| **Random Forests** — Leo Breiman (2001) | Original Paper | [link.springer.com/article/10.1023/A:1010933404324](https://link.springer.com/article/10.1023/A:1010933404324) |
| **Scikit-learn RF Documentation** | Official Docs | [scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees) |
| **StatQuest: Random Forests** — Josh Starmer | Video Tutorial | [youtube.com/watch?v=J4Wdy0Wc_xQ](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) |
| **SHAP for Tree Models** — Lundberg et al. | Research + Library | [shap.readthedocs.io](https://shap.readthedocs.io) |
| **Kaggle: Intro to ML (Random Forests)** | Hands-on Notebook | [kaggle.com/learn/intro-to-machine-learning](https://www.kaggle.com/learn/intro-to-machine-learning) |

---

*Prepared as part of the MIT ML Study Series. All concepts expressed in original language; no reproduction from any source.*
