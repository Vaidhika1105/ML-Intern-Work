# 🎒 Bagging (Bootstrap Aggregating)

> **Ask ten friends, average their answers — you will usually be right even if each friend is sometimes wrong.**

Bagging is the simplest and most intuitive ensemble method. You train many models independently on different random subsets of the data, then average their predictions. The magic is that averaging cancels out individual mistakes — the crowd, as a whole, is far more reliable than any single member.

---

## 📌 What Is Bagging?

> **Quick refresher — Bias vs. Variance:** *Bias* is the error from a model being too simple (it systematically misses the pattern). *Variance* is the error from a model being too complex (it changes a lot when trained on different data). Bagging fixes variance. Boosting fixes bias. Understanding this difference is the single most important insight for ML interviews.

Bagging stands for **Bootstrap Aggregating**. It has two steps:

1. **Bootstrap sampling** — create many datasets by drawing samples *with replacement* from the original training set.
2. **Aggregating** — train a model on each bootstrap sample, then combine their predictions (average for regression, majority vote for classification).

Imagine you are trying to estimate the weight of a fish. You ask ten fishermen to each take a guess. Some will overestimate, some will underestimate. But if you average all their guesses, the errors cancel out and the average is usually close to the true weight. That is Bagging.

Technically, Bagging works because each model sees a slightly different version of the data. The models end up making different errors, and when you average them, those errors cancel. The more models you add, the more error cancellation you get — up to a point of diminishing returns.

---

## 📐 Mathematical Formulation

> **Notation note:** In ML, a "hat" over a symbol ($\hat{f}$, $\hat{y}$) means "predicted value." For example, $\hat{f}(x)$ is the model's prediction for input $x$, while $f(x)$ is the true underlying function (which we never know). Similarly, $\hat{y}$ is the predicted label and $y$ is the true label.

### The Bootstrap

Given a training set $D = \{(x_1, y_1), \dots, (x_n, y_n)\}$, each bootstrap sample $D_b$ is created by drawing $n$ points **with replacement**:

1. Pick a random index from $\{1, \dots, n\}$, add that $(x_i, y_i)$ to $D_b$.
2. Repeat $n$ times. Some indices are picked multiple times; some are never picked.

The probability that a specific training point is *not* selected in a bootstrap sample of size $n$ is:

$$P(\text{not selected}) = \left(1 - \frac{1}{n}\right)^n \approx \frac{1}{e} \approx 0.368$$

| Symbol | Meaning |
|--------|---------|
| $D$ | Original training set with $n$ samples |
| $D_b$ | $b$-th bootstrap dataset |
| $n$ | Number of training samples |
| $e$ | Euler's number ($\approx 2.718$) |

**What this tells us:** Roughly one-third of the data is left out of each bootstrap sample. These left-out points form the **Out-of-Bag (OOB)** set and act as a free validation set.

---

### Ensemble Prediction

For **regression**, the ensemble prediction is the average of all $B$ models:

$$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(x)$$

For **classification**, the ensemble prediction is the majority vote (or the average of predicted probabilities):

$$\hat{y}_{\text{bag}}(x) = \text{mode}\{\hat{y}_1(x), \dots, \hat{y}_B(x)\}$$

| Symbol | Meaning |
|--------|---------|
| $B$ | Total number of bootstrap samples (and models) |
| $\hat{f}_b(x)$ | Prediction of the $b$-th model on input $x$ |
| $\hat{f}_{\text{bag}}(x)$ | Final bagged prediction |
| $\hat{y}_b(x)$ | Class label predicted by the $b$-th model |

**What this tells us:** Each model votes equally. There is no weighting — every model has the same say in the final prediction. This is fundamentally different from Boosting, where models are weighted by their accuracy.

---

### Variance Reduction

For regression, if each model has variance $\sigma^2$ and models are pairwise correlated with correlation $\rho$, the variance of the average is:

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B \hat{f}_b\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

| Symbol | Meaning |
|--------|---------|
| $\sigma^2$ | Variance of a single model's prediction |
| $\rho$ | Average pairwise correlation between any two models |
| $B$ | Number of models |

**What this tells us:** As $B \to \infty$, the second term vanishes. The variance floor is $\rho\sigma^2$. If models were completely independent ($\rho=0$), variance would drop to $\sigma^2/B$ — almost zero for large $B$. In practice, bootstrap samples are similar, so $\rho > 0$, but the reduction is still substantial.

---

## ⚙️ How It Works — Step by Step

**1. Choose a base learner.** Bagging works with any model — decision trees are the most common because they have high variance (perfect for bagging), but you can bag linear models, SVMs, or neural networks.

**2. Create $B$ bootstrap datasets.** For $b = 1$ to $B$: randomly sample $n$ points *with replacement* from the training set. Some points repeat; some are left out.

> *Analogy:* Give 100 students the same exam, but randomly drop 3 questions from each student's paper. Each student's test is slightly different.

**3. Train one model on each bootstrap set.** Train model $\hat{f}_b$ on $D_b$. Each model trains independently — this can be done in parallel across CPU cores.

**4. Aggregate predictions.** For a new input $x$:
- **Regression:** $\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum \hat{f}_b(x)$
- **Classification:** take the majority vote across all $B$ models

**5. (Optional) Compute OOB error.** For each training point $(x_i, y_i)$, predict it using only the models that did *not* see it during training. Average these predictions to get a free, unbiased estimate of generalization error.

```
                 Training Data
                /    |    \
               /     |     \
              ▼      ▼      ▼
         Bootstrap₁ Bootstrap₂ ... Bootstrapᴮ
              |        |            |
              ▼        ▼            ▼
            Model₁  Model₂  ...  Modelᴮ
               \       |           /
                \      |          /
                 ▼     ▼         ▼
                 AGGREGATION (avg / vote)
                      |
                      ▼
               Final Prediction
```

---

## 🏭 Production Considerations

### Real-World Failure Modes

| Problem | What Happens | Mitigation |
|---------|--------------|------------|
| **Data drift** | The bootstrap samples represent the old distribution; the ensemble becomes stale | Retrain periodically; monitor OOB error over time as a drift signal |
| **Serving latency spikes** | 500 trees × 1ms each = 500ms per prediction | Prune trees, use smaller `n_estimators` for inference (distill ensemble), or switch to a single distilled model |
| **Memory blowup** | Each tree is stored; deep trees on large data can use GBs of RAM | Limit `max_depth`, use `max_leaf_nodes`, or use a Random Forest variant with shallow trees |
| **Imbalanced streaming data** | OOB error becomes unreliable for the minority class | Use stratified bootstrap sampling (maintain class ratio per bootstrap) |
| **Categorical features with many levels** | Bootstrap sampling may miss rare categories entirely | Group rare categories into "other" before bootstrapping |
| **Concept drift in production** | The ensemble was trained on data from 2023; patterns in 2025 are different | Implement sliding-window retraining; weight recent bootstrap samples higher |

---

## 🔑 Key Assumptions

| Assumption | What Happens If Violated |
|------------|--------------------------|
| **Base learners have high variance** | If your base model already has low variance (e.g., linear regression on a simple problem), averaging will not help much |
| **Data points are independent** | If data has temporal structure (time series), bootstrap sampling breaks the ordering and OOB error becomes overly optimistic |
| **Features contain signal** | Bagging cannot create signal from noise. If no feature predicts the target, more trees just memorise noise |
| **Training labels are reliable** | Bagging is robust to moderate label noise, but extreme mislabelling still hurts (each bootstrap contains corrupted examples) |
| **Enough data to sample from** | With very small datasets (< 200 samples), bootstraps become too similar and variance reduction is minimal |

**Important distinction:** Bagging makes no assumption about the loss function being differentiable — it works with any model. This is different from Gradient Boosting, which requires gradients.

---

## ✅ When to Use / When Not to Use

| ✅ Use Bagging When… | ❌ Avoid Bagging When… |
|----------------------|------------------------|
| Your base model has high variance (e.g., deep decision trees) | Your base model already generalises well (low variance is not the problem) |
| You need a quick parallelisable ensemble | You want to reduce bias / underfitting (bagging only reduces variance) |
| You have access to multiple CPU cores | You need a lightweight model for real-time inference at <1ms |
| You want free validation via OOB error | The dataset is very small — bootstraps will be too similar |
| You are working with tabular/structured data | You need full model interpretability (bagging is a black box) |
| Your data is not extremely noisy | You need to handle missing values natively (bagging cannot) |

---

## 🛠️ Implementation Overview

### From Scratch (NumPy)

A bare-bones Bagging implementation requires:

1. **Bootstrap sampling** — `np.random.choice(X.shape[0], size=n, replace=True)` to select row indices
2. **Base learner training** — train a `DecisionTreeRegressor` (or any model) on each bootstrap sample
3. **Prediction aggregation** — `np.mean([model.predict(X) for model in models], axis=0)` for regression; `np.bincount` / `np.argmax` for classification

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class BaggingScratch:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        n = X.shape[0]
        self.models = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, size=n, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            tree = DecisionTreeRegressor(max_depth=None)
            tree.fit(X_boot, y_boot)
            self.models.append(tree)
        return self

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0)
```

### Library Implementation (Scikit-Learn)

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),  # Base learner
    n_estimators=100,                     # Number of bootstrap models
    max_samples=1.0,                      # Use 100% of data per bootstrap
    bootstrap=True,                        # Sample with replacement
    oob_score=True,                        # Track OOB error
    n_jobs=-1,                             # Parallelise
    random_state=42
)

bag.fit(X_train, y_train)
print(f"OOB Score: {bag.oob_score_:.4f}")
y_pred = bag.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### From Scratch — Including OOB Error

The library's `oob_score` parameter abstracts away a lot. Here is how OOB works under the hood:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class BaggingWithOOB:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.models = []
        self.oob_indices = []  # Track which samples were OOB for each tree

    def fit(self, X, y):
        n = X.shape[0]
        self.models = []
        self.oob_indices = []

        for _ in range(self.n_estimators):
            # Bootstrap: pick n indices with replacement
            idx = np.random.choice(n, size=n, replace=True)

            # Record which samples are NOT in this bootstrap (OOB)
            oob_idx = np.setdiff1d(np.arange(n), np.unique(idx))
            self.oob_indices.append(oob_idx)

            # Train on bootstrap sample
            tree = DecisionTreeRegressor(max_depth=None)
            tree.fit(X[idx], y[idx])
            self.models.append(tree)
        return self

    def oob_error(self, X, y):
        """Compute OOB error: for each sample, predict using
        only trees that did NOT train on it."""
        n = X.shape[0]
        oob_preds = [[] for _ in range(n)]

        for tree, oob_idx in zip(self.models, self.oob_indices):
            for i in oob_idx:
                oob_preds[i].append(tree.predict(X[i:i+1])[0])

        # Average predictions for each sample using only OOB trees
        errors = []
        for i in range(n):
            if oob_preds[i]:  # Some samples may never be OOB (rare for large B)
                errors.append((y[i] - np.mean(oob_preds[i])) ** 2)

        return np.sqrt(np.mean(errors))

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0)
```

**Key insight:** Each sample is predicted by roughly $0.368 \times B$ trees. The OOB RMSE is a reliable estimate of test RMSE — no separate validation set needed.

---

### Key Hyperparameters

| Parameter | What It Does | Typical Range |
|-----------|-------------|---------------|
| `n_estimators` | Number of bootstrap models | 50–500 (more is better, diminishing returns) |
| `max_samples` | Fraction of data per bootstrap | 0.5–1.0 |
| `bootstrap` | Sample with replacement? | Usually `True` |
| `bootstrap_features` | Also sample features? | `False` for plain bagging (use Random Forest for this) |
| `oob_score` | Compute out-of-bag estimate | `True` for free validation |

---

## 🎯 Top 5 Interview Questions

---

### Q1. How does Bagging reduce variance, and why doesn't it reduce bias?

**Beginner:** Bagging averages many models. If each model overfits in different ways, the averaging cancels out the overfitting (variance goes down). But if every model systematically misses the same pattern (bias), averaging will not fix that.

**Intermediate:** The ensemble variance formula $\text{Var}(\bar{f}) = \rho\sigma^2 + (1-\rho)\sigma^2/B$ shows that increasing $B$ drives the second term to zero, but the $\rho\sigma^2$ floor remains. Bias is unaffected because each bootstrap sample comes from the same distribution — the expected prediction of any model is the same.

**Advanced:** Bagging approximates the bootstrap expectation: $\hat{f}_{\text{bag}}(x) \approx \mathbb{E}^*[\hat{f}(x)]$, where $\mathbb{E}^*$ denotes expectation over the bootstrap distribution (all possible bootstrap samples of size $n$ drawn with replacement from $D$). The bias of the bagged estimator is:

$$\text{Bias}(\hat{f}_{\text{bag}}) = \mathbb{E}_D[\mathbb{E}^*[\hat{f}(x)]] - f(x) = \mathbb{E}_D[\hat{f}(x)] - f(x) = \text{Bias}(\hat{f})$$

because $\mathbb{E}^*[\hat{f}(x)]$ converges in probability to $\mathbb{E}_D[\hat{f}(x)]$ as $B \to \infty$. The key insight: the inner expectation over bootstrap samples does not change the expected value of the estimator — it only averages out the fluctuations around that value. So bias is entirely inherited from the base learner.

The variance reduction comes from:

$$\text{Var}(\hat{f}_{\text{bag}}) = \mathbb{E}_D[\text{Var}^*(\hat{f})] + \text{Var}_D[\mathbb{E}^*(\hat{f})]$$

The first term (within-bootstrap variance) shrinks as $1/B$; the second term (between-bootstrap variance) captures the $\rho\sigma^2$ floor. This decomposition is why bagging is strictly a variance-reduction technique.

**Common mistake:** Saying "bagging reduces both bias and variance". It only reliably reduces variance.

---

### Q2. What is the Out-of-Bag (OOB) error, and why is it useful?

**Beginner:** Each bootstrap sample leaves out about one-third of the data. You can test each model on the data it did not see, getting a free validation score without a separate validation set.

**Intermediate:** OOB error approximates leave-one-out cross-validation in expectation, at a fraction of the cost. It is computed during training — no extra pass needed. For large datasets, OOB is preferred over $k$-fold CV.

**Advanced:** The OOB estimate is nearly unbiased for the generalization error. However, it is slightly more pessimistic than $k$-fold CV because each point is evaluated by fewer trees ($\approx 0.368B$). With $B \geq 100$, this noise is negligible.

**Common mistake:** Using OOB error as a substitute for a held-out test set. OOB is for *model selection* (tuning hyperparameters); a held-out test set is still needed for *final evaluation*.

---

### Q3. Compare Bagging and Boosting. Which one would you use for noisy data?

**Beginner:** Bagging trains models at the same time; Boosting trains them one after another. Bagging makes mistakes cancel out (lowers variance); Boosting learns from past mistakes (lowers bias). For noisy data, use Bagging — Boosting will amplify the noise.

**Intermediate:** Bagging averages independent models → variance shrinks. Boosting adds dependent models sequentially → bias shrinks but variance can grow. For noisy data, Bagging's averaging naturally smooths out noise. Boosting's error-correcting mechanism will learn the noise, causing overfitting.

**Advanced:** From a bias-variance decomposition perspective, Bagging leaves the bias of the base learner unchanged while reducing variance. Boosting performs functional gradient descent, which reduces approximation error (bias) at the cost of increasing estimation error (variance) with each round. On noisy data, the variance increase from Boosting's sequential dependence outweighs the bias reduction, making Bagging the safer choice.

**Common mistake:** Choosing Boosting for noisy data because "more trees = more accuracy." Boosting memorises noise; Bagging cancels it.

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Bias/Variance | Reduces variance | Reduces bias |
| Noise handling | Robust | Prone to overfitting |
| Overfitting risk | Low | Moderate–high |
| Typical learner | High-variance, deep trees | Low-variance, shallow trees |

---

### Q4. Why is Bagging typically used with decision trees specifically?

**Beginner:** Trees are high-variance models. A small change in the training data can produce a very different tree. Bagging exploits this — different bootstraps produce different trees, and averaging them cancels the variance.

**Intermediate:** Most models have a bias-variance tradeoff that is not easily changed. Decision trees have nearly zero bias (if grown deep) but very high variance — the perfect candidate for variance reduction. Linear models have low variance already, so bagging them gives little benefit.

**Advanced:** Additionally, trees are non-parametric and scale well to large $B$. Bagging 100 decision trees is computationally feasible; bagging 100 SVMs on 100k samples is not.

**Common mistake:** Saying bagging only works with trees. Bagging works with *any* high-variance estimator. Trees are just the most practical choice.

---

### Q5. A dataset has 10,000 samples. You bag 500 decision trees. Roughly how many unique samples does each tree see? How many trees evaluate each sample in the OOB estimate?

**Solution:** Each bootstrap of size $n$ contains approximately $1 - 1/e \approx 63.2\%$ of unique samples. So each tree sees about 6,320 unique samples.

For OOB: each sample is left out of roughly $1/e \approx 36.8\%$ of trees. With $B=500$, that is about $500 \times 0.368 \approx 184$ trees per sample. The OOB prediction for each sample is the average of those ~184 trees.

**Key insight:** The fraction $1/e$ is independent of $n$ — it is the same for 100 samples or 1 million. This is why OOB error is such a convenient diagnostic.

---

## 📖 Key Terms Glossary (For Beginners)

| Term | Plain English Meaning |
|------|----------------------|
| **Bias** | Error from being too simple — the model consistently misses the pattern |
| **Variance** | Error from being too complex — the model changes a lot with different training data |
| **Correlation ($\rho$)** | How similarly two models make mistakes. $\rho = 1$ means identical mistakes; $\rho = 0$ means unrelated mistakes |
| **Bootstrap** | A dataset created by randomly picking from the original data, allowing repeats |
| **With Replacement** | After picking a sample, put it back so it can be picked again |
| **Out-of-Bag (OOB)** | Samples that were NOT selected in a given bootstrap — a free test set for that model |
| **Base Learner** | The underlying model being ensembled (e.g., a decision tree, linear model, SVM) |
| **Parallel Ensemble** | All models train independently — they can be built at the same time on different CPU cores |
| **Sequential Ensemble** | Each model depends on the previous one — must be built one after another |

---

## 📊 Quick Reference Table

| Property | Details |
|----------|---------|
| **Algorithm Type** | Parallel ensemble (Bootstrap + Aggregation) |
| **Base Learner** | Any model (typically decision trees) |
| **Training Time Complexity** | $O(B \cdot T(n, d))$ — $T$ depends on the base learner |
| **Inference Time Complexity** | $O(B \cdot T_{\text{pred}})$ |
| **Space Complexity** | $O(B \cdot \text{model size})$ — grows linearly with $B$ |
| **Primary Bias/Variance Effect** | Reduces variance; bias unchanged |
| **Key Hyperparameters** | `n_estimators`, `max_samples`, `bootstrap` |
| **Evaluation Metrics** | Accuracy, F1, AUC-ROC, MSE, RMSE, OOB score |
| **Feature Scaling Required?** | No (depends on base learner) |
| **Handles Missing Values?** | No (depends on base learner) |
| **Parallelisable?** | Yes — trees are fully independent |
| **Overfitting Risk** | Low |
| **Interpretability** | Low — individual predictions are averaged |

---

## 📚 References & Further Reading

| Resource | Description |
|----------|-------------|
| 📄 [Breiman, 1996 — Bagging Predictors](https://link.springer.com/article/10.1007/BF00058655) | The original Bagging paper — short, elegant, required reading |
| 🎓 [StatQuest: Bagging (YouTube)](https://www.youtube.com/watch?v=2Mg8QD0_1jI) | Best visual intuition for bootstrapping and aggregation |
| 📖 [ISLR Ch. 8 — Tree-Based Methods](https://www.statlearning.com/) | Textbook treatment with R/Python labs |
| 📖 [ESL Ch. 8 — Model Inference and Averaging](https://hastie.su.domains/ElemStatLearn/) | Deeper mathematical treatment |
| 🐍 [Scikit-Learn: Bagging Docs](https://scikit-learn.org/stable/modules/ensemble.html#bagging) | Official API reference |

---

*Prepared as part of the MIT ML curriculum and FAANG interview preparation. All equations are final-form. Code examples use production-ready patterns with parallelisation and OOB validation.*
