<div align="center">

# 🎯 K-Means Clustering — From Theory to Practice
### A Complete Beginner-to-Industry Reference Guide

[![Topic](https://img.shields.io/badge/Topic-Unsupervised%20Learning-blueviolet)]()
[![Algorithm](https://img.shields.io/badge/Algorithm-K--Means-blue)]()
[![Level](https://img.shields.io/badge/Level-Beginner%20to%20Industry-success)]()
[![Status](https://img.shields.io/badge/Docs-Complete-brightgreen)]()

*A single, self-contained reference covering the theory, math, workflow, and interview prep for K-Means Clustering — built for students, trainers, and working ML engineers alike.*

</div>

---

## 📌 About This Document

> This document explains **K-Means Clustering** — the most widely used centroid-based clustering algorithm in machine learning — from first principles up to industry application.

**Why K-Means?** It is often the first clustering algorithm anyone learns, and for good reason: it is simple, fast, scalable, and interpretable. Even when it is not the final model, it is almost always the first baseline to try.

| Audience | What You Get |
|----------|-------------|
| **First-year B.Tech / beginner** | Intuition-first explanations, real-world analogies, step-by-step walkthroughs |
| **ML interview candidate** | Structured interview questions (3 difficulty levels), common mistakes, scenario problems |
| **ML practitioner / engineer** | Production-ready code, hyperparameter tuning guidance, model selection, deployment considerations |
| **Professor / trainer** | Complete mathematical notation with symbol tables, references to original papers, consistent notation |

**Prerequisites:**
- Basic Python + NumPy
- High-school-level math (vectors, Euclidean distance)
- No prior ML experience required

**Estimated reading time:** 30–40 minutes.

---

## 📑 Table of Contents

1. [What is K-Means Clustering?](#-1-what-is-k-means-clustering)
2. [Mathematical Formulation](#-2-mathematical-formulation)
3. [Comprehensive Symbol Table](#-3-comprehensive-symbol-table)
4. [How It Works — Step by Step](#-4-how-it-works--step-by-step)
5. [Choosing K: Elbow Method & Silhouette Score](#-5-choosing-k-elbow-method--silhouette-score)
6. [Key Assumptions](#-6-key-assumptions)
7. [When to Use / When Not to Use](#-7-when-to-use--when-not-to-use)
8. [Implementation Overview](#-8-implementation-overview)
9. [Production Considerations](#-9-production-considerations)
10. [Real-World Applications](#-10-real-world-applications)
11. [Evaluation Metrics](#-11-evaluation-metrics)
12. [Quick Reference Table](#-12-quick-reference-table)
13. [Top 7 Interview Questions](#-13-top-7-interview-questions)
14. [Best Practices & Common Mistakes](#-14-best-practices--common-mistakes)
15. [Failure Cases & Debugging Guide](#-15-failure-cases--debugging-guide)
16. [References & Further Reading](#-16-references--further-reading)

---

## 🧩 1. What is K-Means Clustering?

**K-Means Clustering** is an **unsupervised machine learning algorithm** that partitions a dataset into $K$ distinct, non-overlapping groups (clusters) based on feature similarity. Each data point belongs to the cluster with the nearest mean (centroid).

> **Analogy (clustering):** Imagine your desk is covered in a pile of mixed Lego bricks — red ones, blue ones, green ones — all jumbled together. Without looking at the box labels, you sort them into separate piles by color. That's clustering: finding natural groups without any predefined labels.

> **Analogy (K-Means specifically):** You want to split a field of students into K groups based on who stands near whom. Pick K students as "group leaders" at random. Everyone walks to their nearest leader. Each leader then moves to the center of the students now standing around them. Repeat until nobody switches groups. This is exactly K-Means.

| Term | Meaning |
|------|---------|
| **K** | The number of clusters you want the algorithm to find |
| **Means** | Each cluster is represented by the *average* (mean) of its points — the centroid |
| **Unsupervised** | No predefined labels — the algorithm discovers structure on its own |

### Why is K-Means Needed?

Most real-world data has **no labels**. Businesses and researchers need a way to:

- Discover **natural groupings** in raw, unlabeled data
- **Segment** large populations (customers, patients, documents) into meaningful groups
- **Compress** or summarize complex datasets (e.g., color quantization in images)
- Explore data **before** deciding on further modeling steps

> 💡 **Real Scenario:** An e-commerce company has millions of transactions but no "customer type" labels. K-Means can automatically reveal segments like *budget shoppers*, *frequent buyers*, and *premium customers* — directly powering marketing strategy.

### One-Line Mental Model

> **K-Means repeatedly asks "who is closest to whom?" (assignment) and "where is the true center of each group?" (update) — until both answers stop changing.**

---

## 🧮 2. Mathematical Formulation

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called **inertia**.

### 2.1 Objective Function (WCSS)

$$
\mathcal{J} = \sum_{k=1}^{K} \sum_{i \in C_k} \| \mathbf{x}_i - \boldsymbol{\mu}_k \|^2
$$

Where:
- $\mathcal{J}$ is the total **inertia** (WCSS) — the quantity K-Means minimizes
- $K$ is the number of clusters
- $C_k$ is the set of indices of points assigned to cluster $k$
- $\mathbf{x}_i$ is the $i$-th data point (a $d$-dimensional vector)
- $\boldsymbol{\mu}_k$ is the centroid (mean vector) of cluster $k$
- $\| \mathbf{x}_i - \boldsymbol{\mu}_k \|$ is the Euclidean distance between $\mathbf{x}_i$ and $\boldsymbol{\mu}_k$

> 💡 **Practical insight:** WCSS measures how *tight* each cluster is. Lower WCSS means points are closer to their centroid. But WCSS always decreases as $K$ increases — the elbow method exploits this.

### 2.2 Centroid Update Rule

$$
\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i
$$

Each centroid is simply the **mean position** of all points currently in its cluster — hence the name "K-**Means**."

### 2.3 Assignment Rule

$$
\text{assign } \mathbf{x}_i \text{ to cluster } \arg\min_k \| \mathbf{x}_i - \boldsymbol{\mu}_k \|^2
$$

Each point is assigned to the cluster whose centroid is closest in squared Euclidean distance.

### 2.4 Alternative Formulation (Scatter Matrix)

The WCSS objective can be rewritten using the **within-cluster scatter matrix** $\mathbf{W}$:

$$
\mathbf{W} = \sum_{k=1}^{K} \sum_{i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^{\mathsf{T}}
$$

$$
\mathcal{J} = \operatorname{tr}(\mathbf{W})
$$

This formulation connects K-Means to **PCA** (which diagonalizes the total scatter matrix) and **Linear Discriminant Analysis** (which maximizes the ratio of between-cluster to within-cluster scatter).

### 2.5 Convergence Criterion

The algorithm stops when centroids change by less than a tolerance $\epsilon$ between iterations:

$$
\max_k \| \boldsymbol{\mu}_k^{(t)} - \boldsymbol{\mu}_k^{(t-1)} \| < \epsilon
$$

Or when a maximum number of iterations $T$ is reached.

A more principled (but rarely used) criterion checks the relative change in WCSS:

$$
\frac{|\mathcal{J}^{(t)} - \mathcal{J}^{(t-1)}|}{|\mathcal{J}^{(t-1)}|} < \epsilon
$$

### 2.6 Edge Case: Empty Clusters

If a cluster receives zero points during assignment ($|C_k| = 0$), the update step divides by zero — the mean is undefined. Standard practice: reinitialize the empty centroid to a random data point (as shown in the from-scratch code) or the furthest point from all existing centroids. Scikit-learn handles this automatically via K-Means++ reassignment.

### 2.7 K-Means as Block Coordinate Descent

K-Means can be viewed as **block coordinate descent** on the WCSS objective $\mathcal{J}(\mathbf{Z}, \boldsymbol{\mu})$ where:
- $\mathbf{Z} \in \{0,1\}^{n \times K}$ is a binary assignment matrix (each row sums to 1)
- $\boldsymbol{\mu} \in \mathbb{R}^{K \times d}$ is the centroid matrix

The algorithm alternates between minimizing over $\mathbf{Z}$ given $\boldsymbol{\mu}$ (assignment) and minimizing over $\boldsymbol{\mu}$ given $\mathbf{Z}$ (update). Each step is convex and has a closed-form solution — but the joint problem is non-convex and NP-hard.

> ⚠️ **Note:** Finding the globally optimal clustering is NP-hard. K-Means uses an iterative approximation called **Lloyd's Algorithm**, which converges to a **local optimum** — not necessarily the global one.

---

## 📖 3. Comprehensive Symbol Table

| Symbol | Meaning | Units / Domain | Relevance |
|--------|---------|----------------|-----------|
| $\mathbf{X}$ | Dataset matrix (each row is a data point) | $\mathbb{R}^{n \times d}$ | The input data |
| $n$ | Number of data points | $\mathbb{N}$ | Dataset size |
| $d$ | Number of features (dimensions) | $\mathbb{N}$ | Dimensionality of each point |
| $K$ | Number of clusters (user-specified) | $\mathbb{N}, K \ll n$ | Most important hyperparameter |
| $\mathbf{x}_i$ | The $i$-th data point | $\mathbb{R}^d$ | Row $i$ of $\mathbf{X}$ |
| $\boldsymbol{\mu}_k$ | Centroid (mean vector) of cluster $k$ | $\mathbb{R}^d$ | Cluster representative |
| $C_k$ | Set of point indices in cluster $k$ | $\mathcal{P}(\{1,\dots,n\})$ | Cluster membership |
| $|C_k|$ | Number of points in cluster $k$ | $\mathbb{N}$ | Cluster size |
| $\mathcal{J}$ | WCSS / inertia objective | $\mathbb{R}_{\ge 0}$ | Minimization target |
| $\| \mathbf{x}_i - \boldsymbol{\mu}_k \|$ | Euclidean distance | $\mathbb{R}_{\ge 0}$ | Dissimilarity measure |
| $\epsilon$ | Convergence tolerance | $\mathbb{R}_{> 0}$ | Stopping criterion (default $10^{-4}$) |
| $T$ | Max iterations | $\mathbb{N}$ | Safety bound (default 300) |
| $\mathbf{W}$ | Within-cluster scatter matrix | $\mathbb{R}^{d \times d}$ | Used for evaluation |
| $s(i)$ | Silhouette score for point $i$ | $[-1, 1]$ | Cluster quality metric |

---

## ⚙️ 4. How It Works — Step by Step

### 4.1 The Lloyd's Algorithm

```text
Algorithm: K-Means (Lloyd's Algorithm)

Input  : Dataset X with n points, number of clusters K
Output : K clusters with final centroids

1. Initialize K centroids: μ₁, μ₂, ..., μₖ
   (randomly from data points, or via K-Means++)

2. REPEAT until convergence:
   
   a. ASSIGNMENT STEP:
      For each point xᵢ in X:
          Compute distance to every centroid
          Assign xᵢ to the nearest centroid
   
   b. UPDATE STEP:
      For each cluster k:
          μₖ = mean of all points assigned to cluster k

3. STOP when:
      centroids change < ε (tolerance)
      OR max iterations T reached

4. RETURN final clusters and centroids
```

### 4.2 Initialization: Why K-Means++ Works

Random initialization can place centroids in unlucky spots — two centroids near the same cluster, leaving another cluster with no nearby centroid. **K-Means++** fixes this by spreading out initial centroids:

```text
Algorithm: K-Means++ Initialization

1. Pick the first centroid μ₁ uniformly at random from X
2. For k = 2, 3, ..., K:
   a. For each point xᵢ, compute D(xᵢ) = min distance to already-chosen centroids
   b. Pick xᵢ as the next centroid μₖ with probability ∝ D(xᵢ)²
3. Return {μ₁, ..., μₖ}
```

**Why probability proportional to squared distance?** Points far from existing centroids are more likely to be chosen as new centroids — ensuring centroids are spread across the data space. This gives an $O(\log K)$-approximation to the optimal WCSS (Arthur & Vassilvitskii, 2007), meaning K-Means++ initializations are provably within a logarithmic factor of the best possible.

> 💡 **Intuition:** Think of placing K camp sites to cover a national park. You'd place the first one somewhere central. For the second, you'd go far from the first to cover a new area. For the third, you'd go far from both. This is exactly what K-Means++ does.

### 4.3 Visual Flow

```
Start
  │
  ▼
Choose K ──────────────────────────┐
  │                                │
  ▼                                │
Initialize K centroids             │
  │ (random / K-Means++)           │
  ▼                                │
┌──────────────────────┐           │
│ For each point xᵢ:   │           │
│  assign to nearest μₖ│  ASSIGN   │
└─────────┬────────────┘           │
          ▼                        │
┌──────────────────────┐           │
│ For each cluster k:  │           │
│  μₖ = mean(Cₖ)      │  UPDATE   │
└─────────┬────────────┘           │
          ▼                        │
     ┌────────┐                    │
     │Converged│── No ─────────────┘
     └───┬────┘
         │ Yes
         ▼
    Return clusters
```

### 4.3 What the Algorithm is Actually Doing

The intuition behind the two alternating steps:

1. **Assignment (E-like step):** Given fixed centroids, find the optimal cluster membership for each point — this is a *winner-take-all* competition among centroids.

2. **Update (M-like step):** Given fixed memberships, find the optimal centroid for each cluster — this is just averaging.

Each iteration guarantees that WCSS decreases (or stays the same), ensuring convergence.

### 4.4 Time Complexity

$$
O(n \times K \times I \times d)
$$

| Factor | Meaning | Typical Range |
|--------|---------|---------------|
| $n$ | Number of data points | 1,000 – 10⁷ |
| $K$ | Number of clusters | 2 – 100 |
| $I$ | Number of iterations until convergence | 10 – 300 |
| $d$ | Number of features | 1 – 1,000 |

This **near-linear scaling** in $n$ and $d$ is why K-Means works well on large datasets.

> 💡 **Practical insight:** For most datasets, K-Means converges in 10–50 iterations. If your data is very large ($n > 100,000$), use **Mini-Batch K-Means** to process in batches.

### 4.5 Worked Example (6 Points, 2D, K=2)

See K-Means in action on a tiny dataset:

| Point | $x_1$ | $x_2$ |
|-------|-------|-------|
| A | 1 | 1 |
| B | 1 | 2 |
| C | 2 | 1 |
| D | 8 | 8 |
| E | 8 | 9 |
| F | 9 | 8 |

**Step 0 — Initialization (K=2):** Randomly pick A=(1,1) as $\mu_1$, D=(8,8) as $\mu_2$.

**Step 1 — Assign:**
| Point | Distance to $\mu_1$ | Distance to $\mu_2$ | Assigned |
|-------|---------------------|---------------------|----------|
| A(1,1) | $\sqrt{(1-1)^2+(1-1)^2}=0$ | $\sqrt{(1-8)^2+(1-8)^2}=9.90$ | $\mu_1$ |
| B(1,2) | $\sqrt{(1-1)^2+(2-1)^2}=1$ | $\sqrt{(1-8)^2+(2-8)^2}=9.22$ | $\mu_1$ |
| C(2,1) | $\sqrt{(2-1)^2+(1-1)^2}=1$ | $\sqrt{(2-8)^2+(1-8)^2}=9.22$ | $\mu_1$ |
| D(8,8) | $\sqrt{(8-1)^2+(8-1)^2}=9.90$ | $\sqrt{(8-8)^2+(8-8)^2}=0$ | $\mu_2$ |
| E(8,9) | $\sqrt{(8-1)^2+(9-1)^2}=10.63$ | $\sqrt{(8-8)^2+(9-8)^2}=1$ | $\mu_2$ |
| F(9,8) | $\sqrt{(9-1)^2+(8-1)^2}=10.63$ | $\sqrt{(9-8)^2+(8-8)^2}=1$ | $\mu_2$ |

**Step 1 — Update:** $\mu_1 = \frac{A+B+C}{3} = \left(\frac{1+1+2}{3}, \frac{1+2+1}{3}\right) = (1.33, 1.33)$

$\mu_2 = \frac{D+E+F}{3} = \left(\frac{8+8+9}{3}, \frac{8+9+8}{3}\right) = (8.33, 8.33)$

**Step 2 — Assign (all points stay in same cluster):** Check distances — no changes.

**Step 2 — Update:** Centroids unchanged → **converged in 2 iterations.**

Final clusters: {A,B,C} and {D,E,F}. WCSS = $[0+1+1] + [0+1+1] = 4$.

> 💡 **Takeaway:** K-Means found the obvious two clusters. The "elbow" in WCSS: K=1 gives WCSS≈98, K=2 gives WCSS=4, K=3 gives WCSS ≈ 0.5 — the elbow at K=2 is unmistakable.

---

## 📊 5. Choosing K: Elbow Method & Silhouette Score

### 5.1 The Elbow Method

The Elbow Method picks $K$ by plotting **WCSS vs. $K$** and finding the "bend."

**Steps:**
1. Run K-Means for $K = 1, 2, 3, \dots, K_{\max}$
2. Record WCSS (inertia) for each $K$
3. Plot $K$ (x-axis) vs. WCSS (y-axis)
4. Find the "elbow" — where the curve flattens

```text
WCSS
 │  *
 │    *
 │       *
 │          *     ← Elbow: best K is here
 │             *  *  *  *
 │________________________ K
   1  2  3  4  5  6  7  8
```

**Why it works:** WCSS always decreases as $K$ increases (more centroids = points closer to *some* center). The elbow marks the point of diminishing returns.

> ⚠️ **Limitation:** The elbow isn't always sharp or obvious — it can be subjective. Always pair it with the Silhouette Score.

### 5.2 The Silhouette Score

The Silhouette Score gives a **single number** per point measuring cluster quality:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

| Symbol | Meaning |
|--------|---------|
| $a(i)$ | Mean distance from point $i$ to other points in **its own cluster** (cohesion) |
| $b(i)$ | Mean distance from point $i$ to points in the **nearest other cluster** (separation) |

**Interpretation:**

| Score Range | Meaning |
|-------------|---------|
| $+1$ (close to) | Excellent — well-clustered, far from other clusters |
| $0$ (close to) | Ambiguous — point lies on a cluster boundary |
| $-1$ (close to) | Poor — likely misclassified; wrong cluster |

> 💡 **Practical insight:** Compute the average Silhouette Score across $K$ values and pick the $K$ with the **highest** score. This gives an objective, quantitative answer — unlike the visual Elbow Method.

### 5.3 The Bias-Variance Tradeoff of Choosing K

Choosing K is fundamentally a **bias-variance tradeoff** — the same principle that governs model complexity in supervised learning:

| K | Bias | Variance | WCSS | Interpretation |
|---|------|----------|------|----------------|
| **Too low** (e.g., K=1) | High — all points forced into one cluster | Low — result is stable | High | Underfitting — missing real structure |
| **Just right** | Low — clusters fit the data well | Moderate | Elbow region | Good generalization |
| **Too high** (e.g., K=n) | Zero — each point is its own cluster | High — small changes shuffle assignments | Zero | Overfitting — memorizing noise |

The Elbow Method is **visual bias-variance tradeoff analysis**: WCSS drops fast while you're reducing bias (adding real clusters), then plateaus when you start overfitting (splitting real clusters). The elbow is the crossover point.

### 5.4 Additional Methods

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| **Gap Statistic** | Compares WCSS to a random-data baseline | When elbow is unclear |
| **Davies-Bouldin Index** | Ratio of within-cluster to between-cluster distances | For comparing different $K$ values |
| **Calinski-Harabasz Index** | Ratio of between-cluster variance to within-cluster variance | When clusters are well-separated |
| **Domain Knowledge** | "Low / Medium / High" = $K=3$ | Always — never ignore context |

---

## 📐 6. Key Assumptions

| # | Assumption | Why It Matters | What Happens If Violated | How to Detect / Fix |
|---|-----------|---------------|--------------------------|---------------------|
| 1 | **Clusters are spherical / convex** | K-Means partitions space into **Voronoi cells** — regions where every point belongs to the nearest centroid. These regions are always convex (straight-line boundaries). If your true clusters are crescent-shaped or ring-shaped, K-Means *cannot* recover them | Non-convex clusters (crescents, rings) get merged or split unnaturally | Visualize with PCA; use DBSCAN or Spectral Clustering |
| 2 | **Clusters have similar size** | The mean treats all points equally; large clusters dominate | Small clusters get absorbed into larger ones | Check cluster sizes after fitting; try GMM (unequal mixing weights) |
| 3 | **Clusters have similar density** | Euclidean distance is uniform — dense and sparse clusters are treated the same | Sparse clusters get sliced; dense clusters get merged | Try HDBSCAN for variable-density data |
| 4 | **Features are on comparable scales** | Distance is scale-sensitive | Features with larger numeric ranges dominate the distance metric | Always `StandardScaler()` or `MinMaxScaler()` before fitting |
| 5 | **No extreme outliers** | The mean is not a robust statistic | A single outlier can drag a centroid away from the true cluster | Remove or cap outliers; use K-Medoids instead |
| 6 | **The mean is meaningful** | Centroids are arithmetic means | Categorical or binary data can't be meaningfully averaged | Use K-Modes (categorical) or K-Prototypes (mixed data) |
| 7 | **Euclidean distance is appropriate** | Assignment uses squared Euclidean distance | If direction matters more than magnitude (text data), distance-based clustering fails | Use cosine distance with K-Means variants or Spectral Clustering |

---

## ✅ 7. When to Use / When Not to Use

| ✅ Use K-Means When... | ❌ Avoid K-Means When... | ⚠️ Nuance / Edge Case |
|-----------------------|------------------------|----------------------|
| You need a fast, scalable algorithm on large data ($n > 10^5$) | Clusters are non-spherical (crescents, spirals, rings) | Try DBSCAN or Spectral Clustering for arbitrary shapes |
| You have a good reason to believe $K$ is known (domain knowledge or prior analysis) | You have no idea how many clusters exist | Use Hierarchical Clustering first to explore; then fix $K$ for K-Means |
| You need simple, interpretable results (centroids = cluster representatives) | Data has outliers you cannot remove | Use K-Medoids or DBSCAN (outliers = noise) |
| You need to assign new points to existing clusters online | Every point must belong to a cluster (K-Means forces assignment) | K-Means always assigns every point — use DBSCAN for noise points |
| You want a strong baseline before trying more complex models | Data is categorical or mixed-type | Use K-Modes or K-Prototypes instead |
| You are doing image compression / color quantization | Cluster densities vary dramatically within the data | HDBSCAN handles variable density naturally |

---

## 💻 8. Implementation Overview

### From Scratch (NumPy) — Core Lloyd's Algorithm

```python
import numpy as np

def kmeans_scratch(X, K, max_iter=300, tol=1e-4, random_state=42):
    n, d = X.shape
    rng = np.random.RandomState(random_state)

    # Initialize: randomly pick K points as centroids
    idx = rng.choice(n, K, replace=False)
    mu = X[idx].copy()

    for iteration in range(max_iter):
        # ─── ASSIGNMENT STEP ───
        distances = np.zeros((n, K))
        for k in range(K):
            distances[:, k] = np.linalg.norm(X - mu[k], axis=1)
        labels = np.argmin(distances, axis=1)

        # ─── UPDATE STEP ───
        mu_new = np.zeros_like(mu)
        for k in range(K):
            if np.sum(labels == k) > 0:
                mu_new[k] = X[labels == k].mean(axis=0)
            else:
                mu_new[k] = X[rng.choice(n)]  # reinitialize empty clusters

        # ─── Check convergence ───
        shift = np.linalg.norm(mu_new - mu)
        mu = mu_new
        if shift < tol:
            print(f"Converged at iteration {iteration + 1}")
            break

    return mu, labels
```

### Library Implementation (scikit-learn)

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ALWAYS scale features first
X_scaled = StandardScaler().fit_transform(X)

# ─── Fit K-Means ───
kmeans = KMeans(
    n_clusters=3,            # K — number of clusters
    init="k-means++",        # smart initialization (default)
    n_init=10,               # multiple restarts to avoid local optima
    max_iter=300,
    tol=1e-4,
    random_state=42,
)
kmeans.fit(X_scaled)

# ─── Results ───
labels = kmeans.predict(X_scaled)      # cluster assignments for every point
centroids = kmeans.cluster_centers_     # final centroid positions
inertia = kmeans.inertia_              # WCSS value
n_iters = kmeans.n_iter_               # iterations until convergence

print(f"WCSS (inertia): {inertia:.3f}")
print(f"Converged in {n_iters} iterations")
print(f"Cluster sizes: {np.bincount(labels)}")

# ─── Elbow Method (choosing K) ───
inertias = []
for K in range(1, 11):
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, "bo-")
plt.xlabel("K")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal K")
plt.show()
```

### From Scratch vs. Library

| Aspect | From Scratch (NumPy) | Library (scikit-learn) |
|--------|---------------------|------------------------|
| When to use | Learning, interviews, debugging | Production, large data, pipelines |
| Core loop | ~15 lines; explicit assignment + update | `KMeans.fit()` — optimized Cython |
| Initialization | Random points | `init='k-means++'` (default, smarter) |
| Convergence | Manual tolerance check | Built-in with `tol` and `max_iter` |
| Empty clusters | Must handle manually | Automatically reassigns via K-Means++ |
| Speed | Slow — pure Python loops | Fast — vectorized, compiled C |

---

## 🚀 9. Production Considerations

| Concern | Recommendation |
|---------|---------------|
| **Feature scaling** | Always `StandardScaler()` — K-Means is critically distance-sensitive |
| **Number of clusters** | Use elbow + silhouette + domain knowledge; never trust one metric |
| **Initialization** | Always use `init='k-means++'` — it's the default for good reason |
| **Multiple restarts** | Always set `n_init >= 10` — K-Means converges to local optima |
| **Outliers** | Remove or cap extreme values before fitting |
| **High dimensions** | Reduce to < 50 features (curse of dimensionality makes distances meaningless) |
| **Large datasets ($n > 10^5$)** | Use `MiniBatchKMeans` — processes data in batches, 10–100x faster |
| **Reproducibility** | Set `random_state` for both model and train/test split |
| **New points** | `kmeans.predict(new_data)` works — assigns to nearest centroid |
| **Persistence** | Save with `joblib.dump({'kmeans': kmeans, 'scaler': scaler}, 'kmeans_model.pkl')` |

### Advanced Production Topics

#### Parallelization (Distributed K-Means)

K-Means parallelizes naturally via **Map-Reduce**:

1. **Map (shard):** Each worker holds a subset of data. Broadcast centroids. Each worker assigns its points to nearest centroids and computes local sums per cluster
2. **Reduce:** Aggregate local sums across workers, divide by total counts → new centroids
3. **Broadcast:** Send updated centroids to all workers; repeat

This is how **Spark MLlib** implements K-Means — scaling to billions of points. The communication cost is $O(K \times d)$ per iteration (just the centroids), making it highly efficient for distributed settings.

#### Memory Estimation

| Dataset | Each Point | Distances Array | Total (approx) |
|---------|-----------|-----------------|-----------------|
| $n=10^5, d=50, K=10$ | $10^5 \times 50 \times 8\text{B} = 40\text{MB}$ | $10^5 \times 10 \times 8\text{B} = 8\text{MB}$ | $\approx 50\text{MB}$ |
| $n=10^7, d=100, K=100$ | $10^7 \times 100 \times 8\text{B} = 8\text{GB}$ | $10^7 \times 100 \times 8\text{B} = 8\text{GB}$ | $\approx 16\text{GB}$ |

For the large case, use **Mini-Batch K-Means** or out-of-core processing.

#### Cluster Profiling: Understanding Your Clusters

After fitting, ask: *what makes each cluster different?*

- **Compare feature means** — which features are highest/lowest in each cluster?
- **ANOVA / F-test** — which features vary most across clusters?
- **Decision tree** — train a tree to predict cluster labels; inspect splits (a tree "explains" the clustering boundary)
- **Visualize** — for 2D after PCA, plot centroids with cluster-colored points
- **Profile table** — for each cluster, report: size, mean vector, most distinctive features

```python
# Quick cluster profiling
import pandas as pd

df = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(d)])
df["cluster"] = labels
profile = df.groupby("cluster").agg(["mean", "std", "count"])
# Sort by most distinctive feature
feature_means = df.groupby("cluster").mean()
most_varied = feature_means.std().idxmax()
print(f"Most distinguishing feature: {most_varied}")
print(profile)
```

#### Model Monitoring & Data Drift

In production, new data arrives continuously. K-Means centroids become **stale**:

| Signal | Meaning | Action |
|--------|---------|--------|
| Average distance to nearest centroid increases | Data distribution is shifting | Retrain with new data |
| Some clusters grow/shrink disproportionately | Population mix is changing | Investigate root cause; consider retraining |
| Silhouette score drops significantly | Cluster structure is degrading | Full re-evaluation of K and features |
| New points always assigned to same cluster | Centroids are no longer representative | Reinitialize and retrain |

Set up a monitoring pipeline: log average point-to-centroid distance per batch, alert if it exceeds a threshold (e.g., 2$\sigma$ above historical mean).

### Limitations & Failure Modes

| Scenario | What Happens | Mitigation |
|----------|-------------|------------|
| Non-spherical clusters (crescents, rings) | K-Means splits them unnaturally or merges them | Use DBSCAN or Spectral Clustering |
| Very different cluster sizes | Small clusters get absorbed into larger ones | Use GMM with unequal mixing weights |
| Very different cluster densities | Dense regions are over-split; sparse regions are under-split | Use HDBSCAN |
| Outliers present | Centroid pulled away from true cluster center | Remove outliers; use K-Medoids |
| High-dimensional data ($d > 100$) | **Distance concentration:** as $d$ grows, $\frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$ — all pairwise distances converge to the same value, making nearest-centroid assignments meaningless | PCA or feature selection first |
| Categorical features | The mean is meaningless for categories | Use K-Modes or one-hot encode carefully |

---

## 🌍 10. Real-World Applications

| Domain | Application | Why K-Means |
|--------|------------|-------------|
| 🛍️ **Marketing / E-commerce** | Customer segmentation — identifying budget vs. premium vs. frequent shoppers | Fast to run on millions of transactions; centroids are interpretable as "typical customers" |
| 🖼️ **Image Processing** | Color quantization — reducing an image's color palette from millions to $K$ representative colors | Each centroid = a palette color; $K=16$ or $K=32$ for compression |
| 🏦 **Banking / Finance** | Fraud pattern detection — grouping transactions to identify outliers | K-Means flags points far from all centroids as anomalies |
| 🏥 **Healthcare** | Patient stratification — grouping patients by symptom profiles or genetic markers | Quick clustering to discover subgroups before detailed analysis |
| 📡 **Telecom** | Customer churn analysis — identifying patterns in usage data | Segment users by behavior to target retention campaigns |
| 📚 **Document Analysis** | Grouping news articles or research papers by topic (using TF-IDF features) | Fast baseline before topic modeling (LDA) |
| 🔭 **Astronomy** | Grouping stars/galaxies by physical properties (brightness, redshift, mass) | Naturally spherical clusters in physical parameter space |
| 🔐 **Cybersecurity** | Detecting anomalous network traffic patterns | Points far from any centroid = potential intrusion |

> K-Means is often the **first algorithm tried** in any clustering pipeline — it serves as a fast, interpretable baseline before more complex models.

---

## 📏 11. Evaluation Metrics

| Metric | Formula | Range | Best When | Pitfall |
|--------|---------|-------|-----------|---------|
| **WCSS (Inertia)** | $\sum_{k} \sum_{i \in C_k} \|x_i - \mu_k\|^2$ | $[0, \infty)$ | Comparing same $K$ across runs | Always decreases with $K$ — can't compare across $K$ values directly |
| **Silhouette Score** | $\frac{b(i) - a(i)}{\max(a(i), b(i))}$ | $[-1, 1]$ | Choosing $K$; comparing algorithms | Assumes convex clusters — penalizes DBSCAN unfairly |
| **Davies-Bouldin Index** | $\frac{1}{K} \sum_{k} \max_{j \neq k} \frac{\sigma_k + \sigma_j}{d(\mu_k, \mu_j)}$ | $[0, \infty)$ | Lower is better; compact separated clusters | Sensitive to cluster centroid estimation |
| **Calinski-Harabasz Index** | $\frac{\text{tr}(B)}{\text{tr}(W)} \times \frac{n - K}{K - 1}$ | $[0, \infty)$ | Higher is better; spherical clusters | Tends to favor many small clusters |
| **Gap Statistic** | $E_n^*[\log(\text{WCSS})] - \log(\text{WCSS})$ | $\mathbb{R}$ | Comparing against null (random) baseline | Computationally expensive — requires bootstrapping |

> 💡 **Best practice:** Always use **at least two** metrics together. Silhouette + Elbow is the most common combination.

---

## 📊 12. Quick Reference Table

| Property | K-Means |
|----------|---------|
| **Type** | Centroid-based (partitional) |
| **Needs K upfront** | ✅ Yes (use elbow + silhouette to choose) |
| **Cluster shape** | Spherical / convex (Voronoi partitions) |
| **Output type** | Hard labels (each point belongs to exactly one cluster) |
| **Handles outliers** | ❌ Poorly (mean is not robust; outliers pull centroids) |
| **Scalability** | ✅ High — $O(n \times K \times I \times d)$, near-linear |
| **Deterministic** | ❌ No (depends on initialization) |
| **Feature scaling needed** | ✅ Critical |
| **Soft membership** | ❌ No (hard assignment only) |
| **Generative (can synthesize data)** | ❌ No (discriminative / partitional) |
| **Can assign new points** | ✅ Yes — `kmeans.predict(new_data)` |
| **Convergence guarantee** | Local optimum (monotonic WCSS decrease) |
| **Evaluate without labels** | WCSS, Silhouette, Davies-Bouldin, Gap Statistic |
| **Primary library** | `sklearn.cluster.KMeans` |

---

## 🎤 13. Top 7 Interview Questions

### Q1. How does K-Means work, step by step?

**Beginner:** K-Means alternates between two steps: (1) assign each point to its nearest centroid, and (2) move each centroid to the mean of its assigned points. Repeat until nothing moves much anymore.

**Intermediate:** K-Means minimizes the WCSS objective $\mathcal{J} = \sum_{k} \sum_{i \in C_k} \|x_i - \mu_k\|^2$ via coordinate descent. The assignment step minimizes $\mathcal{J}$ with respect to cluster memberships given fixed centroids (winner-take-all). The update step minimizes $\mathcal{J}$ with respect to centroids given fixed memberships (the mean is the minimizer of squared error). Each step is guaranteed to decrease $\mathcal{J}$, ensuring convergence.

**Advanced:** Lloyd's algorithm is a special case of **block coordinate descent** on a non-convex objective. The objective $\mathcal{J}$ is bi-convex: convex in centroids given assignments and convex in assignments given centroids (though the discrete assignment space makes the joint problem NP-hard). The convergence is to a stationary point, but since the assignment step is discrete, this is a local minimum — not necessarily global. The algorithm can be interpreted as a limit of the EM algorithm for GMM as the covariance goes to 0 ($\Sigma_k = \sigma^2 I, \sigma \to 0$), where the soft responsibilities become hard one-hot assignments.

**Common mistake:** Saying K-Means "minimizes distance." It minimizes **squared** Euclidean distance (WCSS). This is why it emphasizes outliers — squaring gives far points more influence.

---

### Q2. Why does K-Means use Euclidean distance by default?

**Beginner:** Because the centroid update (the mean) is the point that minimizes the sum of squared Euclidean distances — the two steps are mathematically paired.

**Intermediate:** The Euclidean norm is the only $\ell_p$ norm where the mean minimizes the sum of $p$-th powers (it works for $p=2$). For $p=1$ (Manhattan), the minimizer is the median, not the mean. For $p \to \infty$, it's the midrange. So if you use Manhattan distance in the assignment step but average in the update step, you're optimizing two different objectives — the math breaks.

**Advanced:** The squared Euclidean distance is intimately connected to the **Gaussian distribution**: minimizing WCSS is equivalent to maximizing the log-likelihood of a model where each cluster is a spherical Gaussian with identity covariance. Specifically, $\sum_i \|x_i - \mu_{z_i}\|^2 = -2 \log \prod_i \mathcal{N}(x_i \mid \mu_{z_i}, I) + \text{const}$. This is why K-Means is a limiting case of GMM with $\Sigma_k = \epsilon I$ as $\epsilon \to 0$.

**Common mistake:** Thinking you can arbitrarily swap distance metrics in K-Means. If you change the distance metric, the centroid update (mean) is no longer optimal for the new metric — use K-Medoids instead.

---

### Q3. What are the limitations of K-Means?

**Beginner:** K-Means only finds round clusters, you have to pick K in advance, it's sensitive to outliers, and different runs can give different results.

**Intermediate:** The four fundamental limitations are: (1) **spherical assumption** — the Voronoi partition forces convex, isotropic clusters; (2) **equal variance** — implicit assumption that all clusters have the same spread; (3) **hard assignment** — no uncertainty or probability estimates; (4) **local optima** — non-deterministic results depending on initialization.

**Advanced:** These limitations are mathematically structural: (1) the spherical assumption follows from using Euclidean distance (isotropic); (2) equal variance follows from using the same $\| \cdot \|^2$ penalty for all clusters; (3) the 0/1 assignment is the $\sigma \to 0$ limit of the E-step in GMM; (4) local optima follow from the non-convexity of the WCSS objective, which has $O(K^n)$ possible partitions. Each limitation is addressed by a different algorithm: GMM (elliptical, soft, unequal variance), DBSCAN (arbitrary shapes), K-Medoids (robust to outliers).

**Common mistake:** Saying "K-Means doesn't work on high-dimensional data" as a blanket statement. It doesn't work when $d$ is large *relative to* the meaningful structure — the signal-to-noise ratio matters more than $d$ itself. PCA preprocessing often helps.

---

### Q4. How do you choose the right K?

**Beginner:** Use the Elbow Method (plot WCSS vs. K, look for the bend) and the Silhouette Score (pick K with highest score). Also use common sense — if the business expects 3 segments, try K=3.

**Intermediate:** Use a multi-method approach: (1) **Elbow Method** — plot WCSS vs. K, look for diminishing returns; (2) **Silhouette Score** — compute average silhouette across K, pick the maximum; (3) **Gap Statistic** — compare WCSS to a null reference distribution; (4) **Domain knowledge** — the "right" K depends on the granularity of analysis. When methods disagree, Silhouette is usually the most reliable single metric.

**Advanced:** There is no statistically "correct" K for real data — clustering is an ill-posed problem. Different K values reveal structure at different resolutions. The question should be "what K gives me the most useful clustering for my task?" not "what is the true K?" The **stability approach** addresses this: run K-Means with jittered data (bootstrapped samples) and measure how consistent the cluster assignments are across runs. The most stable K is often the best choice. Another approach: **consensus clustering** — run multiple algorithms at different K values and find the co-occurrence matrix.

**Common mistake:** Using the Elbow Method alone and picking the absolute minimum WCSS. The elbow is a *relative* measure — look for the bend, not the lowest point.

---

### Q5. Is K-Means guaranteed to converge? Does it find the global optimum?

**Beginner:** Yes, K-Means always converges (centroids stop moving). But no, it only finds a local optimum — different starting points give different results.

**Intermediate:** K-Means guarantees **monotonic convergence** to a local minimum of the WCSS objective. Each iteration strictly decreases WCSS (or leaves it unchanged), and WCSS is bounded below by 0, so convergence is guaranteed. However, the WCSS landscape is non-convex with many local minima. The number of local minima grows exponentially with $K$, $n$, and $d$. In practice, running 10–20 random initializations and keeping the lowest-WCSS result finds a good (often near-global) solution.

**Advanced:** The convergence proof relies on two facts: (1) the assignment step produces the optimal partition for fixed centroids (by definition — each point goes to the nearest centroid); (2) the update step produces the optimal centroids for fixed partition (the mean minimizes squared error). Since there are only finitely many possible partitions ($K^n$, though astronomically large), and each iteration strictly decreases WCSS, the algorithm must terminate in a finite number of steps — at a partition where no single-point reassignment reduces WCSS. This is a **local minimum** in the sense of the 1-opt neighborhood (no single point can improve by moving). But it may not be the global minimum, which is NP-hard to find.

**Common mistake:** Assuming the K-Means++ initialization guarantees a global optimum. K-Means++ only guarantees that the initialization is $O(\log K)$-competitive with the optimal initialization — it still converges to a local optimum from that starting point.

---

### Q6. How does K-Means++ initialization work and why is it better than random?

**Beginner:** Instead of picking all K starting centroids randomly, K-Means++ picks the first one randomly and then picks each next centroid from points that are *far* from already-chosen ones. This spreads out the starting positions.

**Intermediate:** K-Means++ uses a weighted sampling scheme: (1) pick the first centroid uniformly at random; (2) for each subsequent centroid, compute D(x) = squared distance from x to its nearest existing centroid; (3) sample the next centroid with probability proportional to D(x)². This ensures centroids are spread across the data space, reducing the chance of poor initial placements that lead to bad local optima.

**Advanced:** Arthur & Vassilvitskii (2007) proved that K-Means++ yields an $O(\log K)$-approximation to the optimal WCSS *in expectation*. This means the initialization cost $\mathcal{J}_{\text{init}}$ satisfies $\mathbb{E}[\mathcal{J}_{\text{init}}] \leq 8(\log K + 2)\,\mathcal{J}_{\text{opt}}$. The proof uses a potential function argument and the fact that squared distances satisfy a "D² weighting" that gives a provably good cover of the data space. In practice, K-Means++ rarely produces bad initializations, but it does not eliminate the need for multiple restarts (`n_init`) — it just makes the distribution of outcomes much better.

**Common mistake:** Thinking K-Means++ is a separate clustering algorithm. It's just an *initialization strategy* for K-Means — the rest of the algorithm (assignment + update) is identical.

---

### Q7. You run K-Means on customer data and all customers end up in one cluster. What went wrong?

**Beginner:** Either you forgot to scale the features (so one feature dominates the distance), or you chose K=1 by accident, or the data genuinely has no cluster structure.

**Intermediate:** Diagnose systematically: (1) **Feature scaling** — check if features have very different ranges. A feature with range [0, 1000] dominates one with range [0, 1], making all points appear similar along the small-range features. Fix: `StandardScaler()`. (2) **Hopkins statistic** — test whether the data is uniformly distributed (no clusters). If Hopkins ≈ 0.5, the data is essentially random. (3) **Visualization** — project with PCA; if points form a single blob, K-Means can't find structure that doesn't exist. (4) **K parameter** — verify you're not accidentally using K=1.

**Advanced:** The "all in one cluster" failure can also indicate that the **curse of dimensionality** has collapsed distances. In high dimensions ($d > 50$), all pairwise distances converge to the same value — the ratio $\frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$ as $d \to \infty$ for many distributions. When all points are equally far from all centroids, assignment becomes essentially random, and one centroid wins all points by a tiny margin. Fix: reduce dimensionality first (PCA, autoencoder, or feature selection). Another cause: **dominant feature variance** — if one feature has 100× the variance of others, that feature alone drives assignment, effectively clustering along a single axis rather than in the full space.

**Common mistake:** Assuming "all in one cluster" means the data has no structure. It could mean the *algorithm* is failing (wrong scaling, bad initialization, too few K) even when structure exists. Always check feature ranges before concluding data is unclusterable.

---

## 🛠 14. Best Practices & Common Mistakes

### ✅ Best Practices

- **Always scale features** — `StandardScaler()` before fitting K-Means
- **Use K-Means++ initialization** — it's the default and for good reason
- **Set `n_init >= 10`** — multiple restarts protect against poor local optima
- **Validate K with multiple methods** — Elbow + Silhouette + domain knowledge
- **Visualize clusters** — project with PCA/t-SNE to sanity-check results
- **Profile each cluster** — compute mean, std, and size per cluster to assign meaning
- **Document your K choice** — justify with metrics + business reasoning
- **Check for empty clusters** — if any cluster has 0 points, reduce K or re-run
- **Handle outliers first** — remove or cap before fitting
- **Start simple** — try K-Means first, then add complexity if needed

### ❌ Common Beginner Mistakes

| Mistake | Why It's Harmful | Fix |
|---------|-----------------|-----|
| Not scaling features | Features with larger ranges dominate the distance metric | Always `StandardScaler()` first |
| Using K=1 or arbitrary K without validation | Misses real structure; overfits or underfits | Use elbow + silhouette + domain knowledge |
| Running K-Means once | Different initializations give different results | Set `n_init >= 10` |
| Applying K-Means to categorical data | The mean is meaningless for categories | Use K-Modes or one-hot encode (with caution) |
| Assuming cluster labels are "ground truth" | K-Means always produces K clusters, even if none exist | Use Hopkins statistic to check for clusterability |
| Ignoring outliers | Outliers pull centroids, distorting all clusters | Remove or cap outliers first |
| Using K-Means on non-spherical data | Produces garbage clusters that look plausible in centroid space | Use DBSCAN or GMM instead |
| Treating WCSS as comparable across different datasets | WCSS is scale-dependent and dataset-specific | Use Silhouette or DB Index for cross-dataset comparison |

---

## 🐛 15. Failure Cases & Debugging Guide

| Symptom | Likely Cause | Diagnostic | Fix |
|---------|-------------|------------|-----|
| All points assigned to one cluster | Features not scaled; or data genuinely has no cluster structure | Check feature ranges; run Hopkins statistic | Scale features; verify clustering is appropriate |
| Different results each run | Random initialization — local optima | Compare WCSS across runs | Increase `n_init` to 10–25 |
| One cluster has 0 points | Too many clusters ($K$ too high) | Check cluster sizes with `np.bincount(labels)` | Reduce $K$ or re-run with different seed |
| Very high WCSS | Features not scaled; or $K$ too low | Compare scaled vs. unscaled WCSS | Always scale features; try increasing $K$ |
| Elbow plot has no clear bend | Data may not have well-separated clusters | Check with Silhouette Score; try Gap Statistic | Consider DBSCAN or Hierarchical instead |
| Centroids don't match any real points | Outliers pulling centroids to empty space | Visualize with PCA; check cluster centroids | Remove outliers; cap extreme values |
| Performance is slow ($n > 10^5$) | Standard K-Means processes all points each iteration | Time per iteration | Use `MiniBatchKMeans` for large datasets |
| Silhouette score is negative | Many points assigned to wrong cluster | Compare actual labels vs. PCA visualization | Reduce $K$; try GMM or DBSCAN |
| Clusters look "sliced" in visualization | Non-spherical cluster shapes forced into Voronoi cells | Visualize decision boundaries | Try DBSCAN (arbitrary shapes) |
| Different runs give very different WCSS | Objective landscape has many poor local minima | Compare WCSS across `n_init` runs | Increase `n_init`; use K-Means++ initialization |

---

## 📚 16. References & Further Reading

| Resource | Why Read It |
|----------|-------------|
| 📄 MacQueen (1967) — *Some Methods for Classification and Analysis of Multivariate Observations* | **Original K-Means paper.** Introduces the algorithm and name |
| 📄 Lloyd (1982) — *Least Squares Quantization in PCM* | **Lloyd's Algorithm** — the standard K-Means implementation (from Bell Labs, 1957 but published 1982) |
| 📄 Arthur & Vassilvitskii (2007) — *K-Means++: The Advantages of Careful Seeding* | **K-Means++ initialization** — the standard initialization method used today |
| 📄 Rousseeuw (1987) — *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis* | **Original Silhouette Score paper** — essential for cluster evaluation |
| 📄 Tibshirani, Walther & Hastie (2001) — *Estimating the Number of Clusters in a Data Set via the Gap Statistic* | Gap Statistic — formal method for choosing $K$ |
| 📄 Bishop — *Pattern Recognition and Machine Learning* (Ch. 9) | Best textbook treatment — covers K-Means as a limit of GMM |
| 📄 Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning* (Ch. 14) | Advanced theory on unsupervised learning and clustering |
| 🎓 [Scikit-learn K-Means Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) | Official docs with parameter guidance and examples |
| 🎓 [Scikit-learn: Choosing K (Elbow + Silhouette)](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) | Notebook example of K selection |
| 🎓 [StatQuest: K-Means Clustering (YouTube)](https://www.youtube.com/watch?v=4b5d3muPQmA) | Best beginner-friendly video explanation |

---

## 📁 Module Directory Structure

```
04-kmeans/
├── README.md                          ← this file
├── kmeans.ipynb                       ← full Jupyter notebook with code examples
└── Mall_Customers.csv                 ← sample dataset used by this and other modules
```

---

<div align="center">

**⭐ If this guide helped you, consider starring the repository or sharing it with a peer.**

*Built for learners — from first-year fundamentals to placement-ready depth.*

</div>
