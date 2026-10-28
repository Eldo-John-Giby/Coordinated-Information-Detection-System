# Project Facts: Verified Technical & Empirical Single Source of Truth

**Project Title:** Detecting Coordinated Information Network Under LLM-Paraphrased Content Using Semantic Similarity and Cross-Encoder  
**Authors / Student Investigators:** Undergraduate Project Team  
**Verification Date:** October 2026  
**Status:** Completed & Empirically Validated over 5 Random Seeds (`[42, 43, 44, 45, 46]`)

---

## 1. Problem Formulation & Research Question

### 1.1 Problem Statement
Coordinated Inauthentic Behavior (CIB) and state-sponsored Information Operations (IOs) on social networks manipulate public consensus by deploying synchronized clusters of accounts that amplify political narratives. Traditional CIB detection frameworks rely heavily on lexical duplicate matching (string matching, hashing, TF-IDF cosine) and static account profile metadata (follower-to-following ratios, creation dates). Modern threat actors leverage Large Language Models (LLMs) to automatically generate high-entropy semantic paraphrases, eliminating verbatim phrase repetition while preserving core propaganda narratives.

### 1.2 Core Research Question
*How severely does LLM-based semantic paraphrasing degrade lexical and dense bi-encoder coordination signals in real-world Information Operations, and can a retrieve-then-rerank cross-encoder architecture integrated into a multi-signal graph learning framework recover community-level and node-level detection under adaptive adversarial evasion and cross-campaign transfer?*

### 1.3 Hypotheses Under Test
1. **$H_1$ (Lexical Collapse):** LLM paraphrasing causes severe failure in lexical coordination detection (TF-IDF), reducing pairwise coordination scores and community-level group recovery to near zero.
2. **$H_2$ (Cross-Encoder Recovery):** Cross-encoder reranking over candidate pairs retrieved via a low-floor bi-encoder index recovers high semantic similarity scores for paraphrased post pairs.
3. **$H_3$ (Multi-Signal Redundancy):** Multi-modal graph fusion across 6 behavioral channels (semantic, temporal, URL, hashtag, mention, repost) maintains community detection even when an adaptive adversary actively scrubs hashtags and injects $\pm 6$-hour temporal jitter.
4. **$H_4$ (Relational Transfer vs. Tabular Memorization):** High tabular baseline performance (Random Forest $\text{F1} \approx 0.995$) represents profile artifact memorization that degrades under cross-campaign domain transfer and feature perturbation, whereas relational graph topology generalizes robustly across campaigns without retraining.

---

## 2. Dataset Provenance & Preprocessing Specifications

### 2.1 Dataset Provenance
* **Source Archive:** Zenodo open-access research repository (Record 13912659: *Labeled Datasets for Research on Information Operations*, Seckin et al., AAAI ICWSM 2025; Cima et al., IEEE Access 2024).
* **Campaign 1 (Honduras - In-Domain Primary):**
  - Total Posts Loaded: 4,704 posts (4,696 valid after dropping 8 malformed records).
  - Total Unique Accounts: 1,208 accounts.
  - Class Distribution: 440 IO accounts (36.4%), 768 control accounts (63.6%).
  - Temporal Range: September 11, 2019 to January 8, 2020.
  - Repost Volume: 3,800 posts (80.9%).
  - Text Length: Mean 97.4 characters, median 86, max 280.
  - Dominant Languages: **90.8% Spanish (`es`)**, 7.2% unidentified (`und`), 1.0% English (`en`), <1% Catalan, Italian, Portuguese.
* **Campaign 2 (UAE - Held-Out Transfer Target):**
  - Total Posts Loaded: 4,293 posts.
  - Total Unique Accounts: 1,176 accounts.
  - Class Distribution: 395 IO accounts (33.6%), 781 control accounts (66.4%).
  - Text Length: Mean 153.5 characters, median 144, max 307.
  - Dominant Languages: **82.6% Arabic (`ar`)**, 6.1% English (`en`), 4.3% unidentified (`und`), 1.3% French (`fr`), 1.1% Persian (`fa`).

### 2.2 Data Integrity & Splitting Protocol
* **Splitting Strategy:** Canonical 70/15/15 account-level partition (70% train, 15% validation, 15% test).
* **Leakage Prevention:** Partitions are assigned strictly at the *account identifier* level before edge construction. No posts from the same account cross split boundaries.
* **ID Canonicalization:** String-based ID canonicalization eliminates float-precision truncation and prevents synthetic self-loops in the coordination graph.
* **Degradation Tracking:** All pipeline fallbacks or non-fatal missing values log to `src.utils.degradations.DegradationTracker`.

---

## 3. System Architecture & Technical Implementation

### 3.1 Retrieve-then-Rerank NLP Pipeline
1. **Bi-Encoder Stage:** Text is mapped to 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
2. **High-Recall Candidate Retrieval:** Indexed with FAISS (`IndexFlatIP`) with a low retrieval floor threshold ($\text{cosine} \ge 0.30$) to capture semantic candidate dyads.
3. **Account Dyad Aggregation:** Top-3 post pairs per account dyad are selected for reranking.
4. **Cross-Encoder Precision Reranker:** Candidate dyads are scored using `cross-encoder/ms-marco-MiniLM-L-6-v2` / `cross-encoder/nli-deberta-v3-base`.
5. **Budget Cap Optimization:** Reranking is capped at $N=20,000$ pairs per run, achieving 91.2% edge recovery and 88.2% group recall while bounding CPU/GPU inference latency.

### 3.2 Multi-Signal Coordination Feature Extraction
For every candidate account dyad $(u, v)$, six distinct coordination channels are extracted:
1. **Semantic Similarity ($S_{\text{sem}}$):** Cross-encoder max-pooling score over candidate post pairs.
2. **Temporal Synchronicity ($S_{\text{temp}}$):** Exponential decay proximity score $S_{\text{temp}} = \exp(-\Delta t / \tau)$ with time decay $\tau = 3600$s (1 hour) and 5-minute burst window filtering ($t_{\text{burst}} \le 300$s).
3. **Shared URL Score ($S_{\text{url}}$):** Jaccard co-occurrence index over domain links.
4. **Shared Hashtag Score ($S_{\text{ht}}$):** Jaccard co-occurrence index over deployed hashtags.
5. **Shared User Mention Score ($S_{\text{ment}}$):** Jaccard co-occurrence index over mentioned handles.
6. **Shared Repost Score ($S_{\text{rep}}$):** Jaccard co-occurrence index over retweet/quote targets.

**Composite Edge Coordination Formulation:**
$$W(u, v) = \sum_{k \in K} w_k \cdot S_k(u, v), \quad \sum w_k = 1.0$$
where weights are calibrated across semantic (0.25), temporal (0.20), hashtag (0.20), repost (0.15), URL (0.10), and mention (0.10). Edges with $W(u, v) \ge 0.30$ are retained in the graph, with per-node degree pruning capped at 50 edges.

### 3.3 Graph Neural Networks & Classifiers
* **PyTorch Geometric Graph:** Undirected graph $G = (V, E, W)$ where nodes $V$ contain 7 behavioral profile features (post count, log follower count, following count, repost ratio, avg hashtag count, avg URL count, avg mention count) and content PCA embeddings.
* **Graph Convolutional Network (GCN):** 2-layer `GCNConv` with BatchNorm1d, ReLU, Dropout (0.5), and LogSoftmax.
* **GraphSAGE:** Multi-layer `SAGEConv` with mean neighbor aggregation (`aggregator="mean"`), executed via full-batch training on the complete graph.
* **Baselines:** Random Forest (100 trees, max depth 20, class-balanced), L2-regularized Logistic Regression, Adaptive Threshold baseline, Unsupervised Louvain Community density baseline.

---

## 4. Empirical Evaluation Findings & Metric Matrix

All metrics are evaluated across **5 random seeds (`[42, 43, 44, 45, 46]`)** with **1,000 bootstrap iterations** for 95% Confidence Intervals.

### 4.1 In-Campaign Model Comparison (Honduras Canonical Split)
| Model | Accuracy (95% CI) | Precision (95% CI) | Recall (95% CI) | F1 Score (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **GraphSAGE** | **0.971 [0.968, 0.976]** | **0.930 [0.919, 0.940]** | **0.997 [0.991, 1.000]** | **0.962 [0.958, 0.968]** | **0.988 [0.983, 0.993]** | **0.936 [0.910, 0.966]** |
| **GCN** | 0.856 [0.840, 0.873] | 0.721 [0.700, 0.743] | 0.985 [0.961, 1.000] | 0.832 [0.815, 0.851] | 0.932 [0.923, 0.940] | 0.786 [0.767, 0.804] |
| **Random Forest** | 0.864 [0.839, 0.885] | 0.801 [0.760, 0.840] | 0.833 [0.797, 0.870] | 0.816 [0.781, 0.842] | 0.924 [0.903, 0.943] | 0.850 [0.812, 0.889] |
| **Logistic Regression** | 0.722 [0.693, 0.753] | 0.598 [0.566, 0.632] | 0.712 [0.673, 0.764] | 0.650 [0.615, 0.690] | 0.834 [0.802, 0.862] | 0.711 [0.663, 0.755] |

### 4.2 Semantic Component Benchmark Against Topical Hard Negatives (Phase 1)
Evaluated on pairs selected strictly by non-semantic signals vs. keyword-matched non-coordinated controls:
* **TF-IDF Cosine:** ROC-AUC = **0.322 [0.288, 0.355]** (1:1) and **0.372 [0.347, 0.398]** (1:5); Recall@1% FPR = 0.010.
* **MiniLM Bi-Encoder:** ROC-AUC = **0.490 [0.457, 0.523]** (1:1) and **0.505 [0.481, 0.531]** (1:5); Recall@1% FPR = 0.012.
* **Paraphrase Cross-Encoder:** ROC-AUC = **0.525 [0.494, 0.556]** (1:1) and **0.564 [0.543, 0.593]** (1:5); Recall@1% FPR = **0.025 [0.009, 0.044]**.

### 4.3 Paraphrase Injection Matrix (Phase 2, $K=30$ Groups)
* **TF-IDF:** Group recall collapses from **0.867** ($p=0.0$) down to **0.039 [0.000, 0.078]** ($p=1.0$, heavy rewrite).
* **Bi-Encoder Only:** Degrades from **0.993** ($p=0.0$) to **0.844 [0.793, 0.893]** ($p=1.0$, heavy rewrite).
* **Bi-Encoder + Cross-Encoder Rerank:** Retains **0.947 [0.913, 0.980]** ($p=1.0$, heavy) and **0.987** ($p=1.0$, light).
* **Full Multi-Signal Fusion:** Retains **0.806 to 0.987** group recall across all conditions.

### 4.4 Adaptive Adversary Evasion (Phase 3)
* **$A_1$ (Paraphrase Only):** Semantic drop = **0.205 [0.170, 0.237]**; Fused Group Recall = **0.987 [0.960, 1.000]**.
* **$A_2$ ($A_1$ + Hashtag Scrub):** Semantic drop = **0.199 [0.168, 0.231]**; Hashtag signal = 0.0; Fused Group Recall = **0.987 [0.960, 1.000]**.
* **$A_3$ ($A_2$ + $\pm 6$h Jitter):** Temporal drop = **0.350 [0.350, 0.350]**; Semantic drop = **0.197 [0.167, 0.228]**; Fused Group Recall = **0.980 [0.953, 1.000]**.

### 4.5 Cross-Campaign Zero-Shot Transfer & Randomization Attacks (Phase 4)
* **Honduras $\to$ UAE (Clean):**
  - Tabular RF: ROC-AUC = **0.884 [0.871, 0.898]**, PR-AUC = 0.785, F1 = 0.812.
  - Coordination Graph Topology Only: ROC-AUC = **0.910 [0.899, 0.922]**, PR-AUC = 0.834, F1 = 0.844.
  - GNN Full: ROC-AUC = **0.898 [0.885, 0.910]**, PR-AUC = 0.823, F1 = 0.830.
* **Honduras $\to$ UAE (Feature Randomization Attack on Follower/Following Counts):**
  - Tabular RF: ROC-AUC plummets to **0.627 [0.608, 0.647]** ($\Delta = -0.257$).
  - Coordination Graph Topology Only: Invariant at **0.910 [0.899, 0.922]** ($\Delta = 0.000$).
  - GNN Full: Resilient at **0.884 [0.871, 0.897]** ($\Delta = -0.014$).

---

## 5. Nuanced Findings & Honest Limitations

1. **TF-IDF Negative Correlation on Political Controls:** Under non-semantic positive pairing, TF-IDF achieved ROC-AUC = 0.322, worse than chance. Coordinated IO accounts often deployed succinct hashtags or short colloquial text, whereas organic political commentary frequently reused the same trending political nouns.
2. **Tabular Profile Overfitting:** Gini permutation importance proved that Random Forest in-domain accuracy is driven by `repost_ratio` (0.482) and `follower_count` (0.231), which fails under domain shift or adversarial spoofing.
3. **Cross-Lingual Tokenization Gap:** Monolingual English models suffer subword fragmentation on Arabic texts, underscoring the necessity of multilingual representations for international operational deployments.
