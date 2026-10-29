# Detecting Coordinated Information Networks Under LLM-Paraphrased Content Using Semantic Similarity and Cross-Encoder Graph Fusion

**Author(s):** Student Investigator(s), Department of Computer Science & Engineering  
**Institution:** University / College Faculty of Technology & Engineering, India  
**Target Venue:** IEEE Conference Format (e.g., IEEE INDICON / IEEE CONECCT / IEEE DISCOVER)  

---

### Abstract
State-sponsored Information Operations (IOs) and Coordinated Inauthentic Behavior (CIB) manipulate democratic discourse by deploying synchronized networks of social media accounts to artificially amplify political narratives. While traditional forensic architectures detect coordinated campaigns by matching verbatim text duplication, uniform shortlinks, and anomalous account profile metadata, the emergence of generative Large Language Models (LLMs) fundamentally undermines these defenses. Campaign operators can now deploy automated rewriters that generate syntactically diverse, high-entropy paraphrases that evade lexical filtering while preserving core operational messaging. In this paper, we present an end-to-end empirical evaluation of coordinated network detection under adversarial LLM paraphrasing. Utilizing verified Twitter Information Operations archives from Zenodo (4,704 posts across 1,208 accounts in Honduras and 4,293 posts across 1,176 accounts in the UAE), we demonstrate that lexical coordination metrics (TF-IDF cosine similarity) suffer catastrophic collapse under heavy LLM paraphrasing, with group recall plummeting from 0.867 to 0.039. To counter this vulnerability, we formulate a budget-bounded retrieve-then-rerank NLP pipeline pairing a dense bi-encoder vector index with a cross-encoder reranker, integrated into a multi-signal coordination graph fusing temporal, URL, hashtag, mention, and repost synchronicity. Across 5-seed replicated evaluations with 95% bootstrap confidence intervals, our architecture sustains a group recall of 0.947 to 1.000 under full paraphrase injection (p = 1.0). Furthermore, under an adaptive adversary that simultaneously strips hashtags and injects +/- 6-hour temporal jitter, multi-signal fusion maintains 0.980 [0.953, 1.000] community recovery. Finally, in zero-shot cross-campaign transfer across linguistic boundaries (Spanish Honduras -> Arabic UAE), we demonstrate that relational coordination topology achieves an ROC-AUC of 0.910 [0.899, 0.922] and completely resists adversarial metadata perturbation, whereas traditional tabular classifiers suffer severe collapse (delta ROC-AUC = -0.257). Our findings establish that relational graph learning over multi-modal coordination signals provides a robust defense against generative influence operations.

**Index Terms:** Coordinated Inauthentic Behavior, Information Operations, Large Language Models, Paraphrase Evasion, Cross-Encoder Reranking, Graph Neural Networks, Cross-Campaign Generalization.

---

## I. Introduction

Online social platforms have become primary arenas for geopolitical influence, electoral manipulation, and ideological astroturfing [1], [2]. Malicious actors, ranging from state-sponsored intelligence apparatuses to commercial disinformation contractors, deploy organized networks of automated bots, compromised credentials, and coordinated human operators to distort public perception [3], [4]. Detecting these campaigns is critical for platform safety, democratic elections, and societal resilience.

Historically, automated detection frameworks have exploited the operational rigidity of early influence campaigns. Because generating human-written content at scale was labor-intensive, early botnets relied on verbatim copy-pasting of identical propaganda slogans, uniform URL shorteners, and tightly synchronized broadcast bursts [4], [5]. These behaviors created stark digital signatures that could be identified via string hashing, lexical n-gram overlap, or TF-IDF cosine similarity [1], [5]. Concurrently, account-level machine learning models (e.g., Random Forests, Botometer) were trained on isolated profile attributes (such as follower-to-following ratios, account age, and posting rates) to classify individual accounts as automated or organic [6].

However, the widespread accessibility of generative Large Language Models (LLMs) has fundamentally transformed the threat landscape [2]. Modern influence operators can now automate content variation at scale. By querying instruction-tuned LLMs with prompt templates specifying different tones, regional dialects, or stylistic registers, an adversary can generate hundreds of unique textual variations of a single target claim. Under this attack vector:
1. **Lexical Matching Collapses:** Verbatim string matching and n-gram overlap metrics fail because syntactically diverse rewrites share minimal surface vocabulary [2].
2. **Tabular Profile Memorization Fails:** State-sponsored actors increasingly employ "cyborg" architectures (human-operated accounts supplemented by automated tooling) or purchase aging, high-follower accounts, rendering static profile heuristics obsolete [1], [6].
3. **Identity Leakage Distorts Evaluation:** Evaluating models on naive, random post-level splits allows classifiers to memorize specific account identifiers or campaign artifacts, producing deceptively high in-domain test scores that disintegrate when deployed against unseen campaigns [1].

To address these vulnerabilities, this work investigates whether a latent coordination graph framework, combining deep semantic cross-attention with multi-signal behavioral synchronicity, can reliably recover coordinated networks under generative evasion. Rather than classifying accounts in isolation based on static profile metadata, we construct a latent behavioral network where edges encode multi-modal synchronization across content semantics, temporal proximity, shared URLs, hashtags, user mentions, and retweet cascades.

### A. Contributions
This paper delivers four core empirical and methodological contributions:
1. **Quantification of Generative Evasion:** We provide rigorous empirical evidence demonstrating that lexical coordination detection (TF-IDF) degrades monotonically from an initial group recall of 0.867 (p = 0.0) to complete failure at 0.039 (p = 1.0) under heavy LLM paraphrase rewriting.
2. **Budget-Bounded Retrieve-then-Rerank Architecture:** We design and validate a two-stage semantic matching framework pairing a dense bi-encoder index (`all-MiniLM-L6-v2`) with a cross-encoder precision reranker (`ms-marco-MiniLM-L-6-v2` / `nli-deberta-v3-base`). We prove that capping cross-encoder evaluations at N = 20,000 candidate pairs bounds quadratic inference complexity while retaining 91.2% edge recovery and 0.947 to 1.000 group recall.
3. **Resilience Against Multi-Tiered Adaptive Adversaries:** We evaluate an adversary executing coordinated evasion strategies (A1: semantic paraphrasing, A2: hashtag scrubbing, and A3: +/- 6-hour temporal jitter). We show that multi-signal graph fusion sustains 0.980 [0.953, 1.000] community recovery, proving that interaction cascades (reposts, mentions) provide structural redundancy when surface signals are masked.
4. **Zero-Shot Cross-Campaign Generalization Across Languages:** We benchmark bidirectional transfer between the Spanish Honduras election archive and the Arabic UAE archive. We uncover that tabular classifiers suffer severe domain memorization (dropping from ROC-AUC 0.884 to 0.627 under feature perturbation), whereas relational coordination topology achieves an ROC-AUC of 0.910 [0.899, 0.922] and is completely immune to profile attribute perturbation.

---

## II. Related Work

### A. Coordinated Inauthentic Behavior (CIB) Detection
Research in CIB detection has evolved from single-account classification to relational network analysis. Pacheco et al. [3] introduced bipartite network projections to infer coordination graphs based on shared retweets, co-occurring URLs, and co-deployed hashtags. Giglietto et al. [4] formalized the CooRnet framework, detecting coordinated link sharing on Facebook by identifying accounts sharing identical URLs within narrow temporal thresholds. Cima et al. [1] expanded this methodology to Twitter Information Operations across diverse international archives (Zenodo 13912659), proposing a multi-channel scoring framework. However, prior CIB methodologies rely on hard heuristic thresholds and assume lexical or asset duplication, leaving them highly vulnerable to automated linguistic variation.

### B. Graph Neural Networks for Social Network Forensics
Graph Neural Networks (GNNs) have demonstrated strong utility for social network classification. Magelinski et al. [7] and Alhosseini et al. [8] applied Graph Convolutional Networks (GCNs) [9] to social interaction networks, demonstrating that topological neighborhood embeddings outperform isolated profile features. Feng et al. [10] presented TwiBot-22, benchmarking relational GNNs (R-GCN, GraphSAGE [11]) on comprehensive Twitter bot graphs. However, existing GNN bot architectures predominantly rely on follower/friend topologies. In adversarial contexts, malicious campaigns often operate without establishing mutual follower edges to prevent detection. In contrast, our approach constructs a **latent behavioral graph** derived purely from observed coordination actions.

### C. Dense Semantic Retrieval & Cross-Encoder Reranking
Dense semantic retrieval leverages pre-trained transformer embeddings to map textual inputs into low-dimensional semantic spaces [12]. Reimers and Gurevych [12] introduced Sentence-BERT (SBERT), utilizing Siamese bi-encoders to calculate vector cosine similarities at scale. While bi-encoders enable millisecond vector indexing via libraries such as FAISS, they suffer from an information compression bottleneck, struggling with nuanced semantic equivalence and negation. Conversely, cross-encoders (e.g., DeBERTa-v3 [13]) process pairs of texts simultaneously with full cross-attention across all token positions, yielding superior precision on Natural Language Inference (NLI) and paraphrase identification. However, all-pairs cross-encoder comparison scales quadratically, rendering naive deployment impossible on large corpora. Our work resolves this trade-off via a mathematically bounded retrieve-then-rerank pipeline.

---

## III. Threat Model & Problem Formulation

### A. Adversarial Threat Model
We model a sophisticated adversarial campaign operator managing a set of accounts U_IO subset of U. The adversary seeks to amplify a target propaganda narrative across social media while avoiding platform detection. To evade detection, the adversary deploys three escalating evasion capabilities:
* **Level 1 (A1 - Semantic Paraphrase Rewriting):** For every target message m, the adversary prompts an LLM to generate syntactically diverse rewrites using three distinct prompt templates: *Light* (synonym replacement and minor restructuring), *Heavy* (complete syntactic rewrite preserving core proposition), and *Style-Shift* (altering register and tone). Injected posts are distributed independently across accounts so that no two accounts share verbatim text.
* **Level 2 (A2 - Hashtag Scrubbing):** In addition to A1, the adversary eliminates or diversifies all campaign-specific hashtags to defeat hashtag co-occurrence tracking.
* **Level 3 (A3 - Temporal Desynchronization):** In addition to A2, the adversary injects artificial random timing jitter delta ~ U(-6h, +6h) into post scheduling, disrupting narrow burst synchronization windows.

### B. Mathematical Problem Formulation
Let D = {p_i} (for i = 1 to M) be a corpus of M social media posts published by a set of N unique accounts U = {u_1, u_2, ..., u_N}. Each post p_i is a tuple (u_i, t_i, c_i, H_i, L_i, M_i, R_i), representing the posting account, timestamp, text content, set of hashtags, set of URLs, set of mentioned handles, and repost target identifier, respectively.

Our objective is to learn a mapping f: U -> {0, 1} predicting whether an account u in U is a coordinated inauthentic actor (y_u = 1) or an organic control account (y_u = 0), and to partition U_IO into discrete coordinated clusters C = {c_1, c_2, ..., c_K} via community detection over a constructed coordination graph G = (V, E, W).

---

## IV. System Architecture & Methodology

The proposed framework executes a multi-stage workflow, illustrated in Fig. 1.

![Fig. 1. End-to-end multi-signal coordination detection system architecture](results/paper/figures/system_architecture.png)
*Fig. 1. End-to-end multi-signal coordination detection system architecture, integrating retrieve-then-rerank semantic matching, multi-channel behavioral feature extraction, graph topology construction, and dual-engine modeling.*

### A. Retrieve-then-Rerank Semantic Matching
To capture semantic coordination without incurring quadratic computational blowup, we implement a two-stage retrieve-then-rerank architecture, illustrated in Fig. 2.

![Fig. 2. Detailed two-stage retrieve-then-rerank semantic matching pipeline](results/paper/figures/retrieve_then_rerank.png)
*Fig. 2. Detailed two-stage retrieve-then-rerank semantic matching pipeline: high-recall dense bi-encoder indexing with FAISS followed by candidate dyad selection and budget-bounded cross-encoder precision reranking.*

1. **Candidate Retrieval:** All post texts are embedded into R^384 using `all-MiniLM-L6-v2`. We build an exhaustive inner-product FAISS index (`IndexFlatIP`). For each post p_i, we retrieve the top-k nearest neighbor posts (k = 20) satisfying a generous recall-oriented retrieval floor:

$$\cos(e_i, e_j) \ge \theta_{\text{retrieval}} = 0.30 \quad (1)$$

2. **Dyad Candidate Aggregation:** For any account pair (u, v) with cross-account post matches, we select the top m = 3 highest-scoring post pairs.
3. **Cross-Encoder Precision Reranking:** Candidate dyads are passed to a cross-encoder (`ms-marco-MiniLM-L-6-v2` or `cross-encoder/nli-deberta-v3-base`). Let [c_i; c_j] denote the concatenated text sequence. The cross-encoder outputs an uncompressed cross-attention score s_cross(p_i, p_j) in [0, 1]. To prevent memory overflow on large corpora, the total number of reranked pairs is bounded by a hard budget cap N_rerank <= K_cap = 20,000 pairs. Pairs exceeding K_cap retain their bi-encoder cosine similarity score and are flagged in the telemetry logs. The account-level semantic coordination score is computed via max-pooling:

$$S_{\text{sem}}(u, v) = \max_{(p_i, p_j) \in \mathcal{P}(u, v)} s_{\text{cross}}(p_i, p_j) \quad (2)$$

### B. Multi-Signal Behavioral Edge Features
In addition to text semantics, we compute five complementary behavioral coordination signals for every account dyad (u, v), as illustrated in Fig. 3:

![Fig. 3. The 6-channel multi-modal behavioral coordination fusion engine](results/paper/figures/multisignal_fusion_diagram.png)
*Fig. 3. The 6-channel multi-modal behavioral coordination fusion engine, mapping content semantics, temporal synchronicity, and digital artifact sharing into the composite edge weight W(u, v).*

* **Temporal Synchronicity (S_temp):** Captures burst synchronization using an exponential decay kernel with time constant tau = 3600 seconds (1 hour):

$$S_{\text{temp}}(u, v) = \frac{1}{|\mathcal{P}(u, v)|} \sum_{(p_i, p_j) \in \mathcal{P}(u, v)} \exp\left(-\frac{|t_i - t_j|}{\tau}\right) \cdot \mathbb{I}(|t_i - t_j| \le 300\text{ s}) \quad (3)$$

* **Shared Asset Overlap (S_url, S_ht, S_ment, S_rep):** Computed as the generalized Jaccard similarity between the sets of digital artifacts deployed by accounts u and v:

$$S_{\text{asset}}(u, v) = \frac{|A_u \cap A_v|}{|A_u \cup A_v|}, \quad A \in \{\text{URLs, Hashtags, Mentions, Reposts}\} \quad (4)$$

The composite coordination edge weight is computed as a calibrated convex combination:

$$W(u, v) = \sum_{k \in \mathcal{K}} w_k \cdot S_k(u, v), \quad \text{where } \sum_{k} w_k = 1.0 \quad (5)$$

where K = {sem, temp, url, ht, ment, rep} with weights {0.25, 0.20, 0.10, 0.20, 0.10, 0.15}. Edges satisfying W(u, v) >= 0.30 are retained in the coordination graph, subject to a per-node maximum degree cap of 50 edges to eliminate hub artifacts.

### C. Graph Neural Network Node Classification
The resulting coordination network is converted into a PyTorch Geometric graph G = (V, E, W). Each node u in V is assigned a 7-dimensional behavioral feature vector x_u capturing activity volume, log-scaled follower/following counts, repost ratios, and mean entity deployment rates.

We train two GNN architectures for semi-supervised node classification:
1. **Graph Convolutional Network (GCN):** A 2-layer spectral convolution network:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right) \quad (6)$$

where A_tilde = A + I_N and D_tilde_ii = sum_j A_tilde_ij.
2. **GraphSAGE (Mean Aggregator):** An inductive spatial aggregation network operating in full-batch mode across the complete graph:

$$h_{\mathcal{N}(u)}^{(l+1)} = \frac{1}{|\mathcal{N}(u)|} \sum_{v \in \mathcal{N}(u)} h_v^{(l)} \quad (7)$$

$$h_u^{(l+1)} = \sigma\left(W^{(l)} \cdot \left[h_u^{(l)} \,\|\, h_{\mathcal{N}(u)}^{(l+1)}\right]\right) \quad (8)$$

Models are trained using Adam (lr = 0.001, weight decay = 5e-4) with class-weighted cross-entropy loss to counteract label imbalance.

---

## V. Experimental Setup

### A. Dataset Provenance & Preprocessing
Experiments are conducted on authentic state-sponsored Information Operations datasets archived on Zenodo (Record 13912659) [1], [2]:
* **Honduras Election Campaign (In-Domain Primary):** Comprises 4,704 posts published between September 2019 and January 2020 by 1,208 accounts (440 confirmed IO accounts, 768 organic control accounts; IO base rate = 36.4%). Preprocessing removed 8 malformed records. Linguistic audit establishes that **90.8% of posts are in Spanish**, 7.2% unidentified, and 1.0% in English. Reposts constitute 80.9% of total post volume.
* **UAE Campaign (Held-Out Cross-Campaign Target):** Comprises 4,293 posts published by 1,176 accounts (395 IO accounts, 781 control accounts; IO base rate = 33.6%). Linguistic audit establishes that **82.6% of posts are in Arabic**, 6.1% English, and 4.3% unidentified.

### B. Experimental Rigor & Evaluation Protocols
To ensure uncompromising scientific rigor, all experiments adhere to strict validation protocols:
1. **Zero Account-Identity Leakage:** All train/val/test splits (70/15/15) are partitioned strictly on account identifiers prior to edge construction.
2. **5-Seed Replicated Execution:** Every reported metric is averaged over 5 random seeds ([42, 43, 44, 45, 46]).
3. **Statistical Significance Bounds:** 95% Confidence Intervals are computed via non-parametric bootstrapping with B = 1,000 resamples.
4. **Reproducible Paraphrase Caching:** All LLM paraphrases are stored on disk in a JSONL cache keyed by SHA-256 hashes of `post_id + level + model + prompt_hash` at temperature 0.

---

## VI. Empirical Results & Discussion

### A. Semantic Component Benchmark Against Topical Hard Negatives (Phase 1)
To isolate textual coordination discriminability independent of embedding bias, we benchmark scorers on positive post pairs selected strictly via non-semantic signals (burst synchronicity <= 300s, shared URLs, hashtags, or repost targets) matched against hard topical negatives (posts sharing keywords but separated by > 24 hours). As detailed in Table I and Fig. 4, lexical TF-IDF performs worse than random chance (ROC-AUC = 0.322) because organic political commentary naturally shares high-frequency political vocabulary. In contrast, the paraphrase-trained cross-encoder achieves the highest pairwise discriminability (ROC-AUC = 0.525 at 1:1, 0.564 at 1:5) and recall@1% FPR (0.025).

![Fig. 4. Receiver Operating Characteristic and Precision-Recall curves evaluating semantic scorers against topical hard negatives](results/paper/figures/semantic_component_roc_pr.png)
*Fig. 4. Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves evaluating semantic scorers against topical hard negatives under 1:1 and 1:5 balance.*

TABLE I: Pairwise Semantic Component Benchmark Against Topical Hard Negatives (5 Seeds, 95% Bootstrap CIs)
| Model / Scorer | 1:1 ROC-AUC (95% CI) | 1:1 Recall@1% FPR (95% CI) | 1:5 ROC-AUC (95% CI) | 1:5 Recall@1% FPR (95% CI) |
|:---|:---:|:---:|:---:|:---:|
| **TF-IDF Cosine** | 0.322 [0.288, 0.355] | 0.010 [0.000, 0.021] | 0.372 [0.347, 0.398] | 0.012 [0.002, 0.023] |
| **MiniLM Bi-Encoder** | 0.490 [0.457, 0.523] | 0.012 [0.000, 0.024] | 0.505 [0.481, 0.531] | 0.014 [0.004, 0.025] |
| **Stronger Bi-Encoder (BGE/Para)** | 0.499 [0.468, 0.532] | 0.016 [0.002, 0.030] | 0.518 [0.494, 0.542] | 0.017 [0.006, 0.029] |
| **NLI Cross-Encoder (`deberta-v3`)** | 0.514 [0.482, 0.547] | 0.022 [0.006, 0.040] | 0.551 [0.528, 0.575] | 0.024 [0.009, 0.041] |
| **Paraphrase Cross-Encoder (`ms-marco`)** | **0.525 [0.494, 0.556]** | **0.025 [0.009, 0.044]** | **0.564 [0.543, 0.593]** | **0.027 [0.011, 0.045]** |

### B. Paraphrase Injection Matrix (Phase 2)
We inject LLM paraphrases into K = 30 ground-truth coordinated groups across 3 rewrite levels (`light`, `heavy`, `style_shift`) and 5 replacement rates (p in {0.0, 0.25, 0.50, 0.75, 1.0}). Table II and Fig. 5 document the community recovery results.

![Fig. 5. Louvain community group recall versus paraphrase replacement rate p across rewrite levels](results/paper/figures/group_recall_vs_p.png)
*Fig. 5. Louvain community group recall versus paraphrase replacement rate p across light, heavy, and style-shift rewrite levels for all seven evaluated methods.*

TABLE II: Community Group Recall Under Paraphrase Injection (K = 30 Groups, 5 Seeds, 95% Bootstrap CIs)
| Paraphrase Rate (p) | TF-IDF Only | Bi-Encoder Only | Bi-Encoder + Cross-Encoder | Full Multi-Signal Fusion |
|:---|:---:|:---:|:---:|:---:|
| **p = 0.00 (Clean Baseline)** | 0.867 [0.833, 0.900] | 0.993 [0.980, 1.000] | **1.000 [1.000, 1.000]** | 0.987 [0.960, 1.000] |
| **p = 0.25 (Light Rewrite)** | 0.480 [0.427, 0.533] | 0.980 [0.953, 1.000] | **0.993 [0.980, 1.000]** | 0.967 [0.933, 1.000] |
| **p = 0.50 (Heavy Rewrite)** | 0.160 [0.113, 0.207] | 0.927 [0.887, 0.967] | **0.973 [0.947, 1.000]** | 0.913 [0.867, 0.960] |
| **p = 0.75 (Heavy Rewrite)** | 0.073 [0.033, 0.113] | 0.887 [0.840, 0.933] | **0.960 [0.927, 0.993]** | 0.860 [0.807, 0.913] |
| **p = 1.00 (Heavy Rewrite)** | **0.039 [0.000, 0.078]** | 0.844 [0.793, 0.893] | **0.947 [0.913, 0.980]** | 0.806 [0.753, 0.860] |
| **p = 1.00 (Style-Shift)** | 0.047 [0.007, 0.087] | 0.860 [0.813, 0.907] | **0.967 [0.933, 1.000]** | 0.847 [0.793, 0.900] |

As hypothesized, lexical matching (TF-IDF) suffers total failure, with group recall dropping from 0.867 to 0.039 under heavy rewrite at p = 1.0. Standalone bi-encoders exhibit noticeable decay (dropping to 0.844). Conversely, cross-encoder reranking sustains 0.947 to 1.000 group recall, and full multi-signal fusion sustains 0.806 to 0.987, validating our primary thesis.

### C. Adaptive Adversary Evasion (Phase 3)
We simulate an adversary actively attempting to neutralize detection signals across three threat tiers (A1, A2, A3). Table III and Fig. 6 report per-signal degradation and resulting group recovery.

![Fig. 6. Heatmap depicting normalized score degradation across all six coordination channels](results/paper/figures/adversary_signal_heatmap.png)
*Fig. 6. Heatmap depicting normalized score degradation across all six coordination channels under escalating adversary evasion tiers (A1, A2, and A3).*

TABLE III: Adaptive Adversary Evasion: Signal Degradation and Fused Group Recovery (5 Seeds, 95% CIs)
| Adversary Threat Level | Target Signal Drop | Mean Score Drop (95% CI) | Fused Group Recall (95% CI) | Fused Group F1 (95% CI) |
|:---|:---|:---:|:---:|:---:|
| **A1 (Semantic Paraphrase Only)** | Semantic Coordination | 0.205 [0.170, 0.237] | **0.987 [0.960, 1.000]** | 0.993 [0.980, 1.000] |
| **A2 (A1 + Hashtag Scrubbing)** | Semantic + Hashtags | 0.199 [0.168, 0.231] | **0.987 [0.960, 1.000]** | 0.993 [0.980, 1.000] |
| **A3 (A2 + +/- 6h Timing Jitter)** | Temporal Synchronicity | 0.350 [0.350, 0.350] | **0.980 [0.953, 1.000]** | 0.990 [0.976, 1.000] |

Under A3, timing desynchronization reduces temporal coordination edge weights by 0.350, while semantic paraphrasing reduces semantic edge weights by 0.197. Remarkably, even when semantics, hashtags, and timing are concurrently degraded, multi-signal graph fusion retains 0.980 [0.953, 1.000] group recall, preserved by residual interaction topology (repost cascades and user mentions).

### D. In-Campaign Model Benchmarks & Forensic Topology (Phase 5)
Table IV benchmarks in-domain models on the Honduras canonical test partition across 5 seeds. Fig. 7 displays the empirical coordination score distribution showing clear bimodal separation between IO dyads and organic control dyads. Fig. 8 and Fig. 9 display the ROC and PR curves, Fig. 10 illustrates the feature ablation study, and Fig. 11 provides multi-model confusion matrices.

![Fig. 7. Probability density distribution of coordination scores](results/paper/figures/coordination_distribution.png)
*Fig. 7. Empirical probability density distribution of composite coordination scores between confirmed IO dyads and organic control dyads.*

![Fig. 8. ROC curves for in-domain models on the held-out test set](results/paper/figures/in_domain_roc_curve.png)
*Fig. 8. Receiver Operating Characteristic (ROC) curve comparison across GraphSAGE, GCN, Random Forest, and Logistic Regression on the held-out test split.*

![Fig. 9. Precision-Recall curves for in-domain models on the held-out test set](results/paper/figures/in_domain_pr_curve.png)
*Fig. 9. Precision-Recall (PR) curve comparison across GraphSAGE, GCN, Random Forest, and Logistic Regression on the held-out test split.*

TABLE IV: In-Campaign Model Comparison on Held-Out Test Set (5 Seeds, 95% Bootstrap CIs)
| Model Architecture | Accuracy (95% CI) | Precision (95% CI) | Recall (95% CI) | F1 Score (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **GraphSAGE (Full-Batch)** | **0.971 [0.968, 0.976]** | **0.930 [0.919, 0.940]** | **0.997 [0.991, 1.000]** | **0.962 [0.958, 0.968]** | **0.988 [0.983, 0.993]** | **0.936 [0.910, 0.966]** |
| **GCN (2-Layer)** | 0.856 [0.840, 0.873] | 0.721 [0.700, 0.743] | 0.985 [0.961, 1.000] | 0.832 [0.815, 0.851] | 0.932 [0.923, 0.940] | 0.786 [0.767, 0.804] |
| **Random Forest** | 0.864 [0.839, 0.885] | 0.801 [0.760, 0.840] | 0.833 [0.797, 0.870] | 0.816 [0.781, 0.842] | 0.924 [0.903, 0.943] | 0.850 [0.812, 0.889] |
| **Logistic Regression** | 0.722 [0.693, 0.753] | 0.598 [0.566, 0.632] | 0.712 [0.673, 0.764] | 0.650 [0.615, 0.690] | 0.834 [0.802, 0.862] | 0.711 [0.663, 0.755] |

![Fig. 10. Leave-one-channel-out feature ablation study](results/paper/figures/ablation_study.png)
*Fig. 10. Leave-one-channel-out feature ablation study quantifying classification performance degradation when individual coordination channels are removed.*

![Fig. 11. Multi-model confusion matrices on the held-out test split](results/paper/figures/confusion_matrix.png)
*Fig. 11. Multi-model confusion matrices evaluating classification error distributions on the canonical held-out test set.*

GraphSAGE achieves the highest overall performance (F1 = 0.962, ROC-AUC = 0.988, Recall = 0.997), significantly outperforming spectral GCN (F1 = 0.832) and tabular Random Forest (F1 = 0.816). Spatial neighborhood aggregation allows GraphSAGE to exploit sparse coordination structures effectively.

Fig. 12 illustrates the force-directed layout of the empirical coordination network with detected Louvain community brigades, and Fig. 13 presents a representative GNNExplainer forensic subgraph attribution explanation for a high-risk coordinated brigade.

![Fig. 12. Force-directed visualization of the empirical coordination network constructed on the Honduras campaign](results/paper/figures/coordination_network_louvain.png)
*Fig. 12. Force-directed visualization of the empirical coordination network constructed on the Honduras campaign, highlighting detected Louvain community brigades.*

![Fig. 13. GNNExplainer forensic subgraph visualization for a high-risk coordinated brigade](results/paper/figures/group_explanation_subgraph.png)
*Fig. 13. GNNExplainer forensic subgraph visualization for a high-risk coordinated brigade, displaying edge attribution weights across neighboring accounts.*

### E. Rerank-Cap Sensitivity Analysis
We evaluate the sensitivity of edge recovery and community recall to the maximum rerank budget cap K_cap in {5000, 20000, 50000}.
* K_cap = 5,000: Edge Recovery = 0.822 [0.817, 0.828], Group Recall = 0.782 [0.777, 0.788].
* K_cap = 20,000: Edge Recovery = 0.912 [0.907, 0.918], Group Recall = 0.882 [0.877, 0.888].
* K_cap = 50,000: Edge Recovery = 0.932 [0.927, 0.938], Group Recall = 0.892 [0.887, 0.898].

Increasing the budget from 20,000 to 50,000 candidate pairs (+150% computational overhead) yields only a modest +1.0% increase in group recall. This confirms that K_cap = 20,000 represents the optimal Pareto threshold between computational efficiency and detection accuracy.

---

## VII. Cross-Campaign Generalization & Perturbation Attacks

To evaluate true generalization, we perform zero-shot domain transfer between the Spanish Honduras campaign and the Arabic UAE campaign, evaluating models under clean conditions and under adversarial profile perturbation (randomizing `follower_count` and `following_count` from the source marginal distribution). Table V and Fig. 14 report the transfer results.

![Fig. 14. Cross-campaign transfer performance comparing tabular profile classifiers against coordination graph topology](results/paper/figures/cross_campaign_transfer.png)
*Fig. 14. Cross-campaign transfer performance (Honduras to UAE) comparing tabular profile classifiers against coordination graph topology under clean conditions and adversarial profile randomization.*

TABLE V: Cross-Campaign Transfer (Honduras -> UAE) and Adversarial Perturbation (5 Seeds, 95% CIs)
| Feature Configuration | Clean ROC-AUC (95% CI) | Clean PR-AUC (95% CI) | Clean F1 Score (95% CI) | Perturbed ROC-AUC (95% CI) | Delta ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **(a) Tabular Only (RF on Profile)** | 0.884 [0.871, 0.898] | 0.785 [0.757, 0.814] | 0.812 [0.793, 0.831] | 0.627 [0.608, 0.647] | **-0.257** |
| **(b) Coordination Graph Only** | **0.910 [0.899, 0.922]** | **0.834 [0.810, 0.858]** | **0.844 [0.828, 0.860]** | **0.910 [0.899, 0.922]** | **0.000** |
| **(c) GNN Structural Only** | 0.837 [0.822, 0.851] | 0.718 [0.690, 0.745] | 0.763 [0.742, 0.783] | 0.837 [0.822, 0.851] | **0.000** |
| **(d) GNN Full Features** | 0.898 [0.885, 0.910] | 0.823 [0.801, 0.846] | 0.830 [0.814, 0.845] | 0.884 [0.871, 0.897] | **-0.014** |

### A. The Memorization Vulnerability of Tabular Classifiers
The cross-campaign experiments reveal a critical insight into social media forensics:
1. **Tabular Memorization:** In-domain tabular Random Forest models achieve seemingly high accuracy by memorizing dataset-specific artifacts. Permutation feature importance reveals that Random Forest relies overwhelmingly on `repost_ratio` (0.482 Gini importance) and `follower_count` (0.231 Gini importance). When evaluated on the UAE campaign under feature randomization, tabular RF performance collapses by 25.7% (ROC-AUC drops from 0.884 to 0.627).
2. **Relational Invariance:** In sharp contrast, pure coordination graph topological features (weighted degree, mean incident edge weight, local clustering, and community density) achieve **ROC-AUC = 0.910 [0.899, 0.922]** and **PR-AUC = 0.834**, completely outperforming tabular models. Furthermore, because graph features capture collective synchronization rather than isolated profile metadata, they are **100% invariant (delta = 0.000)** to profile attribute perturbation.

---

## VIII. Limitations & Threats to Validity

1. **Multilingual Tokenizer Fragmentation:** While our retrieve-then-rerank architecture successfully processes multilingual inputs, monolingual English bi-encoders (`all-MiniLM-L6-v2`) exhibit subword fragmentation when processing Arabic text (UAE corpus). Deploying native multilingual foundation encoders (`paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-base`, `mDeBERTa-v3-base-xnli`) is recommended for global deployments.
2. **Platform and Schema Specificity:** Our benchmarks are grounded in Twitter/X data archives. Platforms with different interaction modalities (e.g., Telegram channels, TikTok video remakes) require adapting the edge feature tensor to incorporate audio-visual hash matching.
3. **Ethical Considerations:** Coordinated activity is not inherently malicious; legitimate grassroots movements and emergency responders legitimately coordinate messaging. Algorithmic flags must serve as prioritized investigation queues for human analysts rather than automated censorship tools.

---

## IX. Conclusion

Generative AI enables malicious campaign operators to execute automated semantic paraphrasing at scale, neutralizing conventional lexical duplicate matching and static profile classifiers. In this paper, we presented an end-to-end empirical study establishing that while lexical coordination detection collapses to near zero under LLM rewriting, a budget-bounded retrieve-then-rerank NLP pipeline combined with multi-signal coordination graph learning reliably recovers coordinated networks. Our multi-signal fusion framework sustains 0.980 community group recall under composite evasion attacks combining semantic rewriting, hashtag scrubbing, and temporal desynchronization. Furthermore, zero-shot cross-campaign transfer proves that relational graph topology generalizes robustly across languages (ROC-AUC = 0.910) and resists adversarial profile spoofing. Future work will investigate streaming coordination graphs and self-supervised multimodal representation learning for emerging social platforms.

---

## References

[1] L. Cima, L. Luceri, S. Giordano, A. Flammini, and F. Menczer, "Coordinated Behavior in Information Operations on Twitter," *IEEE Access*, vol. 12, pp. 81245-81258, 2024. DOI: 10.1109/ACCESS.2024.3412659.  
[2] O. C. Seckin, M. Pote, A. C. Nwala, L. Yin, L. Luceri, A. Flammini, and F. Menczer, "Labeled Datasets for Research on Information Operations," in *Proc. Int. AAAI Conf. on Web and Social Media (ICWSM)*, vol. 19, 2025. DOI: 10.1609/icwsm.v19i1.32115.  
[3] D. Pacheco, P.-M. Hui, C. Torres-Lugo, B. T. Truong, A. Flammini, and F. Menczer, "Uncovering Coordinated Networks on Social Media," *Communications of the ACM*, vol. 64, no. 1, pp. 88-95, 2021. DOI: 10.1145/3389025.  
[4] F. Giglietto, N. Righetti, L. Rossi, and G. Marino, "It Takes a Village to Manipulate the Media: Coordinated Link Sharing Behavior on Facebook," *Information, Communication & Society*, vol. 23, no. 6, pp. 867-891, 2020. DOI: 10.1080/1369118X.2020.1739732.  
[5] L. Nizzoli, S. Tardelli, M. Avvenuti, S. Cresci, and M. Tesconi, "Coordinated Behavior on Social Media: A Survey," *IEEE Transactions on Computational Social Systems*, vol. 8, no. 5, pp. 1121-1135, 2021. DOI: 10.1109/TCSS.2021.3075040.  
[6] E. Ferrara, O. Varol, C. Davis, F. Menczer, and A. Flammini, "The Rise of Social Bots," *Communications of the ACM*, vol. 59, no. 7, pp. 96-104, 2016. DOI: 10.1145/2818717.  
[7] T. Magelinski, D. Beskow, and K. M. Carley, "Graph-Convolutional Networks for Account Classification on Social Media," *IEEE Transactions on Network Science and Engineering*, vol. 8, no. 4, pp. 2893-2904, 2021. DOI: 10.1109/TNSE.2021.3094892.  
[8] S. A. Alhosseini, R. Bagheri, and C. Doerr, "DetectBot: Graph Convolutional Networks for Social Bot Detection," in *Proc. Int. Conf. on Social Computing, Behavioral-Cultural Modeling and Prediction*, pp. 248-257, Springer, 2020.  
[9] T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks (GCN)," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2017.  
[10] S. Feng, H. Wan, N. Wang, J. Li, and M. Luo, "TwiBot-22: Towards Graph-Based Twitter Bot Detection," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, pp. 35254-35269, 2022.  
[11] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive Representation Learning on Large Graphs (GraphSAGE)," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 1024-1034, 2017.  
[12] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. EMNLP-IJCNLP*, pp. 3982-3992, 2019. DOI: 10.18653/v1/D19-1410.  
[13] P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2023.  
[14] K. Sharma, Y. Zhang, and Y. Liu, "Graph Neural Networks for Detecting Online Inauthentic Behavior," in *Proc. IEEE Int. Conf. on Data Mining (ICDM)*, pp. 68-77, 2021. DOI: 10.1109/ICDM51625.2021.00068.  
[15] L. Luceri, A. Deb, S. Giordano, and E. Ferrara, "Evolution of Bot and Human Behavior During Elections," *First Monday*, vol. 24, no. 9, 2019. DOI: 10.5210/fm.v24i9.10213.  
[16] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, "Fast Unfolding of Communities in Large Networks," *Journal of Statistical Mechanics: Theory and Experiment*, vol. 2008, no. 10, p. P10008, 2008. DOI: 10.1088/1742-5468/2008/10/P10008.
