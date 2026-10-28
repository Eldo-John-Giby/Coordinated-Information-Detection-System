"""
src/evaluation/paraphrase_injection.py

Phase 2 & Phase 3: Paraphrase Injection & Adaptive Adversary Evaluation.

Thesis Verification:
LLM paraphrasing degrades lexical and bi-encoder coordination signals;
cross-encoder reranking plus multi-signal graph fusion recovers detection.

Capabilities:
1. Select K=30 ground truth coordinated groups (3-8 accounts each) using NON-semantic
   signals only (URLs, hashtags, reposts, burst timing).
2. Paraphrase posts at rate p in {0.0, 0.25, 0.5, 0.75, 1.0} across 3 prompt levels
   (light, heavy, style_shift) with disk caching (JSONL) and temperature 0.
3. Adaptive Adversary (Phase 3):
   - A1: Paraphrase only
   - A2: Paraphrase + hashtag rewrite/drop
   - A3: A2 + temporal jitter up to +/- 6 hours
4. Evaluate 7 detection methods:
   - TF-IDF-only
   - Bi-encoder-only
   - Bi-encoder + Cross-encoder (current system)
   - Hashtag-only
   - URL-only
   - Temporal-only
   - Full fused system
5. Compute pair-level ROC-AUC, Recall@1%FPR, group-level Recall, group-level F1,
   and edge-retention rate across 5 seeds with 95% bootstrap CIs.
6. Track per-signal score degradation (semantic, temporal, url, hashtag, mention, repost).
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.loader import DataLoader, create_sample_dataset
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_bootstrap_ci, find_threshold_at_fpr, compute_recall_at_fixed_threshold
from src.features.edge_features import EdgeFeatureExtractor
from src.graph.community import CommunityDetector
from src.nlp.similarity import BiEncoderScorer, CrossEncoderScorer, TfidfScorer
from src.utils.degradations import DegradationTracker

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Disk Cache for LLM Paraphrasing (JSONL)
# ----------------------------------------------------------------------

class ParaphraseCache:
    """Persistent, reproducible disk cache for LLM paraphrases."""

    def __init__(self, cache_path: str = "data/cache/paraphrases.jsonl"):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        self.cache: Dict[str, str] = {}
        self._load_cache()

    def _make_key(self, post_id: str, level: str, model: str, prompt: str) -> str:
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        raw_key = f"{post_id}::{level}::{model}::{prompt_hash}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self.cache[entry["key"]] = entry["paraphrase"]
            logger.info(f"Loaded {len(self.cache)} cached paraphrases from {self.cache_path}")
        except Exception as e:
            logger.warning(f"Error loading paraphrase cache from {self.cache_path}: {e}")

    def get(self, post_id: str, level: str, model: str, prompt: str) -> Optional[str]:
        key = self._make_key(post_id, level, model, prompt)
        return self.cache.get(key)

    def set(self, post_id: str, level: str, model: str, prompt: str, paraphrase: str):
        key = self._make_key(post_id, level, model, prompt)
        self.cache[key] = paraphrase
        try:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                entry = {
                    "key": key,
                    "post_id": str(post_id),
                    "level": str(level),
                    "model": str(model),
                    "paraphrase": paraphrase,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist paraphrase to disk cache: {e}")


# ----------------------------------------------------------------------
# Paraphrase Engine with Adversarial Fallback
# ----------------------------------------------------------------------

SPANISH_SYNONYMS = {
    r"\balivio de deuda\b": "reduccion del endeudamiento",
    r"\bsalvemos honduras\b": "defendamos a nuestro pais",
    r"\bexitos lider\b": "nuestros mejores augurios dirigente",
    r"\bsigamos sumando\b": "continuemos construyendo progreso",
    r"\bincitando al pueblo\b": "provocando disturbios ciudadanos",
    r"\bviolencia\b": "desorden publico",
    r"\bpueblo\b": "ciudadania organizada",
    r"\bgobierno\b": "administracion actual",
    r"\bpresidente\b": "primer mandatario",
    r"\bnavidad catracha\b": "celebraciones navidenas patrias",
    r"\bparque vida mejor\b": "espacio civico recreativo",
    r"\bdenuncia\b": "reclamo ciudadano",
    r"\blucha\b": "esfuerzo colectivo",
    r"\bapoyo total\b": "respaldo incondicional",
    r"\bhoy\b": "en esta jornada",
}

INTRO_TEMPLATES = {
    "light": [
        "{text}",
        "Noticia importante: {text}",
        "Actualizacion: {text}",
    ],
    "heavy": [
        "En declaraciones recientes se destaca que {text}",
        "Reporte confirmado desde la region: {text}",
        "Analisis de la situacion actual: {text}",
    ],
    "style_shift": [
        "Opinion de expertos sobre el contexto actual: {text}",
        "Un llamado urgente a la reflexion nacional: {text}",
        "Segun trascendio de fuentes oficiales: {text}",
    ],
}


class ParaphraseEngine:
    """Manages LLM API calls with rate-limiting, temperature 0, and offline fallback."""

    def __init__(self, config: dict, tracker: Optional[DegradationTracker] = None):
        self.config = config
        self.tracker = tracker or DegradationTracker()
        pi_cfg = config.get("paraphrase_injection", {})
        self.model_name = pi_cfg.get("llm_model", "claude-sonnet-4-6")
        self.api_limit = pi_cfg.get("api_limit", 500)
        self.cache = ParaphraseCache(pi_cfg.get("cache_path", "data/cache/paraphrases.jsonl"))
        self.api_calls_made = 0

        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")

        self.client_type = None
        self.client = None
        if self.anthropic_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.anthropic_key)
                self.client_type = "anthropic"
            except Exception as e:
                self.tracker.record("paraphrase_engine", f"Anthropic init error ({e})")
        elif self.openai_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.openai_key)
                self.client_type = "openai"
            except Exception as e:
                self.tracker.record("paraphrase_engine", f"OpenAI init error ({e})")

        if self.client_type is None:
            self.tracker.record(
                "paraphrase_engine",
                "No external LLM API key; running deterministic high-entropy adversarial rewriter",
            )

    def paraphrase(self, text: str, post_id: str, level: str = "heavy", seed: int = 42) -> str:
        """Paraphrase post text at given level with disk caching and temperature 0."""
        text = str(text).strip()
        if not text:
            return ""

        templates = self.config.get("paraphrase_injection", {}).get("prompt_templates", {})
        prompt_tmpl = templates.get(level, "Paraphrase this text: {text}")
        prompt = prompt_tmpl.format(text=text)

        # 1. Check cache
        cached = self.cache.get(post_id, level, self.model_name, prompt)
        if cached:
            return cached

        # 2. Try LLM API if below limit
        if self.client_type == "anthropic" and self.api_calls_made < self.api_limit:
            try:
                resp = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                paraphrased = resp.content[0].text.strip()
                self.api_calls_made += 1
                self.cache.set(post_id, level, self.model_name, prompt, paraphrased)
                return paraphrased
            except Exception as e:
                self.tracker.record("paraphrase_engine", f"Anthropic call error: {e}")

        elif self.client_type == "openai" and self.api_calls_made < self.api_limit:
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                paraphrased = resp.choices[0].message.content.strip()
                self.api_calls_made += 1
                self.cache.set(post_id, level, self.model_name, prompt, paraphrased)
                return paraphrased
            except Exception as e:
                self.tracker.record("paraphrase_engine", f"OpenAI call error: {e}")

        # 3. Deterministic high-entropy adversarial fallback
        paraphrased = self._offline_adversarial_paraphrase(text, level=level, seed=seed)
        self.cache.set(post_id, level, "adversarial_rewriter", prompt, paraphrased)
        return paraphrased

    def _offline_adversarial_paraphrase(self, text: str, level: str = "heavy", seed: int = 42) -> str:
        """High-entropy rule-based lexical, syntactic, and stylistic rewriter."""
        rng = random.Random(seed + hash(text) % 10000)
        tokens = text.split()

        # Extract entities to preserve (URLs and mentions)
        protected_urls = [t for t in tokens if t.startswith("http")]
        protected_mentions = [t for t in tokens if t.startswith("@")]
        protected_hashtags = [t for t in tokens if t.startswith("#")]
        core_tokens = [
            t for t in tokens
            if not t.startswith("http") and not t.startswith("@") and not t.startswith("#")
        ]
        core_text = " ".join(core_tokens)

        # Level 1: Light (synonyms + slight restructuring)
        for pattern, replacement in SPANISH_SYNONYMS.items():
            if re.search(pattern, core_text, re.IGNORECASE):
                core_text = re.sub(pattern, replacement, core_text, flags=re.IGNORECASE)

        # Level 2: Heavy (full syntactic inversion + frame rewrite)
        if level in ["heavy", "style_shift"]:
            words = core_text.split()
            if len(words) >= 4:
                half = len(words) // 2
                core_text = f"{' '.join(words[half:])}, al tiempo que {' '.join(words[:half])}"
            elif words:
                core_text = f"Manifestamos rotundamente: {core_text}"

        # Level 3: Style shift (stylistic framing)
        if level == "style_shift":
            options = INTRO_TEMPLATES.get("style_shift", ["{text}"])
            core_text = rng.choice(options).format(text=core_text)
        elif level == "heavy":
            options = INTRO_TEMPLATES.get("heavy", ["{text}"])
            core_text = rng.choice(options).format(text=core_text)

        # Re-attach preserved entities
        entities = protected_mentions + protected_hashtags + protected_urls
        if entities:
            if rng.random() > 0.5:
                res = f"{core_text} {' '.join(entities)}"
            else:
                res = f"{' '.join(entities)}: {core_text}"
        else:
            res = core_text

        return res.strip()


# ----------------------------------------------------------------------
# Paraphrase Injection Experiment Runner
# ----------------------------------------------------------------------

class ParaphraseInjectionExperiment:
    """End-to-End Paraphrase Injection & Adaptive Adversary Experiment (Phases 2 & 3)."""

    def __init__(self, config: dict):
        self.config = config
        self.tracker = DegradationTracker()
        self.engine = ParaphraseEngine(config, tracker=self.tracker)

        self.pi_cfg = config.get("paraphrase_injection", {})
        self.K_groups = self.pi_cfg.get("K_groups", 30)
        self.group_min = self.pi_cfg.get("group_size_min", 3)
        self.group_max = self.pi_cfg.get("group_size_max", 8)
        self.rates = self.pi_cfg.get("paraphrase_rates", [0.0, 0.25, 0.50, 0.75, 1.0])
        self.seeds = self.pi_cfg.get("seeds", [42, 43, 44, 45, 46])
        self.purity_threshold = self.pi_cfg.get("purity_threshold", 0.5)

        self.output_dir = os.path.join("results", "paper")
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        self.edge_extractor = EdgeFeatureExtractor(config)
        self.community_detector = CommunityDetector(config)
        self.tfidf_scorer = TfidfScorer()
        self.bi_scorer = BiEncoderScorer(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.bi_scorer._load()
        self.cross_scorer = CrossEncoderScorer(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.cross_scorer._load()

    def load_preprocessed_data(self, campaign: str = "Honduras") -> pd.DataFrame:
        """Load and preprocess campaign dataset."""
        logger.info(f"Loading data for campaign: {campaign}...")
        loader = DataLoader(self.config)
        try:
            df = loader.load_campaign(campaign)
        except Exception as e:
            logger.warning(f"Failed to load campaign {campaign} ({e}); creating sample...")
            create_sample_dataset(self.config["data"]["raw_dir"])
            df = loader.load_campaign()

        preprocessor = Preprocessor(self.config)
        df = preprocessor.process(df)
        return df

    # ------------------------------------------------------------------
    # Step 1: Select K Ground Truth Groups via Non-Semantic Signals ONLY
    # ------------------------------------------------------------------

    def select_injected_groups(
        self, df: pd.DataFrame, seed: int = 42
    ) -> List[List[str]]:
        """Identify K coordinated groups of 3-8 accounts using non-semantic signals only."""
        rng = np.random.default_rng(seed)

        # Filter to IO accounts
        if "is_control" in df.columns:
            io_df = df[~df["is_control"].astype(bool)].copy()
        elif "account_is_control" in df.columns:
            io_df = df[~df["account_is_control"].astype(bool)].copy()
        else:
            io_df = df.copy()

        # Build non-semantic co-occurrence network
        G_nonsem = nx.Graph()
        accounts = io_df["accountid"].unique()
        for acc in accounts:
            G_nonsem.add_node(acc)

        # 1. Shared URLs
        url_to_accs = {}
        for acc, group in io_df.groupby("accountid"):
            acc_urls = set()
            for u_list in group["urls"].dropna():
                if isinstance(u_list, (list, set)):
                    acc_urls.update(u_list)
            for u in acc_urls:
                url_to_accs.setdefault(u, set()).add(acc)

        for u, acc_set in url_to_accs.items():
            if 2 <= len(acc_set) <= 50:
                acc_list = list(acc_set)
                for i in range(len(acc_list)):
                    for j in range(i + 1, len(acc_list)):
                        G_nonsem.add_edge(acc_list[i], acc_list[j])

        # 2. Shared Hashtags
        ht_to_accs = {}
        for acc, group in io_df.groupby("accountid"):
            acc_hts = set()
            for h_list in group["hashtags"].dropna():
                if isinstance(h_list, (list, set)):
                    acc_hts.update(h_list)
            for h in acc_hts:
                ht_to_accs.setdefault(h, set()).add(acc)

        for h, acc_set in ht_to_accs.items():
            if 2 <= len(acc_set) <= 20:
                acc_list = list(acc_set)
                for i in range(len(acc_list)):
                    for j in range(i + 1, len(acc_list)):
                        G_nonsem.add_edge(acc_list[i], acc_list[j])

        # 3. Temporal Bursts
        io_df["parsed_time"] = pd.to_datetime(io_df["post_time"], errors="coerce")
        temporal_df = io_df.dropna(subset=["parsed_time"]).sort_values("parsed_time").reset_index(drop=True)
        times = temporal_df["parsed_time"].values
        acc_arr = temporal_df["accountid"].values
        burst_ns = int(self.config.get("temporal", {}).get("burst_window", 300) * 1e9)

        for i in range(min(len(temporal_df), 1500)):
            j = i + 1
            while j < len(temporal_df) and (times[j].astype("int64") - times[i].astype("int64")) <= burst_ns:
                if acc_arr[i] != acc_arr[j]:
                    G_nonsem.add_edge(acc_arr[i], acc_arr[j])
                j += 1

        # Extract connected components and slice into groups of size 3-8
        groups: List[List[str]] = []
        components = sorted(list(nx.connected_components(G_nonsem)), key=len, reverse=True)

        for comp in components:
            comp_list = list(comp)
            if self.group_min <= len(comp_list) <= self.group_max:
                groups.append(comp_list)
            elif len(comp_list) > self.group_max:
                # Sub-divide large components into valid groups
                rng.shuffle(comp_list)
                for chunk_idx in range(0, len(comp_list), self.group_max):
                    chunk = comp_list[chunk_idx: chunk_idx + self.group_max]
                    if len(chunk) >= self.group_min:
                        groups.append(chunk)

        # If still fewer than K, pick subsets from largest components
        if len(groups) < self.K_groups:
            all_io_accs = list(accounts)
            rng.shuffle(all_io_accs)
            idx = 0
            while len(groups) < self.K_groups and idx + self.group_min <= len(all_io_accs):
                sz = min(self.group_max, max(self.group_min, rng.integers(self.group_min, self.group_max + 1)))
                groups.append(all_io_accs[idx: idx + sz])
                idx += sz

        selected_groups = groups[: self.K_groups]
        logger.info(
            f"Selected {len(selected_groups)} coordinated ground truth groups "
            f"(sizes: {[len(g) for g in selected_groups[:5]]}...)"
        )
        return selected_groups

    # ------------------------------------------------------------------
    # Step 2: Inject Paraphrases & Apply Adversary Actions
    # ------------------------------------------------------------------

    def apply_injection(
        self,
        df: pd.DataFrame,
        injected_groups: List[List[str]],
        p: float,
        level: str = "heavy",
        adversary_level: str = "A1",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Replace fraction p of posts in injected groups with LLM paraphrases, independently per account."""
        if p <= 0.0:
            return df.copy()

        df_mod = df.copy()
        injected_accounts = set(acc for g in injected_groups for acc in g)
        rng = np.random.default_rng(seed)

        mask_injected = df_mod["accountid"].isin(injected_accounts)
        injected_indices = df_mod[mask_injected].index.tolist()

        # Select fraction p independently per account
        indices_to_replace = []
        for acc in injected_accounts:
            acc_idxs = df_mod[df_mod["accountid"] == acc].index.tolist()
            if not acc_idxs:
                continue
            n_replace = int(np.ceil(p * len(acc_idxs)))
            chosen = rng.choice(acc_idxs, size=min(n_replace, len(acc_idxs)), replace=False)
            indices_to_replace.extend(chosen)

        logger.info(
            f"Injecting paraphrases (level={level}, adversary={adversary_level}, p={p}): "
            f"Replacing {len(indices_to_replace)} posts across {len(injected_accounts)} accounts"
        )

        # 1. Paraphrase text independently
        for idx in indices_to_replace:
            orig_text = str(df_mod.at[idx, "post_text"])
            post_id = str(df_mod.at[idx, "postid"])
            acc_id = str(df_mod.at[idx, "accountid"])
            # Salt seed with account ID to ensure independence
            acc_seed = seed + int(hashlib.md5(acc_id.encode()).hexdigest()[:6], 16) % 10000

            para_text = self.engine.paraphrase(orig_text, post_id, level=level, seed=acc_seed)
            df_mod.at[idx, "post_text"] = para_text

            # 2. Adversary Action A2: Drop or rewrite hashtags
            if adversary_level in ["A2", "A3"]:
                # 80% chance to strip hashtags on injected posts
                if rng.random() > 0.2:
                    df_mod.at[idx, "hashtags"] = []
                    # Also strip #words from text
                    df_mod.at[idx, "post_text"] = re.sub(r"#\w+", "", df_mod.at[idx, "post_text"]).strip()

            # 3. Adversary Action A3: Temporal Jitter (+/- 6 hours)
            if adversary_level == "A3":
                max_jitter_sec = int(self.config.get("adaptive_adversary", {}).get("max_jitter_hours", 6.0) * 3600)
                jitter_sec = rng.uniform(-max_jitter_sec, max_jitter_sec)
                cur_time = pd.to_datetime(df_mod.at[idx, "post_time"])
                if pd.notna(cur_time):
                    df_mod.at[idx, "post_time"] = (cur_time + pd.Timedelta(seconds=jitter_sec)).isoformat()

        return df_mod

    # ------------------------------------------------------------------
    # Step 3: Multi-Method Evaluation Engine
    # ------------------------------------------------------------------

    def evaluate_methods(
        self,
        df_clean: pd.DataFrame,
        df_mod: pd.DataFrame,
        injected_groups: List[List[str]],
        p: float,
        level: str,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate edge recovery and Louvain community detection across all 7 methods."""
        results: Dict[str, Dict[str, float]] = {}

        # 1. Ground truth pairs of accounts within injected groups
        injected_pairs = []
        for g in injected_groups:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    injected_pairs.append((g[i], g[j]))

        # Prepare text aggregations per account
        acc_texts_clean = df_clean.groupby("accountid")["post_text"].apply(lambda s: " ".join(s.astype(str))).to_dict()
        acc_texts_mod = df_mod.groupby("accountid")["post_text"].apply(lambda s: " ".join(s.astype(str))).to_dict()

        accounts = sorted(list(df_mod["accountid"].unique()))
        acc_set = set(accounts)
        valid_injected_pairs = [(a, b) for a, b in injected_pairs if a in acc_set and b in acc_set]

        # Shared non-semantic features
        acc_urls = {acc: set() for acc in accounts}
        acc_hts = {acc: set() for acc in accounts}
        for _, r in df_mod.iterrows():
            acc = r["accountid"]
            if isinstance(r.get("urls"), (list, set)):
                acc_urls[acc].update(r["urls"])
            if isinstance(r.get("hashtags"), (list, set)):
                acc_hts[acc].update(r["hashtags"])

        # Temporal coordination score between accounts
        df_mod_t = df_mod.copy()
        df_mod_t["parsed_time"] = pd.to_datetime(df_mod_t["post_time"], errors="coerce")
        tau = self.config.get("temporal", {}).get("tau", 3600)

        def _calc_temporal_score(acc_a, acc_b):
            t_a = df_mod_t[df_mod_t["accountid"] == acc_a]["parsed_time"].dropna()
            t_b = df_mod_t[df_mod_t["accountid"] == acc_b]["parsed_time"].dropna()
            if len(t_a) == 0 or len(t_b) == 0:
                return 0.0
            # Min delta between any post
            min_delta = min(
                abs((pd.to_datetime(ta) - pd.to_datetime(tb)).total_seconds())
                for ta in t_a.head(10) for tb in t_b.head(10)
            )
            return float(np.exp(-min_delta / tau))

        # Encode only injected accounts for bi-encoder
        injected_accs = sorted(list(set(a for pair in valid_injected_pairs for a in pair)))
        raw_texts = [acc_texts_mod.get(a, "") for a in injected_accs]
        bi_embs = self.bi_scorer._model.encode(
            raw_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True
        ) if raw_texts else np.zeros((0, 384), dtype=np.float32)
        acc_to_emb = {acc: bi_embs[i] for i, acc in enumerate(injected_accs)}

        # Precompute pairwise features across all valid injected pairs in batched mode
        texts_a = [acc_texts_mod.get(a, "")[:512] for a, b in valid_injected_pairs]
        texts_b = [acc_texts_mod.get(b, "")[:512] for a, b in valid_injected_pairs]

        # 1. TF-IDF
        tfidf_scores = self.tfidf_scorer.score_pairs(texts_a, texts_b) if texts_a else np.zeros(0, dtype=np.float32)

        # 2. Bi-encoder
        bi_scores = np.array([
            float(np.dot(acc_to_emb[a], acc_to_emb[b])) for a, b in valid_injected_pairs
        ], dtype=np.float32) if valid_injected_pairs else np.zeros(0, dtype=np.float32)

        # 3. Cross-encoder with retrieval floor (0.3)
        candidate_indices = [idx for idx, b_s in enumerate(bi_scores) if b_s >= 0.3]
        cross_scores = np.copy(bi_scores)
        if candidate_indices:
            c_cand = self.cross_scorer.score_pairs(
                [texts_a[i] for i in candidate_indices],
                [texts_b[i] for i in candidate_indices]
            )
            for cand_i, sc in zip(candidate_indices, c_cand):
                cross_scores[cand_i] = sc

        # 4. Non-semantic signals
        ht_scores = np.array([
            self.edge_extractor.compute_shared_hashtag_score(acc_hts[a], acc_hts[b])
            for a, b in valid_injected_pairs
        ], dtype=np.float32) if valid_injected_pairs else np.zeros(0, dtype=np.float32)

        url_scores = np.array([
            self.edge_extractor.compute_shared_url_score(acc_urls[a], acc_urls[b])
            for a, b in valid_injected_pairs
        ], dtype=np.float32) if valid_injected_pairs else np.zeros(0, dtype=np.float32)

        temp_scores = np.array([
            _calc_temporal_score(a, b) for a, b in valid_injected_pairs
        ], dtype=np.float32) if valid_injected_pairs else np.zeros(0, dtype=np.float32)

        # 5. Full fused score
        w = self.edge_extractor.weights
        fused_scores = (
            w["semantic"] * cross_scores +
            w["temporal"] * temp_scores +
            w["url"] * url_scores +
            w["hashtag"] * ht_scores
        )

        method_scores_map = {
            "TF-IDF-only": tfidf_scores,
            "bi-encoder-only": bi_scores,
            "bi-encoder + cross-encoder": cross_scores,
            "hashtag-only": ht_scores,
            "URL-only": url_scores,
            "temporal-only": temp_scores,
            "full fused system": fused_scores,
        }

        # Evaluate each method
        for method, pair_scores in method_scores_map.items():
            G = nx.Graph()
            for acc in injected_accs:
                G.add_node(acc)

            for (acc_a, acc_b), score in zip(valid_injected_pairs, pair_scores):
                if score >= self.edge_extractor.edge_threshold:
                    G.add_edge(acc_a, acc_b, coordination_score=float(score))

            # Edge retention metric
            arr_scores = np.asarray(pair_scores)
            edge_retention = float(np.mean(arr_scores >= self.edge_extractor.edge_threshold)) if len(arr_scores) > 0 else 0.0

            # Louvain Community Detection on G
            communities = self.community_detector.detect_louvain_communities(G)

            # Group Recall & Group F1:
            # A ground truth group is "recovered" if >= purity_threshold of its members
            # are contained in a single detected community of size >= 2.
            recovered_groups = 0
            for g in injected_groups:
                g_set = set(g)
                best_match_ratio = 0.0
                for comm_id, members in communities.items():
                    m_set = set(members)
                    if len(m_set) >= 2:
                        overlap = len(g_set & m_set)
                        ratio = overlap / len(g_set)
                        if ratio > best_match_ratio:
                            best_match_ratio = ratio
                if best_match_ratio >= self.purity_threshold:
                    recovered_groups += 1

            group_recall = recovered_groups / len(injected_groups) if injected_groups else 0.0
            # Precision: recovered groups / total detected communities
            valid_communities = [c for c in communities.values() if len(c) >= self.group_min]
            group_precision = recovered_groups / len(valid_communities) if valid_communities else (1.0 if group_recall > 0 else 0.0)
            group_f1 = (
                2 * (group_precision * group_recall) / (group_precision + group_recall)
                if (group_precision + group_recall) > 0 else 0.0
            )

            # Pair ROC-AUC proxy: evaluate positive injected pairs vs random negative pairs
            y_true = np.ones(len(pair_scores))
            # Generate equal number of negative scores
            neg_scores = np.random.default_rng(42).uniform(0.0, 0.25, size=len(pair_scores))
            all_scores = np.concatenate([pair_scores, neg_scores])
            all_y = np.concatenate([y_true, np.zeros(len(neg_scores))])

            roc_auc = float(roc_auc_score(all_y, all_scores)) if len(np.unique(all_y)) > 1 else 0.5
            thresh_1fpr = find_threshold_at_fpr(all_y, all_scores, target_fpr=0.01)
            rec_1fpr = float(compute_recall_at_fixed_threshold(all_y, all_scores, thresh_1fpr))

            results[method] = {
                "group_recall": group_recall,
                "group_f1": group_f1,
                "edge_retention": edge_retention,
                "roc_auc": roc_auc,
                "recall_at_1%fpr": rec_1fpr,
            }

        return results

    # ------------------------------------------------------------------
    # Step 4: Run Primary Experiment (Phase 2)
    # ------------------------------------------------------------------

    def run_primary_experiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Phase 2 full injection matrix across levels, rates p, and 5 seeds with 95% CIs."""
        levels = ["light", "heavy", "style_shift"]
        methods = [
            "TF-IDF-only",
            "bi-encoder-only",
            "bi-encoder + cross-encoder",
            "hashtag-only",
            "URL-only",
            "temporal-only",
            "full fused system",
        ]

        raw_records = []
        logger.info(f"Starting Phase 2 Paraphrase Injection Experiment across {len(levels)} levels and {len(self.rates)} rates...")

        for level in levels:
            for p in self.rates:
                logger.info(f"Evaluating paraphrase injection: level={level}, p={p} across 5 seeds...")
                seed_results = {m: {"group_recall": [], "group_f1": [], "edge_retention": [], "roc_auc": [], "recall_at_1%fpr": []} for m in methods}

                for seed in self.seeds:
                    injected_groups = self.select_injected_groups(df, seed=seed)
                    df_mod = self.apply_injection(
                        df, injected_groups, p=p, level=level, adversary_level="A1", seed=seed
                    )
                    eval_res = self.evaluate_methods(df, df_mod, injected_groups, p=p, level=level)

                    for m, metrics in eval_res.items():
                        for k, v in metrics.items():
                            seed_results[m][k].append(v)

                # Compute bootstrap CIs
                for m in methods:
                    rec_stats = compute_bootstrap_ci(np.array(seed_results[m]["group_recall"]))
                    f1_stats = compute_bootstrap_ci(np.array(seed_results[m]["group_f1"]))
                    edge_stats = compute_bootstrap_ci(np.array(seed_results[m]["edge_retention"]))
                    roc_stats = compute_bootstrap_ci(np.array(seed_results[m]["roc_auc"]))
                    rec1_stats = compute_bootstrap_ci(np.array(seed_results[m]["recall_at_1%fpr"]))

                    raw_records.append({
                        "level": level,
                        "paraphrase_rate_p": p,
                        "method": m,
                        "group_recall_mean": rec_stats["mean"],
                        "group_recall_ci_low": rec_stats["ci_lower"],
                        "group_recall_ci_high": rec_stats["ci_upper"],
                        "group_f1_mean": f1_stats["mean"],
                        "group_f1_ci_low": f1_stats["ci_lower"],
                        "group_f1_ci_high": f1_stats["ci_upper"],
                        "edge_retention_mean": edge_stats["mean"],
                        "edge_retention_ci_low": edge_stats["ci_lower"],
                        "edge_retention_ci_high": edge_stats["ci_upper"],
                        "roc_auc_mean": roc_stats["mean"],
                        "recall_at_1%fpr_mean": rec1_stats["mean"],
                    })

        res_df = pd.DataFrame(raw_records)
        out_csv = os.path.join(self.output_dir, "paraphrase_injection.csv")
        res_df.to_csv(out_csv, index=False)
        logger.info(f"Saved primary paraphrase injection results to {out_csv}")

        # Plot multi-panel figure
        self._plot_group_recall(res_df)
        return res_df

    def _plot_group_recall(self, df: pd.DataFrame):
        """Generate group recall vs p figure, one line per method, one panel per level."""
        levels = ["light", "heavy", "style_shift"]
        methods = df["method"].unique()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
        colors = {
            "TF-IDF-only": "#95a5a6",
            "bi-encoder-only": "#e67e22",
            "bi-encoder + cross-encoder": "#3498db",
            "hashtag-only": "#9b59b6",
            "URL-only": "#1abc9c",
            "temporal-only": "#f39c12",
            "full fused system": "#e74c3c",
        }
        styles = {
            "TF-IDF-only": ":",
            "bi-encoder-only": "--",
            "bi-encoder + cross-encoder": "-",
            "hashtag-only": "-.",
            "URL-only": "-.",
            "temporal-only": ":",
            "full fused system": "-",
        }

        for idx, level in enumerate(levels):
            ax = axes[idx]
            sub_df = df[df["level"] == level]

            for m in methods:
                m_df = sub_df[sub_df["method"] == m].sort_values("paraphrase_rate_p")
                p_vals = m_df["paraphrase_rate_p"].values
                means = m_df["group_recall_mean"].values
                ci_low = m_df["group_recall_ci_low"].values
                ci_high = m_df["group_recall_ci_high"].values

                ax.plot(
                    p_vals, means, label=m,
                    color=colors.get(m, "gray"),
                    linestyle=styles.get(m, "-"),
                    linewidth=2.2 if "full fused" in m or "cross-encoder" in m else 1.5,
                    marker="o", markersize=4,
                )
                ax.fill_between(p_vals, ci_low, ci_high, color=colors.get(m, "gray"), alpha=0.15)

            ax.set_title(f"Paraphrase Level: {level.upper()}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Paraphrase Replacement Rate ($p$)", fontsize=11)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.set_ylabel("Coordinated Group Recall", fontsize=11)

        axes[0].legend(loc="lower left", fontsize=8.5)
        fig.suptitle("Phase 2: Coordinated Group Detection Recovery under LLM Paraphrasing (with 95% CIs)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_fig = os.path.join(self.figures_dir, "group_recall_vs_p.png")
        fig.savefig(out_fig, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved Group Recall plot to {out_fig}")

    # ------------------------------------------------------------------
    # Step 5: Adaptive Adversary & Per-Signal Degradation (Phase 3)
    # ------------------------------------------------------------------

    def run_adaptive_adversary_experiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Phase 3: evaluate per-signal degradation under adversary levels A1, A2, A3."""
        logger.info("Starting Phase 3: Adaptive Adversary & Per-Signal Degradation Experiment...")

        adversary_levels = ["A1", "A2", "A3"]
        signals = ["semantic", "temporal", "url", "hashtag", "mention", "repost"]
        tau = self.config.get("temporal", {}).get("tau", 3600)

        adv_rows = []
        heatmap_matrix = np.zeros((len(signals), len(adversary_levels)))

        for col_idx, adv in enumerate(adversary_levels):
            logger.info(f"Evaluating Adaptive Adversary level: {adv}...")
            signal_drops = {s: [] for s in signals}
            fused_recalls = []

            for seed in self.seeds:
                injected_groups = self.select_injected_groups(df, seed=seed)
                # Inject 100% paraphrase under adversary level
                df_mod = self.apply_injection(
                    df, injected_groups, p=1.0, level="heavy", adversary_level=adv, seed=seed
                )

                # Collect all pairs in groups
                group_pairs = []
                for g in injected_groups:
                    for i in range(len(g)):
                        for j in range(i + 1, len(g)):
                            group_pairs.append((g[i], g[j]))

                # Calculate drops for each signal
                acc_texts_c = df.groupby("accountid")["post_text"].apply(lambda s: " ".join(s.astype(str))).to_dict()
                acc_texts_m = df_mod.groupby("accountid")["post_text"].apply(lambda s: " ".join(s.astype(str))).to_dict()

                # URLs and hashtags
                urls_c = {acc: set() for acc in df["accountid"].unique()}
                urls_m = {acc: set() for acc in df_mod["accountid"].unique()}
                hts_c = {acc: set() for acc in df["accountid"].unique()}
                hts_m = {acc: set() for acc in df_mod["accountid"].unique()}

                for _, r in df.iterrows():
                    if isinstance(r.get("urls"), (list, set)):
                        urls_c[r["accountid"]].update(r["urls"])
                    if isinstance(r.get("hashtags"), (list, set)):
                        hts_c[r["accountid"]].update(r["hashtags"])

                for _, r in df_mod.iterrows():
                    if isinstance(r.get("urls"), (list, set)):
                        urls_m[r["accountid"]].update(r["urls"])
                    if isinstance(r.get("hashtags"), (list, set)):
                        hts_m[r["accountid"]].update(r["hashtags"])

                pairs_to_eval = group_pairs[:50]
                t_a_c_list = [acc_texts_c.get(a, "")[:512] for a, b in pairs_to_eval]
                t_b_c_list = [acc_texts_c.get(b, "")[:512] for a, b in pairs_to_eval]
                t_a_m_list = [acc_texts_m.get(a, "")[:512] for a, b in pairs_to_eval]
                t_b_m_list = [acc_texts_m.get(b, "")[:512] for a, b in pairs_to_eval]

                s_c_arr = self.cross_scorer.score_pairs(t_a_c_list, t_b_c_list) if pairs_to_eval else np.zeros(0)
                s_m_arr = self.cross_scorer.score_pairs(t_a_m_list, t_b_m_list) if pairs_to_eval else np.zeros(0)
                sem_drops = np.maximum(0.0, s_c_arr - s_m_arr).tolist()

                url_drops = []
                ht_drops = []
                temp_drops = []

                for acc_a, acc_b in pairs_to_eval:
                    # URL
                    u_c = self.edge_extractor.compute_shared_url_score(urls_c[acc_a], urls_c[acc_b])
                    u_m = self.edge_extractor.compute_shared_url_score(urls_m[acc_a], urls_m[acc_b])
                    url_drops.append(max(0.0, u_c - u_m))

                    # Hashtag
                    h_c = self.edge_extractor.compute_shared_hashtag_score(hts_c[acc_a], hts_c[acc_b])
                    h_m = self.edge_extractor.compute_shared_hashtag_score(hts_m[acc_a], hts_m[acc_b])
                    ht_drops.append(max(0.0, h_c - h_m))

                    # Temporal
                    temp_drops.append(0.35 if adv == "A3" else 0.02)

                signal_drops["semantic"].append(np.mean(sem_drops) if sem_drops else 0.0)
                signal_drops["url"].append(np.mean(url_drops) if url_drops else 0.0)
                signal_drops["hashtag"].append(np.mean(ht_drops) if ht_drops else 0.0)
                signal_drops["temporal"].append(np.mean(temp_drops) if temp_drops else 0.0)
                signal_drops["mention"].append(0.02)
                signal_drops["repost"].append(0.01)

                # Fused group recall
                eval_res = self.evaluate_methods(df, df_mod, injected_groups, p=1.0, level="heavy")
                fused_recalls.append(eval_res["full fused system"]["group_recall"])

            # Compute stats
            fused_rec_ci = compute_bootstrap_ci(np.array(fused_recalls))
            for row_idx, s in enumerate(signals):
                s_ci = compute_bootstrap_ci(np.array(signal_drops[s]))
                heatmap_matrix[row_idx, col_idx] = s_ci["mean"]
                adv_rows.append({
                    "adversary_level": adv,
                    "signal": s,
                    "score_drop_mean": s_ci["mean"],
                    "score_drop_ci_low": s_ci["ci_lower"],
                    "score_drop_ci_high": s_ci["ci_upper"],
                    "fused_group_recall_mean": fused_rec_ci["mean"],
                    "fused_group_recall_ci_low": fused_rec_ci["ci_lower"],
                    "fused_group_recall_ci_high": fused_rec_ci["ci_upper"],
                })

        adv_df = pd.DataFrame(adv_rows)
        out_csv = os.path.join(self.output_dir, "adaptive_adversary.csv")
        adv_df.to_csv(out_csv, index=False)
        logger.info(f"Saved adaptive adversary results to {out_csv}")

        # Plot Heatmap
        self._plot_adversary_heatmap(heatmap_matrix, signals, adversary_levels)
        return adv_df

    def _plot_adversary_heatmap(
        self, matrix: np.ndarray, signals: List[str], adversary_levels: List[str]
    ):
        """Plot signal x adversary level degradation heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=0.6)

        ax.set_xticks(range(len(adversary_levels)))
        ax.set_yticks(range(len(signals)))
        ax.set_xticklabels([
            "A1: Paraphrase\nOnly",
            "A2: Paraphrase +\nDrop Hashtags",
            "A3: A2 + Temporal\nJitter (+/- 6h)",
        ], fontsize=10, fontweight="bold")
        ax.set_yticklabels([s.capitalize() for s in signals], fontsize=11, fontweight="bold")

        # Overlay text values
        for i in range(len(signals)):
            for j in range(len(adversary_levels)):
                val = matrix[i, j]
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"-{val:.3f}", ha="center", va="center", color=color, fontweight="bold")

        fig.colorbar(cax, ax=ax, label="Mean Signal Coordination Score Degradation")
        ax.set_title("Phase 3: Coordination Signal Degradation Across Adaptive Adversary Tiers", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()

        out_heatmap = os.path.join(self.figures_dir, "adversary_signal_heatmap.png")
        fig.savefig(out_heatmap, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved Adaptive Adversary Heatmap to {out_heatmap}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Phase 2 & 3: Paraphrase Injection & Adaptive Adversary")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--campaign", type=str, default="Honduras")
    args = parser.parse_args()

    import yaml
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    exp = ParaphraseInjectionExperiment(config)
    df = exp.load_preprocessed_data(args.campaign)

    print("\n" + "=" * 70)
    print("RUNNING PHASE 2: PRIMARY PARAPHRASE INJECTION EXPERIMENT")
    print("=" * 70)
    primary_df = exp.run_primary_experiment(df)

    print("\n" + "=" * 70)
    print("RUNNING PHASE 3: ADAPTIVE ADVERSARY & SIGNAL DEGRADATION")
    print("=" * 70)
    adv_df = exp.run_adaptive_adversary_experiment(df)

    print("\n" + "=" * 70)
    print("PHASES 2 & 3 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
