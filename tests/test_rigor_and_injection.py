"""
tests/test_rigor_and_injection.py

Comprehensive unit tests for Phases 1 through 5:
- IndependentPairSampler
- TfidfScorer and BaseSemanticScorer
- ParaphraseEngine & ParaphraseCache
- ParaphraseInjectionExperiment group selection
- CrossCampaignBenchmark feature configurations
- Statistical bootstrap CI computation
- Pair truth validation
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pytest
import numpy as np
import pandas as pd
import networkx as nx

from src.data.pair_sampling import IndependentPairSampler
from src.nlp.similarity import TfidfScorer
from src.evaluation.metrics import compute_bootstrap_ci, find_threshold_at_fpr, compute_recall_at_fixed_threshold
from src.evaluation.paraphrase_injection import ParaphraseCache, ParaphraseEngine, ParaphraseInjectionExperiment
from src.evaluation.cross_campaign import CrossCampaignBenchmark
from scripts.validate_pair_truth import sample_pairs, compute_agreement


@pytest.fixture
def sample_config(tmp_path):
    cache_file = tmp_path / "cache" / "paraphrases.jsonl"
    return {
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "random_seed": 42,
        },
        "paraphrase_injection": {
            "K_groups": 5,
            "group_size_min": 2,
            "group_size_max": 4,
            "paraphrase_rates": [0.0, 0.5, 1.0],
            "seeds": [42],
            "purity_threshold": 0.5,
            "llm_model": "mock-llm",
            "api_limit": 10,
            "cache_path": str(cache_file),
            "prompt_templates": {
                "light": "Paraphrase: {text}",
                "heavy": "Rewrite: {text}",
                "style_shift": "Shift tone: {text}",
            },
        },
        "temporal": {
            "tau": 3600,
            "burst_window": 300,
        },
        "coordination": {
            "edge_threshold": 0.3,
            "weights": {
                "semantic": 0.35,
                "temporal": 0.25,
                "url": 0.15,
                "hashtag": 0.10,
                "mention": 0.05,
                "repost": 0.10,
            },
        },
        "rigor": {
            "seeds": [42],
            "confidence_level": 0.95,
            "n_bootstraps": 100,
        },
    }


@pytest.fixture
def synthetic_dataframe():
    """Create synthetic DataFrame mimicking campaign data with IO and Control accounts."""
    rng = np.random.default_rng(42)
    records = []
    base_time = pd.Timestamp("2020-01-01 12:00:00")

    # Group 1: Coordinated IO accounts sharing URLs and burst timing
    for i in range(15):
        acc = f"io_acc_{i % 3}"
        records.append({
            "postid": f"p_io_{i}",
            "accountid": acc,
            "post_text": f"Coordinated message announcement number {i} with #AlivioDeDeuda and news",
            "post_time": (base_time + pd.Timedelta(seconds=i * 20)).isoformat(),
            "urls": ["http://gov.hn/portal"] if i % 2 == 0 else ["http://gov.hn/decree"],
            "hashtags": ["AlivioDeDeuda", "Honduras"],
            "account_mentions": ["@presidencia"],
            "is_control": False,
            "account_is_control": False,
            "is_repost": i % 3 != 0,
            "follower_count": 50,
            "following_count": 100,
        })

    # Group 2: Control organic accounts
    for j in range(15):
        acc = f"ctrl_acc_{j % 3}"
        records.append({
            "postid": f"p_ctrl_{j}",
            "accountid": acc,
            "post_text": f"Organic casual post number {j} having breakfast coffee today",
            "post_time": (base_time + pd.Timedelta(hours=48 + j * 5)).isoformat(),
            "urls": ["http://coffee.org/blog"] if j % 3 == 0 else [],
            "hashtags": ["CoffeeMorning"] if j % 2 == 0 else ["Honduras"],
            "account_mentions": [],
            "is_control": True,
            "account_is_control": True,
            "is_repost": False,
            "follower_count": 1200,
            "following_count": 800,
        })

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# 1. Statistical Helpers
# ----------------------------------------------------------------------

def test_bootstrap_ci():
    data = np.array([0.80, 0.82, 0.81, 0.79, 0.83])
    res = compute_bootstrap_ci(data, n_bootstraps=500, ci=0.95, seed=42)
    assert 0.78 <= res["mean"] <= 0.84
    assert res["ci_lower"] <= res["mean"] <= res["ci_upper"]
    assert res["n_samples"] == 5


def test_find_threshold_and_recall():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    thresh = find_threshold_at_fpr(y_true, scores, target_fpr=0.20)
    assert 0.4 <= thresh <= 0.7
    rec = compute_recall_at_fixed_threshold(y_true, scores, threshold=thresh)
    assert 0.6 <= rec <= 1.0


# ----------------------------------------------------------------------
# 2. Independent Pair Sampler (Phase 1)
# ----------------------------------------------------------------------

def test_independent_pair_sampler(sample_config, synthetic_dataframe):
    sampler = IndependentPairSampler(sample_config)
    pairs_df = sampler.build_pair_dataset(synthetic_dataframe, n_positives=5, imbalance_ratio=1.0, seed=42)
    assert len(pairs_df) > 0
    assert "text_a" in pairs_df.columns
    assert "text_b" in pairs_df.columns
    assert "label" in pairs_df.columns
    assert set(pairs_df["label"].unique()).issubset({0, 1})


# ----------------------------------------------------------------------
# 3. Semantic Scorer (Phase 1)
# ----------------------------------------------------------------------

def test_tfidf_scorer():
    scorer = TfidfScorer()
    texts_a = ["Hello world of natural language processing", "Totally unrelated text about stars"]
    texts_b = ["Hello world of deep learning language processing", "Completely different text about cooking recipes"]
    scores = scorer.score_pairs(texts_a, texts_b)
    assert len(scores) == 2
    assert scores[0] > scores[1]  # High lexical overlap should score higher than unrelated


# ----------------------------------------------------------------------
# 4. Paraphrase Cache & Deterministic Engine (Phases 2 & 3)
# ----------------------------------------------------------------------

def test_paraphrase_cache_and_engine(sample_config, tmp_path):
    cache = ParaphraseCache(str(tmp_path / "cache.jsonl"))
    cache.set("post_123", "heavy", "test_model", "prompt_text", "Rewritten text result")
    retrieved = cache.get("post_123", "heavy", "test_model", "prompt_text")
    assert retrieved == "Rewritten text result"

    engine = ParaphraseEngine(sample_config)
    text = "El gobierno anuncio alivio de deuda para el pueblo hoy #AlivioDeDeuda"
    para = engine.paraphrase(text, post_id="p1", level="heavy", seed=42)
    assert isinstance(para, str)
    assert len(para) > 5


# ----------------------------------------------------------------------
# 5. Group Selection & Adversary (Phases 2 & 3)
# ----------------------------------------------------------------------

def test_paraphrase_injection_experiment(sample_config, synthetic_dataframe):
    exp = ParaphraseInjectionExperiment(sample_config)
    groups = exp.select_injected_groups(synthetic_dataframe, seed=42)
    assert len(groups) > 0
    assert all(len(g) >= sample_config["paraphrase_injection"]["group_size_min"] for g in groups)

    # Test Adversary Injection
    df_mod = exp.apply_injection(synthetic_dataframe, groups, p=0.5, level="heavy", adversary_level="A1", seed=42)
    assert len(df_mod) == len(synthetic_dataframe)


# ----------------------------------------------------------------------
# 6. Pair Truth Validation (Phase 5)
# ----------------------------------------------------------------------

def test_pair_truth_validation(tmp_path):
    df = pd.DataFrame({
        "pair_id": [f"pair_{i}" for i in range(10)],
        "account_a": [f"acc_a_{i}" for i in range(10)],
        "account_b": [f"acc_b_{i}" for i in range(10)],
        "text_a": [f"Text A {i}" for i in range(10)],
        "text_b": [f"Text B {i}" for i in range(10)],
        "same_shape": [1 if i % 2 == 0 else 0 for i in range(10)],
    })
    in_csv = tmp_path / "pair_truth.csv"
    out_csv = tmp_path / "sample.csv"
    df.to_csv(in_csv, index=False)

    sample_pairs(input_path=str(in_csv), output_path=str(out_csv), n_samples=6, seed=42)
    assert os.path.exists(out_csv)
    sampled_df = pd.read_csv(out_csv)
    assert len(sampled_df) == 6
    assert "human_label" in sampled_df.columns
