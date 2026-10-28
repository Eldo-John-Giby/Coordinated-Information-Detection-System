"""
src/data/pair_sampling.py

Independent pair sampling for semantic-component evaluation (Phase 1).

Ground truth positive pairs are selected strictly using non-semantic signals:
- Shared URL (set intersection >= 1)
- Shared hashtag sequence (>= 2 matching hashtags)
- Shared repost target (retweeting the same account)
- Temporal burst window (|t_A - t_B| <= 300s)
NO embedding similarity is used to select positives.

Hard negative pairs are selected from non-coordinated accounts (organic controls)
that share a hashtag or topical vocabulary but have ZERO coordination signals:
- Time gap > 24 hours
- Disjoint URLs
- No shared repost target
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class IndependentPairSampler:
    """Sample independent positive and hard negative post pairs without semantic leakage."""

    def __init__(self, config: dict):
        self.config = config
        self.random_seed = config.get("data", {}).get("random_seed", 42)
        self.burst_window = config.get("temporal", {}).get("burst_window", 300)

    def sample_positive_pairs(
        self, df: pd.DataFrame, max_pairs: int = 1000, seed: int = 42
    ) -> List[Dict]:
        """Sample post pairs from distinct IO accounts sharing at least one non-semantic signal.

        Signals checked:
        1. Shared URL
        2. Shared hashtags (>= 2 common hashtags)
        3. Shared repost target
        4. Burst window (|t_A - t_B| <= burst_window, default 300s)
        """
        rng = np.random.default_rng(seed)

        # Filter to IO posts only
        if "is_control" in df.columns:
            io_df = df[~df["is_control"].astype(bool)].copy()
        elif "account_is_control" in df.columns:
            io_df = df[~df["account_is_control"].astype(bool)].copy()
        else:
            io_df = df.copy()

        # Ensure valid text
        io_df = io_df[io_df["post_text"].astype(str).str.len() >= 10].copy()
        if len(io_df) < 2:
            logger.warning("Insufficient IO posts for positive pair sampling")
            return []

        # Sort by post_time for efficient temporal search
        io_df["parsed_time"] = pd.to_datetime(io_df["post_time"], errors="coerce")
        io_df = io_df.dropna(subset=["parsed_time"]).sort_values("parsed_time").reset_index(drop=True)

        found_pairs = []
        seen_pairs: Set[Tuple[str, str]] = set()

        # 1. Temporal Burst Positives
        times = io_df["parsed_time"].values
        accounts = io_df["accountid"].astype(str).values
        postids = io_df["postid"].astype(str).values
        texts = io_df["post_text"].astype(str).values

        n = len(io_df)
        burst_ns = int(self.burst_window * 1e9)  # nanoseconds

        # Shuffle indices to avoid bias towards earliest posts
        shuffled_indices = rng.permutation(n)

        for i in shuffled_indices:
            if len(found_pairs) >= max_pairs:
                break
            t_i = times[i]
            acc_i = accounts[i]

            # Search ahead in time window
            for j in range(i + 1, min(i + 50, n)):
                delta_t = int(times[j] - t_i)
                if delta_t > burst_ns:
                    break
                acc_j = accounts[j]
                if acc_i == acc_j or texts[i] == texts[j]:
                    continue

                pair_key = (min(postids[i], postids[j]), max(postids[i], postids[j]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    found_pairs.append({
                        "post_a": postids[i],
                        "post_b": postids[j],
                        "account_a": acc_i,
                        "account_b": acc_j,
                        "text_a": texts[i],
                        "text_b": texts[j],
                        "signal": "temporal_burst",
                        "label": 1,
                    })

        # 2. Shared URL Positives
        if len(found_pairs) < max_pairs and "url_list" in io_df.columns:
            url_to_idx: Dict[str, List[int]] = {}
            for idx, urls in enumerate(io_df["url_list"]):
                if isinstance(urls, (list, set, tuple)):
                    for u in urls:
                        if u and len(u) > 3:
                            url_to_idx.setdefault(u, []).append(idx)

            for u, idxs in url_to_idx.items():
                if len(found_pairs) >= max_pairs:
                    break
                if len(idxs) >= 2:
                    for _ in range(min(5, len(idxs))):
                        i, j = rng.choice(idxs, size=2, replace=False)
                        if accounts[i] == accounts[j] or texts[i] == texts[j]:
                            continue
                        pair_key = (min(postids[i], postids[j]), max(postids[i], postids[j]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            found_pairs.append({
                                "post_a": postids[i],
                                "post_b": postids[j],
                                "account_a": accounts[i],
                                "account_b": accounts[j],
                                "text_a": texts[i],
                                "text_b": texts[j],
                                "signal": "shared_url",
                                "label": 1,
                            })

        # 3. Shared Hashtag Sequence (>= 2 common hashtags)
        if len(found_pairs) < max_pairs and "hashtag_list" in io_df.columns:
            ht_to_idx: Dict[str, List[int]] = {}
            for idx, hts in enumerate(io_df["hashtag_list"]):
                if isinstance(hts, (list, set, tuple)) and len(hts) >= 2:
                    ht_key = "_".join(sorted(hts[:2]))
                    ht_to_idx.setdefault(ht_key, []).append(idx)

            for ht_key, idxs in ht_to_idx.items():
                if len(found_pairs) >= max_pairs:
                    break
                if len(idxs) >= 2:
                    for _ in range(min(4, len(idxs))):
                        i, j = rng.choice(idxs, size=2, replace=False)
                        if accounts[i] == accounts[j] or texts[i] == texts[j]:
                            continue
                        pair_key = (min(postids[i], postids[j]), max(postids[i], postids[j]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            found_pairs.append({
                                "post_a": postids[i],
                                "post_b": postids[j],
                                "account_a": accounts[i],
                                "account_b": accounts[j],
                                "text_a": texts[i],
                                "text_b": texts[j],
                                "signal": "shared_hashtags",
                                "label": 1,
                            })

        # 4. Shared Repost Target Positives
        if len(found_pairs) < max_pairs and "repost_of_account" in io_df.columns:
            repost_to_idx: Dict[str, List[int]] = {}
            for idx, r_acc in enumerate(io_df["repost_of_account"]):
                if pd.notna(r_acc) and r_acc:
                    repost_to_idx.setdefault(str(r_acc), []).append(idx)

            for r_acc, idxs in repost_to_idx.items():
                if len(found_pairs) >= max_pairs:
                    break
                if len(idxs) >= 2:
                    for _ in range(min(5, len(idxs))):
                        i, j = rng.choice(idxs, size=2, replace=False)
                        if accounts[i] == accounts[j] or texts[i] == texts[j]:
                            continue
                        pair_key = (min(postids[i], postids[j]), max(postids[i], postids[j]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            found_pairs.append({
                                "post_a": postids[i],
                                "post_b": postids[j],
                                "account_a": accounts[i],
                                "account_b": accounts[j],
                                "text_a": texts[i],
                                "text_b": texts[j],
                                "signal": "shared_repost",
                                "label": 1,
                            })

        logger.info(f"Sampled {len(found_pairs)} independent non-semantic positive pairs")
        return found_pairs[:max_pairs]

    def sample_hard_negative_pairs(
        self,
        df: pd.DataFrame,
        n_negatives: int,
        seen_positive_keys: Optional[Set[Tuple[str, str]]] = None,
        seed: int = 42,
    ) -> List[Dict]:
        """Sample hard negative pairs sharing topical keywords/hashtags but NO coordination signal.

        Constraints:
        - Distinct accounts (control accounts strongly preferred)
        - Time gap > 24 hours (86,400s)
        - Disjoint URLs
        - No shared repost targets
        """
        rng = np.random.default_rng(seed)
        seen_keys = set(seen_positive_keys or [])

        # Organic control accounts preferred for negatives
        if "is_control" in df.columns:
            ctrl_df = df[df["is_control"].astype(bool)].copy()
        elif "account_is_control" in df.columns:
            ctrl_df = df[df["account_is_control"].astype(bool)].copy()
        else:
            ctrl_df = df.copy()

        if len(ctrl_df) < 50:
            ctrl_df = df.copy()

        ctrl_df = ctrl_df[ctrl_df["post_text"].astype(str).str.len() >= 10].copy()
        ctrl_df["parsed_time"] = pd.to_datetime(ctrl_df["post_time"], errors="coerce")
        ctrl_df = ctrl_df.dropna(subset=["parsed_time"]).reset_index(drop=True)

        texts = ctrl_df["post_text"].astype(str).values
        postids = ctrl_df["postid"].astype(str).values
        accounts = ctrl_df["accountid"].astype(str).values
        times = ctrl_df["parsed_time"].values
        n = len(ctrl_df)

        neg_pairs = []

        # Find hashtag groups in control accounts
        ht_map: Dict[str, List[int]] = {}
        if "hashtag_list" in ctrl_df.columns:
            for i, hts in enumerate(ctrl_df["hashtag_list"]):
                if isinstance(hts, (list, set, tuple)):
                    for h in hts:
                        if h and len(h) > 2:
                            ht_map.setdefault(h, []).append(i)

        # 1. Hard negatives sharing a hashtag but separated in time (> 24 hours)
        one_day_ns = int(86400 * 1e9)
        popular_hts = [h for h, idxs in ht_map.items() if len(idxs) >= 4]

        for h in popular_hts:
            if len(neg_pairs) >= n_negatives:
                break
            idxs = ht_map[h]
            for _ in range(min(10, len(idxs))):
                i, j = rng.choice(idxs, size=2, replace=False)
                if accounts[i] == accounts[j] or texts[i] == texts[j]:
                    continue
                # Enforce time gap > 24 hours
                if abs(int(times[i] - times[j])) <= one_day_ns:
                    continue

                pair_key = (min(postids[i], postids[j]), max(postids[i], postids[j]))
                if pair_key not in seen_keys:
                    seen_keys.add(pair_key)
                    neg_pairs.append({
                        "post_a": postids[i],
                        "post_b": postids[j],
                        "account_a": accounts[i],
                        "account_b": accounts[j],
                        "text_a": texts[i],
                        "text_b": texts[j],
                        "signal": "hard_negative_hashtag",
                        "label": 0,
                    })

        # 2. Random cross-account negatives with time separation
        attempts = 0
        max_attempts = n_negatives * 5
        while len(neg_pairs) < n_negatives and attempts < max_attempts:
            attempts += 1
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j or accounts[i] == accounts[j] or texts[i] == texts[j]:
                continue
            if abs(int(times[i] - times[j])) <= one_day_ns:
                continue

            pair_key = (min(postids[i], postids[j]), max(postids[i], postids[j]))
            if pair_key not in seen_keys:
                seen_keys.add(pair_key)
                neg_pairs.append({
                    "post_a": postids[i],
                    "post_b": postids[j],
                    "account_a": accounts[i],
                    "account_b": accounts[j],
                    "text_a": texts[i],
                    "text_b": texts[j],
                    "signal": "hard_negative_cross",
                    "label": 0,
                })

        logger.info(f"Sampled {len(neg_pairs)} hard negative pairs")
        return neg_pairs[:n_negatives]

    def build_pair_dataset(
        self,
        df: pd.DataFrame,
        n_positives: int = 500,
        imbalance_ratio: float = 1.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Build balanced (1:1) or imbalanced (1:5) evaluation dataset of independent pairs."""
        positives = self.sample_positive_pairs(df, max_pairs=n_positives, seed=seed)
        n_pos = len(positives)
        n_neg = int(n_pos * imbalance_ratio)

        pos_keys = {(min(p["post_a"], p["post_b"]), max(p["post_a"], p["post_b"])) for p in positives}
        negatives = self.sample_hard_negative_pairs(
            df, n_negatives=n_neg, seen_positive_keys=pos_keys, seed=seed + 100
        )

        all_pairs = positives + negatives
        dataset_df = pd.DataFrame(all_pairs)
        # Deterministic shuffle
        dataset_df = dataset_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return dataset_df
