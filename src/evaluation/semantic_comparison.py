"""
src/evaluation/semantic_comparison.py

Phase 1: Semantic-component comparison with independent negatives.

Compares:
1. TF-IDF Cosine
2. MiniLM Bi-Encoder (all-MiniLM-L6-v2)
3. Multilingual Bi-Encoder (paraphrase-multilingual-MiniLM-L12-v2)
4. NLI Cross-Encoder (cross-encoder/nli-deberta-v3-base / mDeBERTa)
5. Paraphrase-Trained Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)

Protocol:
- Positive pairs are chosen strictly by NON-semantic coordination signals (URLs, hashtags, reposts, temporal bursts).
- Hard negative pairs share a hashtag or topical vocabulary but have ZERO coordination signals (time gap > 24h).
- Evaluated under both 1:1 and 1:5 positive:negative balance.
- Thresholds for 1% and 5% FPR are chosen strictly on a held-out calibration split, NEVER on the test pairs.
- Every metric reports mean and 95% bootstrap CI across 5 random seeds (42, 43, 44, 45, 46).
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from src.data.pair_sampling import IndependentPairSampler
from src.data.loader import DataLoader, create_sample_dataset
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_bootstrap_ci, find_threshold_at_fpr, compute_recall_at_fixed_threshold
from src.nlp.similarity import (
    BaseSemanticScorer,
    TfidfScorer,
    BiEncoderScorer,
    CrossEncoderScorer,
)

logger = logging.getLogger(__name__)


class SemanticComparisonExperiment:
    """Run semantic-component comparison benchmark on independent pairs."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = os.path.join("results", "paper")
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        self.seeds = [42, 43, 44, 45, 46]
        self.n_positives = config.get("semantic_experiment", {}).get("n_positives", 300)

    def load_preprocessed_data(self, campaign: str = "Honduras") -> pd.DataFrame:
        """Load and preprocess campaign data."""
        logger.info(f"Loading campaign data for {campaign}...")
        data_loader = DataLoader(self.config)
        try:
            df = data_loader.load_campaign(campaign)
        except Exception as e:
            logger.warning(f"Could not load {campaign} via loader ({e}); creating sample...")
            create_sample_dataset(self.config["data"]["raw_dir"])
            df = data_loader.load_campaign()

        preprocessor = Preprocessor(self.config)
        df = preprocessor.process(df)
        return df

    def get_scorers(self) -> Dict[str, BaseSemanticScorer]:
        """Instantiate all 5 semantic scoring components."""
        scorers = {
            "TF-IDF Cosine (Lexical)": TfidfScorer(),
            "MiniLM Bi-Encoder (English)": BiEncoderScorer(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
            "Multilingual Bi-Encoder (MiniLM-L12)": BiEncoderScorer(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            "NLI Cross-Encoder": CrossEncoderScorer(
                model_name="cross-encoder/nli-deberta-v3-base"
            ),
            "Paraphrase Cross-Encoder (MiniLM-L6)": CrossEncoderScorer(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
        }
        return scorers

    def run_benchmark(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the benchmark across 5 seeds and both 1:1 and 1:5 ratios."""
        sampler = IndependentPairSampler(self.config)
        scorers = self.get_scorers()

        results_rows = []
        roc_plot_data = {}  # for plotting curve under seed 42

        for ratio in [1.0, 5.0]:
            ratio_label = "1:1 Balanced" if ratio == 1.0 else "1:5 Imbalanced"
            logger.info(f"\n{'='*60}\nRunning Semantic Benchmark with ratio {ratio_label}\n{'='*60}")

            for scorer_name, scorer in scorers.items():
                logger.info(f"Evaluating scorer: {scorer_name}...")

                seed_roc_aucs = []
                seed_pr_aucs = []
                seed_recalls_at_1fpr = []
                seed_recalls_at_5fpr = []

                for seed in self.seeds:
                    # 1. Sample independent dataset
                    pairs_df = sampler.build_pair_dataset(
                        df, n_positives=self.n_positives, imbalance_ratio=ratio, seed=seed
                    )

                    # 2. Split into 50% calibration (train/val) and 50% test
                    n_pairs = len(pairs_df)
                    cal_n = n_pairs // 2
                    cal_df = pairs_df.iloc[:cal_n].copy()
                    test_df = pairs_df.iloc[cal_n:].copy()

                    # 3. Score calibration and test pairs
                    cal_scores = scorer.score_pairs(
                        cal_df["text_a"].tolist(), cal_df["text_b"].tolist()
                    )
                    test_scores = scorer.score_pairs(
                        test_df["text_a"].tolist(), test_df["text_b"].tolist()
                    )

                    y_cal = cal_df["label"].values
                    y_test = test_df["label"].values

                    # 4. Pick decision thresholds on calibration split (held-out)
                    thresh_1fpr = find_threshold_at_fpr(y_cal, cal_scores, target_fpr=0.01)
                    thresh_5fpr = find_threshold_at_fpr(y_cal, cal_scores, target_fpr=0.05)

                    # 5. Evaluate on test split
                    roc_auc = roc_auc_score(y_test, test_scores)
                    pr_auc = average_precision_score(y_test, test_scores)
                    rec_1fpr = compute_recall_at_fixed_threshold(y_test, test_scores, thresh_1fpr)
                    rec_5fpr = compute_recall_at_fixed_threshold(y_test, test_scores, thresh_5fpr)

                    seed_roc_aucs.append(roc_auc)
                    seed_pr_aucs.append(pr_auc)
                    seed_recalls_at_1fpr.append(rec_1fpr)
                    seed_recalls_at_5fpr.append(rec_5fpr)

                    # Stash curve data for seed 42
                    if seed == 42 and ratio == 1.0:
                        roc_plot_data[scorer_name] = (y_test, test_scores)

                # Compute bootstrap CI across 5 seeds
                def _calc_stats(arr):
                    arr = np.array(arr)
                    ci_res = compute_bootstrap_ci(arr, arr, lambda a, b: float(np.mean(a)), n_bootstraps=1000)
                    return ci_res["mean"], ci_res["ci_lower"], ci_res["ci_upper"]

                roc_mean, roc_lo, roc_hi = _calc_stats(seed_roc_aucs)
                pr_mean, pr_lo, pr_hi = _calc_stats(seed_pr_aucs)
                rec1_mean, rec1_lo, rec1_hi = _calc_stats(seed_recalls_at_1fpr)
                rec5_mean, rec5_lo, rec5_hi = _calc_stats(seed_recalls_at_5fpr)

                results_rows.append({
                    "ratio": ratio_label,
                    "model": scorer_name,
                    "roc_auc_mean": roc_mean,
                    "roc_auc_95ci": f"[{roc_lo:.4f}, {roc_hi:.4f}]",
                    "pr_auc_mean": pr_mean,
                    "pr_auc_95ci": f"[{pr_lo:.4f}, {pr_hi:.4f}]",
                    "recall_at_1%fpr_mean": rec1_mean,
                    "recall_at_1%fpr_95ci": f"[{rec1_lo:.4f}, {rec1_hi:.4f}]",
                    "recall_at_5%fpr_mean": rec5_mean,
                    "recall_at_5%fpr_95ci": f"[{rec5_lo:.4f}, {rec5_hi:.4f}]",
                })

        res_df = pd.DataFrame(results_rows)
        out_csv = os.path.join(self.output_dir, "semantic_component_comparison.csv")
        res_df.to_csv(out_csv, index=False)
        logger.info(f"Saved semantic component results to {out_csv}")

        # --------------------------------------------------------------
        # Plot ROC and PR Curves for Seed 42
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        ax_roc = axes[0]
        ax_pr = axes[1]

        colors = ["#7f8c8d", "#3498db", "#2980b9", "#e67e22", "#27ae60"]

        for (name, (y_true, scores)), color in zip(roc_plot_data.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

            prec, rec, _ = precision_recall_curve(y_true, scores)
            prauc = average_precision_score(y_true, scores)
            ax_pr.plot(rec, prec, label=f"{name} (PR-AUC={prauc:.3f})", color=color, linewidth=2)

        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax_roc.set_title("ROC Curves: Independent Coordination Pair Matching", fontsize=11, fontweight="bold")
        ax_roc.legend(loc="lower right", fontsize=8.5)
        ax_roc.grid(True, alpha=0.3)

        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel("Recall", fontsize=11)
        ax_pr.set_ylabel("Precision", fontsize=11)
        ax_pr.set_title("PR Curves: Independent Coordination Pair Matching", fontsize=11, fontweight="bold")
        ax_pr.legend(loc="lower left", fontsize=8.5)
        ax_pr.grid(True, alpha=0.3)

        fig.suptitle("Phase 1: Semantic Component Performance on Non-Semantic Coordinated Ground Truth", fontsize=13, fontweight="bold")
        plt.tight_layout()

        out_plot = os.path.join(self.figures_dir, "semantic_component_roc_pr.png")
        fig.savefig(out_plot, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved ROC/PR figure to {out_plot}")

        return res_df


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Semantic Component Comparison")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--campaign", type=str, default="Honduras")
    parser.add_argument("--n-positives", type=int, default=250)
    args = parser.parse_args()

    import yaml
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)

    if "semantic_experiment" not in config:
        config["semantic_experiment"] = {}
    config["semantic_experiment"]["n_positives"] = args.n_positives

    exp = SemanticComparisonExperiment(config)
    df = exp.load_preprocessed_data(args.campaign)
    res_df = exp.run_benchmark(df)

    print("\n" + "=" * 70)
    print("PHASE 1: SEMANTIC COMPONENT COMPARISON RESULTS (WITH 95% BOOTSTRAP CIs)")
    print("=" * 70)
    print(res_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
