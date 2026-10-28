"""
Tests for Paraphrase Robustness and Cross-Campaign Transfer Experiments.
"""

import os
import pytest
import pandas as pd
import numpy as np


class TestExperimentSuite:
    """Test the newly added experiment modules."""

    def test_paraphrase_summary_exists_and_valid(self):
        summary_path = os.path.join("results", "paraphrase_robustness_summary.csv")
        assert os.path.exists(summary_path), "Paraphrase summary file missing"
        df = pd.read_csv(summary_path)
        metrics = dict(zip(df["metric"], df["value"]))

        assert "mean_biencoder_original" in metrics
        assert "mean_crossencoder_original" in metrics
        assert "biencoder_score_drop" in metrics
        assert "crossencoder_score_drop" in metrics
        assert metrics["biencoder_score_drop"] > metrics["crossencoder_score_drop"]

    def test_cross_campaign_results_exist_and_valid(self):
        results_path = os.path.join("results", "cross_campaign_results.csv")
        assert os.path.exists(results_path), "Cross campaign results missing"
        df = pd.read_csv(results_path)

        assert "model" in df.columns
        assert "source_accuracy" in df.columns
        assert "target_accuracy" in df.columns
        assert "accuracy_drop" in df.columns

        # Verify that Random Forest suffers a larger accuracy drop than GraphSAGE
        rf_drop = df[df["model"].str.contains("Random Forest")]["accuracy_drop"].values[0]
        sage_drop = df[df["model"].str.contains("GraphSAGE")]["accuracy_drop"].values[0]
        assert rf_drop > sage_drop, f"Expected RF drop ({rf_drop}) > GraphSAGE drop ({sage_drop})"

    def test_figures_exist(self):
        fig_paths = [
            os.path.join("results", "figures", "paraphrase_robustness.png"),
            os.path.join("results", "figures", "dist_semantic_similarity.png"),
            os.path.join("results", "figures", "cross_campaign_transfer.png"),
        ]
        for p in fig_paths:
            assert os.path.exists(p), f"Figure not found: {p}"
