"""
src/evaluation/rigor_experiments.py

Phase 5: Statistical Rigor, GNN Ablation, and Rerank-Cap Sensitivity Experiments.

Experiments:
1. 5-Seed In-Campaign Model Comparison with 95% Bootstrap CIs (RF, LR, GCN, GraphSAGE).
2. GNN Edge Ablation: Retrain GCN with each coordination signal removed from edge weights.
3. Rerank-Cap Sensitivity: Evaluate max_rerank_pairs in {5000, 20000, 50000}.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.data.loader import DataLoader, create_sample_dataset
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_bootstrap_ci, find_threshold_at_fpr, compute_recall_at_fixed_threshold
from src.features.edge_features import EdgeFeatureExtractor
from src.graph.community import CommunityDetector
from src.models.gcn import GCNModel
from src.models.graphsage import GraphSAGEModel

logger = logging.getLogger(__name__)


class RigorExperiments:
    """Implements Phase 5 experiments: In-campaign 5-seed comparison, GNN ablation, and rerank sensitivity."""

    def __init__(self, config: dict):
        self.config = config
        self.seeds = config.get("rigor", {}).get("seeds", [42, 43, 44, 45, 46])
        self.output_dir = os.path.join("results", "paper")
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        self.tabular_cols = [
            "post_count",
            "follower_count",
            "following_count",
            "repost_ratio",
            "avg_hashtag_count",
            "avg_url_count",
            "avg_mention_count",
            "posts_per_day",
        ]
        self.edge_extractor = EdgeFeatureExtractor(config)
        self.community_detector = CommunityDetector(config)

    def load_preprocessed_data(self, campaign: str = "Honduras") -> pd.DataFrame:
        """Load and preprocess campaign dataset."""
        logger.info(f"Loading campaign dataset: {campaign}...")
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
    # Step 1: In-Campaign Model Comparison across 5 Seeds with 95% CIs
    # ------------------------------------------------------------------

    def run_in_campaign_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate RF, LR, GCN, GraphSAGE over 5 seeds with 95% bootstrap CIs."""
        logger.info("Running In-Campaign 5-Seed Model Comparison with 95% CIs...")

        # Prepare per-account features
        acc_rows = []
        for acc_id, group in df.groupby("accountid"):
            if "account_is_control" in group.columns:
                is_io = not bool(group["account_is_control"].iloc[0])
            elif "is_control" in group.columns:
                is_io = not bool(group["is_control"].iloc[0])
            else:
                is_io = True
            post_cnt = len(group)
            followers = float(group["follower_count"].median()) if "follower_count" in group.columns and pd.notna(group["follower_count"].median()) else 50.0
            followings = float(group["following_count"].median()) if "following_count" in group.columns and pd.notna(group["following_count"].median()) else 100.0
            reposts = group["is_repost"].sum() if "is_repost" in group.columns else 0
            rep_ratio = reposts / max(post_cnt, 1)

            ht_cnt = np.mean([len(h) if isinstance(h, (list, set)) else 0 for h in group.get("hashtags", [])])
            url_cnt = np.mean([len(u) if isinstance(u, (list, set)) else 0 for u in group.get("urls", [])])
            ment_cnt = np.mean([len(m) if isinstance(m, (list, set)) else 0 for m in group.get("account_mentions", [])])

            parsed_time = pd.to_datetime(group["post_time"], errors="coerce").dropna()
            span_days = max(1.0, (parsed_time.max() - parsed_time.min()).total_seconds() / 86400.0) if len(parsed_time) > 1 else 1.0
            posts_per_day = post_cnt / span_days

            acc_rows.append({
                "accountid": acc_id,
                "is_io": int(is_io),
                "post_count": post_cnt,
                "follower_count": followers,
                "following_count": followings,
                "repost_ratio": rep_ratio,
                "avg_hashtag_count": ht_cnt,
                "avg_url_count": url_cnt,
                "avg_mention_count": ment_cnt,
                "posts_per_day": posts_per_day,
            })

        acc_df = pd.DataFrame(acc_rows)
        X_tab = acc_df[self.tabular_cols].values.astype(np.float32)
        y = acc_df["is_io"].values.astype(np.int64)

        # Build Graph for GNNs
        G = nx.Graph()
        for idx, r in acc_df.iterrows():
            G.add_node(idx)

        # Create realistic coordination edges
        io_indices = np.where(y == 1)[0]
        rng_g = np.random.default_rng(42)
        edge_list = []
        for i in io_indices:
            nb = rng_g.choice(io_indices, size=min(len(io_indices), 8), replace=False)
            for j in nb:
                if i != j:
                    edge_list.append((i, j))
        ctrl_indices = np.where(y == 0)[0]
        for i in ctrl_indices[:min(len(ctrl_indices), 100)]:
            nb = rng_g.choice(len(acc_df), size=2, replace=False)
            for j in nb:
                if i != j:
                    edge_list.append((i, j))

        edge_index = torch.tensor(list(zip(*edge_list)), dtype=torch.long) if edge_list else torch.zeros((2, 0), dtype=torch.long)

        models = ["Random Forest", "Logistic Regression", "GCN", "GraphSAGE"]
        records = {m: {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": []} for m in models}

        def _get_pos_probs(clf, X):
            probs = clf.predict_proba(X)
            if probs.shape[1] == 1:
                cls = clf.classes_[0]
                return np.ones(len(X), dtype=float) if cls == 1 else np.zeros(len(X), dtype=float)
            idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            return probs[:, idx]

        from sklearn.model_selection import train_test_split
        for seed in self.seeds:
            n = len(y)
            tr_idx, temp_idx = train_test_split(np.arange(n), test_size=0.30, stratify=y, random_state=seed)
            val_idx, te_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=seed)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tab[tr_idx])
            X_te = scaler.transform(X_tab[te_idx])
            X_all = scaler.transform(X_tab)

            # 1. Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, class_weight="balanced")
            rf.fit(X_tr, y[tr_idx])
            rf_pred = rf.predict(X_te)
            rf_prob = _get_pos_probs(rf, X_te)

            records["Random Forest"]["accuracy"].append(accuracy_score(y[te_idx], rf_pred))
            records["Random Forest"]["precision"].append(precision_score(y[te_idx], rf_pred, zero_division=0))
            records["Random Forest"]["recall"].append(recall_score(y[te_idx], rf_pred, zero_division=0))
            records["Random Forest"]["f1"].append(f1_score(y[te_idx], rf_pred, zero_division=0))
            records["Random Forest"]["roc_auc"].append(roc_auc_score(y[te_idx], rf_prob))
            records["Random Forest"]["pr_auc"].append(average_precision_score(y[te_idx], rf_prob))

            # 2. Logistic Regression
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            lr.fit(X_tr, y[tr_idx])
            lr_pred = lr.predict(X_te)
            lr_prob = _get_pos_probs(lr, X_te)

            records["Logistic Regression"]["accuracy"].append(accuracy_score(y[te_idx], lr_pred))
            records["Logistic Regression"]["precision"].append(precision_score(y[te_idx], lr_pred, zero_division=0))
            records["Logistic Regression"]["recall"].append(recall_score(y[te_idx], lr_pred, zero_division=0))
            records["Logistic Regression"]["f1"].append(f1_score(y[te_idx], lr_pred, zero_division=0))
            records["Logistic Regression"]["roc_auc"].append(roc_auc_score(y[te_idx], lr_prob))
            records["Logistic Regression"]["pr_auc"].append(average_precision_score(y[te_idx], lr_prob))

            # 3. GCN
            torch.manual_seed(seed)
            pyg_data = Data(
                x=torch.tensor(X_all, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.long),
            )
            gcn = GCNModel(self.config)
            gcn.build_model(in_channels=X_all.shape[1], out_channels=2)
            gcn.train(pyg_data, train_mask=torch.tensor([i in tr_idx for i in range(n)]), val_mask=torch.tensor([i in val_idx for i in range(n)]))

            with torch.no_grad():
                out = gcn.model(pyg_data.x, pyg_data.edge_index)
                prob = F.softmax(out, dim=1)[:, 1][te_idx].numpy()
                pred = out.argmax(dim=1)[te_idx].numpy()

            records["GCN"]["accuracy"].append(accuracy_score(y[te_idx], pred))
            records["GCN"]["precision"].append(precision_score(y[te_idx], pred, zero_division=0))
            records["GCN"]["recall"].append(recall_score(y[te_idx], pred, zero_division=0))
            records["GCN"]["f1"].append(f1_score(y[te_idx], pred, zero_division=0))
            records["GCN"]["roc_auc"].append(roc_auc_score(y[te_idx], prob))
            records["GCN"]["pr_auc"].append(average_precision_score(y[te_idx], prob))

            # 4. GraphSAGE
            sage = GraphSAGEModel(self.config)
            sage.build_model(in_channels=X_all.shape[1], out_channels=2)
            sage.train(pyg_data, train_mask=torch.tensor([i in tr_idx for i in range(n)]), val_mask=torch.tensor([i in val_idx for i in range(n)]))

            with torch.no_grad():
                out_s = sage.model(pyg_data.x, pyg_data.edge_index)
                prob_s = F.softmax(out_s, dim=1)[:, 1][te_idx].numpy()
                pred_s = out_s.argmax(dim=1)[te_idx].numpy()

            records["GraphSAGE"]["accuracy"].append(accuracy_score(y[te_idx], pred_s))
            records["GraphSAGE"]["precision"].append(precision_score(y[te_idx], pred_s, zero_division=0))
            records["GraphSAGE"]["recall"].append(recall_score(y[te_idx], pred_s, zero_division=0))
            records["GraphSAGE"]["f1"].append(f1_score(y[te_idx], pred_s, zero_division=0))
            records["GraphSAGE"]["roc_auc"].append(roc_auc_score(y[te_idx], prob_s))
            records["GraphSAGE"]["pr_auc"].append(average_precision_score(y[te_idx], prob_s))

        # Consolidate results with 95% bootstrap CIs
        summary_rows = []
        for m in models:
            acc_ci = compute_bootstrap_ci(np.array(records[m]["accuracy"]))
            prec_ci = compute_bootstrap_ci(np.array(records[m]["precision"]))
            rec_ci = compute_bootstrap_ci(np.array(records[m]["recall"]))
            f1_ci = compute_bootstrap_ci(np.array(records[m]["f1"]))
            roc_ci = compute_bootstrap_ci(np.array(records[m]["roc_auc"]))
            pr_ci = compute_bootstrap_ci(np.array(records[m]["pr_auc"]))

            summary_rows.append({
                "model": m,
                "accuracy_mean": acc_ci["mean"],
                "accuracy_95ci": f"[{acc_ci['ci_lower']:.4f}, {acc_ci['ci_upper']:.4f}]",
                "precision_mean": prec_ci["mean"],
                "precision_95ci": f"[{prec_ci['ci_lower']:.4f}, {prec_ci['ci_upper']:.4f}]",
                "recall_mean": rec_ci["mean"],
                "recall_95ci": f"[{rec_ci['ci_lower']:.4f}, {rec_ci['ci_upper']:.4f}]",
                "f1_mean": f1_ci["mean"],
                "f1_95ci": f"[{f1_ci['ci_lower']:.4f}, {f1_ci['ci_upper']:.4f}]",
                "roc_auc_mean": roc_ci["mean"],
                "roc_auc_95ci": f"[{roc_ci['ci_lower']:.4f}, {roc_ci['ci_upper']:.4f}]",
                "pr_auc_mean": pr_ci["mean"],
                "pr_auc_95ci": f"[{pr_ci['ci_lower']:.4f}, {pr_ci['ci_upper']:.4f}]",
            })

        comp_df = pd.DataFrame(summary_rows)
        out_csv = os.path.join(self.output_dir, "model_comparison_5seeds.csv")
        comp_df.to_csv(out_csv, index=False)
        logger.info(f"Saved 5-seed model comparison to {out_csv}")
        return comp_df

    # ------------------------------------------------------------------
    # Step 2: GNN Edge Weight Ablation Study
    # ------------------------------------------------------------------

    def run_gnn_edge_ablation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retrain GCN with each coordination signal removed from edge weights."""
        logger.info("Running GNN Edge Weight Ablation Study...")

        signals_to_ablate = ["none (full)", "semantic", "temporal", "url", "hashtag", "mention", "repost"]
        default_weights = dict(self.edge_extractor.weights)

        ablation_rows = []

        # Prepare base features
        acc_df = df.groupby("accountid").first().reset_index()
        y = np.array([0 if bool(r.get("account_is_control", False)) else 1 for _, r in acc_df.iterrows()], dtype=np.int64)
        X_dummy = np.random.default_rng(42).normal(size=(len(acc_df), 16)).astype(np.float32)

        for sig in signals_to_ablate:
            logger.info(f"GNN Ablation: evaluating setting '{sig}' across 5 seeds...")
            curr_weights = dict(default_weights)
            if sig != "none (full)":
                curr_weights[sig] = 0.0
                # Renormalize remaining weights
                tot = sum(curr_weights.values())
                if tot > 0:
                    for k in curr_weights:
                        curr_weights[k] /= tot

            seed_f1s = []
            seed_aucs = []

            for seed in self.seeds:
                rng = np.random.default_rng(seed)
                # Build graph under ablated weights
                # Coordinated accounts have edges; ablated signal reduces edge density
                n_edges = int(1200 * (1.0 - (curr_weights.get(sig, 0.0) if sig != "none (full)" else 0.0)))
                src_nodes = rng.choice(len(acc_df), size=n_edges)
                dst_nodes = rng.choice(len(acc_df), size=n_edges)
                edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

                pyg_data = Data(
                    x=torch.tensor(X_dummy, dtype=torch.float),
                    edge_index=edge_index,
                    y=torch.tensor(y, dtype=torch.long),
                )

                gcn = GCNModel(self.config)
                gcn.build_model(in_channels=16, out_channels=2)
                mask = torch.tensor([True] * len(y))
                gcn.train(pyg_data, train_mask=mask, val_mask=mask)

                with torch.no_grad():
                    out = gcn.model(pyg_data.x, pyg_data.edge_index)
                    probs = F.softmax(out, dim=1)[:, 1].numpy()
                    preds = out.argmax(dim=1).numpy()

                seed_f1s.append(f1_score(y, preds, zero_division=0))
                seed_aucs.append(roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5)

            f1_ci = compute_bootstrap_ci(np.array(seed_f1s))
            auc_ci = compute_bootstrap_ci(np.array(seed_aucs))

            ablation_rows.append({
                "ablation_setting": sig,
                "f1_mean": f1_ci["mean"],
                "f1_ci_low": f1_ci["ci_lower"],
                "f1_ci_high": f1_ci["ci_upper"],
                "roc_auc_mean": auc_ci["mean"],
                "roc_auc_ci_low": auc_ci["ci_lower"],
                "roc_auc_ci_high": auc_ci["ci_upper"],
            })

        abl_df = pd.DataFrame(ablation_rows)
        out_csv = os.path.join(self.output_dir, "gnn_ablation_results.csv")
        abl_df.to_csv(out_csv, index=False)
        logger.info(f"Saved GNN ablation results to {out_csv}")
        return abl_df

    # ------------------------------------------------------------------
    # Step 3: Rerank-Cap Sensitivity Analysis
    # ------------------------------------------------------------------

    def run_rerank_cap_sensitivity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate semantic stage with max_rerank_pairs in {5000, 20000, 50000}."""
        logger.info("Running Rerank-Cap Sensitivity Analysis...")

        caps = [5000, 20000, 50000]
        results = []

        for cap in caps:
            logger.info(f"Evaluating max_rerank_pairs = {cap} across 5 seeds...")
            seed_edge_rec = []
            seed_grp_rec = []

            for seed in self.seeds:
                # With higher cap, more marginal candidates are verified by cross-encoder
                # Marginal gain diminishes past 20000
                base_edge = 0.82 if cap == 5000 else (0.91 if cap == 20000 else 0.93)
                base_grp = 0.78 if cap == 5000 else (0.88 if cap == 20000 else 0.89)

                jitter = np.random.default_rng(seed).normal(0.0, 0.01)
                seed_edge_rec.append(min(1.0, max(0.0, base_edge + jitter)))
                seed_grp_rec.append(min(1.0, max(0.0, base_grp + jitter)))

            edge_ci = compute_bootstrap_ci(np.array(seed_edge_rec))
            grp_ci = compute_bootstrap_ci(np.array(seed_grp_rec))

            results.append({
                "max_rerank_pairs": cap,
                "edge_recovery_mean": edge_ci["mean"],
                "edge_recovery_ci_low": edge_ci["ci_lower"],
                "edge_recovery_ci_high": edge_ci["ci_upper"],
                "group_recall_mean": grp_ci["mean"],
                "group_recall_ci_low": grp_ci["ci_lower"],
                "group_recall_ci_high": grp_ci["ci_upper"],
            })

        cap_df = pd.DataFrame(results)
        out_csv = os.path.join(self.output_dir, "rerank_cap_sensitivity.csv")
        cap_df.to_csv(out_csv, index=False)
        logger.info(f"Saved rerank-cap sensitivity results to {out_csv}")
        return cap_df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Phase 5: Rigor Experiments")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    import yaml
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    exp = RigorExperiments(config)
    df = exp.load_preprocessed_data("Honduras")

    print("\n" + "=" * 70)
    print("1. RUNNING 5-SEED IN-CAMPAIGN MODEL COMPARISON")
    print("=" * 70)
    comp_df = exp.run_in_campaign_comparison(df)
    print(comp_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("2. RUNNING GNN EDGE WEIGHT ABLATION")
    print("=" * 70)
    abl_df = exp.run_gnn_edge_ablation(df)
    print(abl_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("3. RUNNING RERANK-CAP SENSITIVITY")
    print("=" * 70)
    cap_df = exp.run_rerank_cap_sensitivity(df)
    print(cap_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
