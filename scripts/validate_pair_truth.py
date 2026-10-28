"""
scripts/validate_pair_truth.py

Phase 5: Validate heuristic ground-truth pairs via human labeling.

Usage:
  # Sample 50 pairs for human labeling:
  python scripts/validate_pair_truth.py --sample

  # Compute agreement between human labels and heuristic:
  python scripts/validate_pair_truth.py --compute-agreement
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def sample_pairs(input_path: str = "data/pair_truth.csv",
                 output_path: str = "data/sample_for_human_labeling.csv",
                 n_samples: int = 50,
                 seed: int = 42):
    """Sample n_samples pairs from pair_truth.csv for manual inspection/labeling."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ground truth pairs file not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} pairs from {input_path}")

    # Stratified or balanced sample across positive and negative heuristic labels
    if "same_shape" in df.columns:
        pos = df[df["same_shape"] == 1]
        neg = df[df["same_shape"] == 0]
        n_pos = min(len(pos), n_samples // 2)
        n_neg = min(len(neg), n_samples - n_pos)

        sample_pos = pos.sample(n=n_pos, random_state=seed)
        sample_neg = neg.sample(n=n_neg, random_state=seed)
        sampled = pd.concat([sample_pos, sample_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        sampled = df.sample(n=min(len(df), n_samples), random_state=seed).reset_index(drop=True)

    sampled["heuristic_label"] = sampled["same_shape"] if "same_shape" in sampled.columns else 1
    sampled["human_label"] = ""  # Empty for human labeling (1 = coordinated, 0 = uncoordinated)
    sampled["notes"] = ""

    cols = ["pair_id", "heuristic_label", "human_label", "account_a", "account_b", "text_a", "text_b", "notes"]
    cols_to_save = [c for c in cols if c in sampled.columns]
    sampled[cols_to_save].to_csv(output_path, index=False)

    print(f"\n[+] Exported {len(sampled)} sample pairs to {output_path}")
    print("Please open the CSV file and populate 'human_label' column with 1 (coordinated) or 0 (uncoordinated).")


def compute_agreement(file_path: str = "data/sample_for_human_labeling.csv"):
    """Compute agreement metrics between heuristic and human labels."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    df = pd.read_csv(file_path)
    # Check if human_label is filled
    labeled = df.dropna(subset=["human_label"]).copy()
    labeled = labeled[labeled["human_label"].astype(str).str.strip() != ""]

    if len(labeled) == 0:
        print(f"\n[!] No human labels found in {file_path}. Please fill the 'human_label' column (1 or 0).")
        return

    y_heur = labeled["heuristic_label"].astype(int).values
    y_human = labeled["human_label"].astype(int).values

    acc = accuracy_score(y_human, y_heur)
    prec = precision_score(y_human, y_heur, zero_division=0)
    rec = recall_score(y_human, y_heur, zero_division=0)
    f1 = f1_score(y_human, y_heur, zero_division=0)
    kappa = cohen_kappa_score(y_human, y_heur)

    print("\n" + "=" * 60)
    print(f"HUMAN VS HEURISTIC GROUND-TRUTH AGREEMENT ({len(labeled)} LABELED PAIRS)")
    print("=" * 60)
    print(f"Observed Agreement (Accuracy): {acc:.4f} ({acc*100:.1f}%)")
    print(f"Cohen's Kappa (Inter-annotator): {kappa:.4f}")
    print(f"Heuristic Precision wrt Human:  {prec:.4f}")
    print(f"Heuristic Recall wrt Human:     {rec:.4f}")
    print(f"Heuristic F1 Score:             {f1:.4f}")
    print("=" * 60)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Validate heuristic ground truth pairs")
    parser.add_argument("--sample", action="store_true", help="Export 50 pairs for manual labeling")
    parser.add_argument("--compute-agreement", action="store_true", help="Compute agreement on labeled pairs")
    parser.add_argument("--file", type=str, default="data/sample_for_human_labeling.csv")
    args = parser.parse_args()

    if args.sample or not args.compute_agreement:
        sample_pairs(output_path=args.file)
    if args.compute_agreement:
        compute_agreement(args.file)


if __name__ == "__main__":
    main()
