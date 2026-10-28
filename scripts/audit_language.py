#!/usr/bin/env python3
"""
scripts/audit_language.py

Audit post_language distributions, text length statistics, and IO vs. Control
class balance for both Honduras and UAE campaigns in the Zenodo dataset (record 13912659).

Usage:
    python scripts/audit_language.py [--out results/audit_language_summary.json]
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("audit_language")


def load_campaign_from_local(campaign: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load campaign posts from local raw files."""
    camp_dir = os.path.join(raw_dir, campaign)
    if not os.path.exists(camp_dir):
        logger.error(f"Campaign directory not found: {camp_dir}")
        return pd.DataFrame()

    records = []

    # Check for CSV first
    csv_file = os.path.join(camp_dir, f"{campaign.lower()}_io.csv")
    if os.path.exists(csv_file):
        logger.info(f"Loading {campaign} from CSV: {csv_file}")
        df = pd.read_csv(csv_file, low_memory=False)
        for _, row in df.iterrows():
            text = str(row.get("post_text", ""))
            lang = str(row.get("post_language", "und")).strip().lower()
            is_ctrl = bool(row.get("is_control", False))
            records.append({
                "campaign": campaign,
                "post_text": text,
                "post_language": lang if lang else "und",
                "is_control": is_ctrl,
                "is_io": not is_ctrl,
            })
        return pd.DataFrame(records)

    # Check for JSONL files
    jsonl_files = [f for f in os.listdir(camp_dir) if f.endswith(".jsonl")]
    logger.info(f"Loading {campaign} from {len(jsonl_files)} JSONL file(s): {jsonl_files}")

    for jf in jsonl_files:
        path = os.path.join(camp_dir, jf)
        # Infer IO status from filename if not inside record
        is_bad_file = "bad" in jf.lower()
        count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("tweet_text", "") or obj.get("post_text", "")
                    lang = obj.get("tweet_language", "") or obj.get("post_language", "und")

                    if "good" in obj:
                        is_ctrl = bool(obj["good"])
                    elif "is_control" in obj:
                        is_ctrl = bool(obj["is_control"])
                    else:
                        is_ctrl = not is_bad_file

                    records.append({
                        "campaign": campaign,
                        "post_text": str(text),
                        "post_language": str(lang).strip().lower() if lang else "und",
                        "is_control": is_ctrl,
                        "is_io": not is_ctrl,
                    })
                    count += 1
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {count} records from {jf}")

    return pd.DataFrame(records)


def compute_campaign_stats(df: pd.DataFrame, campaign_name: str) -> Dict:
    """Compute detailed language distribution, text lengths, and IO balance."""
    if df.empty:
        return {"campaign": campaign_name, "error": "No data available"}

    total_posts = len(df)

    # 1. Class Balance (IO vs. Control)
    io_count = int(df["is_io"].sum())
    control_count = int(total_posts - io_count)
    io_share = io_count / total_posts if total_posts > 0 else 0.0

    # 2. Language Distribution
    lang_counts = df["post_language"].value_counts()
    lang_shares = (lang_counts / total_posts * 100).round(2)
    top_langs = []
    for lang, cnt in lang_counts.head(10).items():
        top_langs.append({
            "language": lang,
            "count": int(cnt),
            "share_percent": float(lang_shares[lang]),
        })

    # 3. Text Length Statistics
    lengths = df["post_text"].astype(str).str.len()
    word_counts = df["post_text"].astype(str).str.split().str.len()

    char_stats = {
        "mean": float(np.round(lengths.mean(), 2)),
        "std": float(np.round(lengths.std(), 2)),
        "median": float(lengths.median()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "p25": float(lengths.quantile(0.25)),
        "p75": float(lengths.quantile(0.75)),
    }

    word_stats = {
        "mean": float(np.round(word_counts.mean(), 2)),
        "std": float(np.round(word_counts.std(), 2)),
        "median": float(word_counts.median()),
        "max": int(word_counts.max()),
    }

    return {
        "campaign": campaign_name,
        "total_posts": total_posts,
        "class_balance": {
            "io_posts": io_count,
            "control_posts": control_count,
            "io_share_percent": round(io_share * 100, 2),
            "control_share_percent": round((1 - io_share) * 100, 2),
        },
        "top_languages": top_langs,
        "dominant_language": top_langs[0]["language"] if top_langs else "unknown",
        "dominant_language_share": top_langs[0]["share_percent"] if top_langs else 0.0,
        "char_length_stats": char_stats,
        "word_count_stats": word_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit dataset language and length distribution")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw dataset directory")
    parser.add_argument("--out", type=str, default="results/audit_language_summary.json", help="Output JSON path")
    args = parser.parse_args()

    results = {}

    for campaign in ["Honduras", "UAE"]:
        logger.info(f"=== Auditing Campaign: {campaign} ===")
        df = load_campaign_from_local(campaign, raw_dir=args.raw_dir)
        stats = compute_campaign_stats(df, campaign)
        results[campaign] = stats

    # Save output JSON
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Audit results written to {args.out}")

    # Print formatted markdown report
    print("\n" + "=" * 70)
    print("PHASE 0 AUDIT REPORT: LANGUAGE & DATASET BALANCE")
    print("=" * 70)

    for campaign, s in results.items():
        print(f"\n### Campaign: {campaign}")
        print(f"- **Total Sampled Posts:** {s['total_posts']:,}")
        cb = s["class_balance"]
        print(f"- **Class Balance:** IO = {cb['io_posts']:,} ({cb['io_share_percent']}%) | Control = {cb['control_posts']:,} ({cb['control_share_percent']}%)")
        print(f"- **Character Length (chars):** Mean={s['char_length_stats']['mean']}, Median={s['char_length_stats']['median']}, Min={s['char_length_stats']['min']}, Max={s['char_length_stats']['max']}")
        print(f"- **Word Count (words):** Mean={s['word_count_stats']['mean']}, Median={s['word_count_stats']['median']}, Max={s['word_count_stats']['max']}")
        print("\n**Top Languages Distribution:**")
        print("| Language Code | Post Count | Share (%) |")
        print("|---|---|---|")
        for item in s["top_languages"]:
            print(f"| `{item['language']}` | {item['count']:,} | {item['share_percent']}% |")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
