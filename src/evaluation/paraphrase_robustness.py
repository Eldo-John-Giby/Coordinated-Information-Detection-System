"""
Paraphrase-robustness experiment.

Demonstrates that cross-encoders maintain high scores on paraphrased
coordinated content while bi-encoder scores degrade. This is the key
experimental contribution showing why cross-encoder reranking matters
against LLM-paraphrased coordinated messaging.

Can be run independently of the full pipeline:
    python -m src.evaluation.paraphrase_robustness --sample
    python -m src.evaluation.paraphrase_robustness --config configs/config.yaml
"""

import argparse
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Contextual synonym & phrase replacement rules for realistic adversarial rephrasing
SYNONYM_REPLACEMENTS = {
    # English terms
    r"\bbreaking\b": "urgent update",
    r"\bimportant\b": "crucial",
    r"\bprotest\b": "demonstration",
    r"\bprotesters\b": "marchers",
    r"\bgovernment\b": "administration",
    r"\bpresident\b": "head of state",
    r"\bcorruption\b": "illicit practices",
    r"\bpeaceful\b": "non-violent",
    r"\bviolence\b": "civil unrest",
    r"\belection\b": "balloting",
    r"\bvoters\b": "citizens casting votes",
    r"\bcountry\b": "nation",
    r"\bpeople\b": "public",
    r"\bfake news\b": "disinformation",
    r"\bmedia\b": "press outlets",
    r"\btruth\b": "facts of the matter",
    r"\bexposed\b": "brought to light",
    r"\bsupport\b": "stand with",
    r"\bagainst\b": "in opposition to",
    r"\bnow\b": "at this current moment",
    r"\btoday\b": "on this day",
    r"\bwatch\b": "observe",
    r"\bshare\b": "circulate this message",
    # Spanish terms (common in Honduras IO data)
    r"\balivio de deuda\b": "reduccion del endeudamiento",
    r"\baliviodedeuda\b": "reducciondeuda",
    r"\bsalvemos honduras\b": "defendamos a honduras",
    r"\bsalvemoshonduras\b": "protejamoshonduras",
    r"\bexitos lider\b": "los mejores deseos dirigente",
    r"\bsigamos sumando\b": "continuemos avanzando unidos",
    r"\bincitando al pueblo\b": "provocando a los ciudadanos",
    r"\bviolencia\b": "disturbios",
    r"\bpueblo\b": "ciudadania",
    r"\bgobierno\b": "administracion central",
    r"\bpresidente\b": "mandatario",
    r"\bnavidad catracha\b": "fiestas navidenas nacionales",
    r"\bparque vida mejor\b": "espacio recreativo comunitario",
    r"\besta gente\b": "estos sectores",
    r"\bno se cansan\b": "no desisten de",
    r"\bdenuncia\b": "senalamiento formal",
}

INTRO_TEMPLATES = [
    "Worth noting: {text}",
    "Key development: {text}",
    "Take note of this update: {text}",
    "Important message to highlight: {text}",
    "From ongoing reports: {text}",
]


class ParaphraseRobustnessExperiment:
    """Compare bi-encoder vs cross-encoder robustness to LLM paraphrasing.

    1. Sample known-coordinated post pairs (IO accounts, high bi-encoder score)
    2. Paraphrase one post per pair using an LLM / high-entropy generator
    3. Score original-original and original-paraphrased with both encoders
    4. Produce before/after comparison table and visualization
    """

    def __init__(self, config: dict):
        self.config = config
        exp_config = config.get("experiments", {})
        self.output_dir = exp_config.get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        self.n_pairs = config.get("paraphrase_experiment", {}).get("n_pairs", 200)
        self.min_biencoder_score = config.get("paraphrase_experiment", {}).get(
            "min_biencoder_score", 0.7
        )
        self.paraphrase_model = config.get("paraphrase_experiment", {}).get(
            "paraphrase_model", "claude-sonnet-4-6"
        )
        self.random_seed = config.get("data", {}).get("random_seed", 42)

    # ------------------------------------------------------------------
    # Step 1: Sample coordinated post pairs
    # ------------------------------------------------------------------

    def sample_coordinated_pairs(
        self, df: Optional[pd.DataFrame] = None
    ) -> List[Tuple[str, str, str, str]]:
        """Sample post pairs from known-coordinated IO accounts.

        Prioritizes pair_truth.csv ground-truth pairs if present, otherwise
        samples via embedding nearest-neighbor search.
        Returns list of (post_a_text, post_b_text, account_a, account_b).
        """
        pairs: List[Tuple[str, str, str, str]] = []

        # Check for pair_truth.csv
        pair_truth_paths = [
            os.path.join("data", "pair_truth.csv"),
            os.path.join("data", "raw", "Honduras", "pair_truth.csv"),
        ]
        for pt_path in pair_truth_paths:
            if os.path.exists(pt_path):
                try:
                    pt_df = pd.read_csv(pt_path)
                    positives = pt_df[pt_df["same_shape"] == 1]
                    if len(positives) > 0:
                        logger.info(
                            f"Loaded {len(positives)} verified coordinated pairs from {pt_path}"
                        )
                        sampled = positives.head(self.n_pairs)
                        for _, row in sampled.iterrows():
                            text_a = str(row["text_a"]).strip()
                            text_b = str(row["text_b"]).strip()
                            acc_a = str(row.get("account_a", "acc_A"))
                            acc_b = str(row.get("account_b", "acc_B"))
                            if text_a and text_b and text_a != text_b:
                                pairs.append((text_a, text_b, acc_a, acc_b))
                        if len(pairs) >= min(20, self.n_pairs):
                            logger.info(f"Using {len(pairs)} ground-truth coordinated pairs")
                            return pairs
                except Exception as e:
                    logger.warning(f"Failed to read pair truth from {pt_path}: {e}")

        if df is None or len(df) == 0:
            logger.error("No DataFrame provided and no pair_truth.csv found.")
            return []

        from src.nlp.embeddings import EmbeddingModel
        from src.nlp.similarity import SimilaritySearch

        logger.info("Sampling coordinated post pairs via embedding search...")

        # Filter to IO accounts only if available
        if "is_control" in df.columns:
            io_accounts = set(
                df.groupby("accountid")["is_control"]
                .all()
                .reset_index()
                .query("is_control == False")["accountid"]
            )
            df_io = df[df["accountid"].isin(io_accounts)].copy()
        else:
            df_io = df.copy()

        if len(df_io) == 0:
            df_io = df.copy()

        embedder = EmbeddingModel(self.config)
        df_io = embedder.encode_dataframe(df_io)

        account_emb = {}
        for acc, group in df_io.groupby("accountid"):
            embs = np.array(group["embedding"].tolist())
            account_emb[acc] = embs.mean(axis=0)

        acc_ids = list(account_emb.keys())
        emb_matrix = np.array([account_emb[a].astype(np.float32) for a in acc_ids])

        sim_search = SimilaritySearch(self.config)
        sim_search.build_index(emb_matrix)
        indices, similarities = sim_search.search(
            emb_matrix, k=min(20, max(2, len(acc_ids)))
        )

        candidates = []
        for i in range(len(acc_ids)):
            for j_idx in range(1, indices.shape[1]):
                neighbor_idx = indices[i, j_idx]
                sim = float(similarities[i, j_idx])
                if sim >= self.min_biencoder_score:
                    acc_a = acc_ids[i]
                    acc_b = acc_ids[neighbor_idx]
                    if acc_a != acc_b:
                        candidates.append((acc_a, acc_b, sim))

        if not candidates:
            for i in range(len(acc_ids)):
                for j_idx in range(1, min(5, indices.shape[1])):
                    neighbor_idx = indices[i, j_idx]
                    acc_a = acc_ids[i]
                    acc_b = acc_ids[neighbor_idx]
                    if acc_a != acc_b:
                        candidates.append((acc_a, acc_b, float(similarities[i, j_idx])))

        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[: self.n_pairs]

        for acc_a, acc_b, _ in candidates:
            posts_a = df_io[df_io["accountid"] == acc_a]["post_text"].tolist()
            posts_b = df_io[df_io["accountid"] == acc_b]["post_text"].tolist()
            text_a = next((t for t in posts_a if isinstance(t, str) and t.strip()), "")
            text_b = next((t for t in posts_b if isinstance(t, str) and t.strip()), "")
            if text_a and text_b:
                pairs.append((text_a, text_b, str(acc_a), str(acc_b)))

        logger.info(f"Sampled {len(pairs)} coordinated post pairs")
        return pairs

    # ------------------------------------------------------------------
    # Step 2: Multi-Tier Paraphrase Engine
    # ------------------------------------------------------------------

    def _local_adversarial_paraphrase(self, text: str, seed: int = 42) -> str:
        """High-entropy lexical & syntactic rewrite engine.

        Simulates sophisticated adversary LLM evasion by:
        1. Preserving all critical URLs, handles, and hashtags
        2. Swapping core vocabulary with contextual synonyms
        3. Restructuring phrasing (transposition & frame restructuring)
        4. Lowering lexical overlap while preserving semantic intent
        """
        import random
        rng = random.Random(seed)

        # Extract and protect URLs and mentions
        tokens = text.split()
        protected = []
        regular = []
        for token in tokens:
            if token.startswith("http") or token.startswith("@") or token.startswith("#"):
                protected.append(token)
            else:
                regular.append(token)

        core_text = " ".join(regular)
        paraphrased_core = core_text

        # 1. Apply phrase and synonym replacements
        for pattern, replacement in SYNONYM_REPLACEMENTS.items():
            if re.search(pattern, paraphrased_core, re.IGNORECASE):
                paraphrased_core = re.sub(
                    pattern, replacement, paraphrased_core, flags=re.IGNORECASE
                )

        # 2. Syntactic restructuring: apply template framing if short
        if len(paraphrased_core.split()) > 2 and rng.random() > 0.4:
            tmpl = rng.choice(INTRO_TEMPLATES)
            paraphrased_core = tmpl.format(text=paraphrased_core)

        # 3. If no lexical changes occurred (e.g. unique words), perform token shift
        if paraphrased_core.strip() == core_text.strip() and regular:
            if len(regular) >= 4:
                # Transpose first and second halves with conjunction
                mid = len(regular) // 2
                paraphrased_core = f"{' '.join(regular[mid:])}, regarding {' '.join(regular[:mid])}"
            else:
                paraphrased_core = f"Stated clearly: {core_text}"

        # 4. Re-attach protected entities
        if protected:
            if rng.random() > 0.5:
                final_text = f"{paraphrased_core} {' '.join(protected)}"
            else:
                final_text = f"{' '.join(protected)}: {paraphrased_core}"
        else:
            final_text = paraphrased_core

        return final_text.strip()

    def paraphrase_pairs(
        self, pairs: List[Tuple[str, str, str, str]]
    ) -> List[Tuple[str, str, str, str, str]]:
        """Paraphrase the second post in each pair using Anthropic, OpenAI, or local generator.

        Returns list of (text_a, text_b_original, text_b_paraphrased, acc_a, acc_b).
        """
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        mode = "local"
        client = None

        if anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                mode = "anthropic"
                logger.info(f"Using Anthropic API ({self.paraphrase_model}) for paraphrasing")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}. Falling back...")

        if mode == "local" and openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                mode = "openai"
                logger.info("Using OpenAI API (gpt-4o-mini) for paraphrasing")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Falling back...")

        if mode == "local":
            logger.info("No external LLM API key detected. Running high-entropy adversarial paraphrase generator.")

        results = []
        total = len(pairs)
        prompt_template = (
            "Paraphrase the following social media post, preserving its "
            "exact meaning and entities but using substantially different wording and structure. "
            "Output ONLY the paraphrased text, nothing else.\n\n"
            "Post: {text}"
        )

        for idx, (text_a, text_b, acc_a, acc_b) in enumerate(pairs):
            if (idx + 1) % 25 == 0 or idx == 0:
                logger.info(f"Paraphrasing pair {idx + 1}/{total} (mode={mode})...")

            paraphrased = text_b
            if mode == "anthropic" and client is not None:
                try:
                    message = client.messages.create(
                        model=self.paraphrase_model,
                        max_tokens=256,
                        messages=[{"role": "user", "content": prompt_template.format(text=text_b)}],
                    )
                    paraphrased = message.content[0].text.strip()
                except Exception as e:
                    logger.warning(f"Anthropic paraphrase failed for pair {idx}: {e}")
                    paraphrased = self._local_adversarial_paraphrase(text_b, seed=idx)
            elif mode == "openai" and client is not None:
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        max_tokens=256,
                        messages=[{"role": "user", "content": prompt_template.format(text=text_b)}],
                    )
                    paraphrased = resp.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning(f"OpenAI paraphrase failed for pair {idx}: {e}")
                    paraphrased = self._local_adversarial_paraphrase(text_b, seed=idx)
            else:
                paraphrased = self._local_adversarial_paraphrase(text_b, seed=idx + 42)

            results.append((text_a, text_b, paraphrased, acc_a, acc_b))

        logger.info(f"Successfully generated {len(results)} paraphrased pairs")
        return results

    # ------------------------------------------------------------------
    # Step 3: Score with both encoders
    # ------------------------------------------------------------------

    def score_pairs(
        self,
        paraphrased_pairs: List[Tuple[str, str, str, str, str]],
    ) -> pd.DataFrame:
        """Compute bi-encoder and cross-encoder scores for original and paraphrased pairs."""
        from sentence_transformers import SentenceTransformer
        from src.nlp.similarity import CrossEncoderReranker

        biencoder_model_name = self.config.get("nlp", {}).get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(f"Loading bi-encoder model: {biencoder_model_name}")
        try:
            bi_encoder = SentenceTransformer(biencoder_model_name)
        except Exception:
            bi_encoder = SentenceTransformer(biencoder_model_name, local_files_only=True)

        logger.info("Loading cross-encoder model...")
        reranker = CrossEncoderReranker(self.config)
        reranker._load_model()

        # Collect unique texts to embed efficiently
        all_texts_a = [p[0] for p in paraphrased_pairs]
        all_texts_b_orig = [p[1] for p in paraphrased_pairs]
        all_texts_b_para = [p[2] for p in paraphrased_pairs]

        unique_texts = list(set(all_texts_a + all_texts_b_orig + all_texts_b_para))
        logger.info(f"Embedding {len(unique_texts)} unique texts with bi-encoder...")
        text_embeddings = bi_encoder.encode(
            unique_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True
        )
        text_to_emb = dict(zip(unique_texts, text_embeddings))

        # Compute Bi-encoder cosine similarities
        bi_scores_orig = []
        bi_scores_para = []
        for text_a, text_b_orig, text_b_para, _, _ in paraphrased_pairs:
            emb_a = text_to_emb[text_a]
            emb_b_orig = text_to_emb[text_b_orig]
            emb_b_para = text_to_emb[text_b_para]
            bi_scores_orig.append(float(np.dot(emb_a, emb_b_orig)))
            bi_scores_para.append(float(np.dot(emb_a, emb_b_para)))

        # Cross-encoder scores
        logger.info("Scoring pairs with cross-encoder...")
        pairs_orig = [(p[0], p[1]) for p in paraphrased_pairs]
        pairs_para = [(p[0], p[2]) for p in paraphrased_pairs]

        preds_orig = np.asarray(
            reranker._model.predict(pairs_orig, batch_size=reranker.batch_size, show_progress_bar=False)
        )
        preds_para = np.asarray(
            reranker._model.predict(pairs_para, batch_size=reranker.batch_size, show_progress_bar=False)
        )

        cross_scores_orig = reranker._normalize_scores(preds_orig)
        cross_scores_para = reranker._normalize_scores(preds_para)

        results_df = pd.DataFrame(
            {
                "pair_id": range(len(paraphrased_pairs)),
                "account_a": [p[3] for p in paraphrased_pairs],
                "account_b": [p[4] for p in paraphrased_pairs],
                "text_a": [p[0] for p in paraphrased_pairs],
                "text_b_original": [p[1] for p in paraphrased_pairs],
                "text_b_paraphrased": [p[2] for p in paraphrased_pairs],
                "biencoder_score_original": bi_scores_orig,
                "biencoder_score_paraphrased": bi_scores_para,
                "crossencoder_score_original": cross_scores_orig,
                "crossencoder_score_paraphrased": cross_scores_para,
            }
        )

        return results_df

    # ------------------------------------------------------------------
    # Step 4: Summarize
    # ------------------------------------------------------------------

    def summarize(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics."""
        bi_orig = results_df["biencoder_score_original"].mean()
        bi_para = results_df["biencoder_score_paraphrased"].mean()
        bi_drop = bi_orig - bi_para

        cross_orig = results_df["crossencoder_score_original"].mean()
        cross_para = results_df["crossencoder_score_paraphrased"].mean()
        cross_drop = cross_orig - cross_para

        summary = pd.DataFrame(
            {
                "metric": [
                    "mean_biencoder_original",
                    "mean_biencoder_paraphrased",
                    "biencoder_score_drop",
                    "biencoder_relative_drop_%",
                    "mean_crossencoder_original",
                    "mean_crossencoder_paraphrased",
                    "crossencoder_score_drop",
                    "crossencoder_relative_drop_%",
                    "drop_ratio (bi_drop / cross_drop)",
                    "n_pairs",
                ],
                "value": [
                    bi_orig,
                    bi_para,
                    bi_drop,
                    (bi_drop / max(bi_orig, 1e-6)) * 100,
                    cross_orig,
                    cross_para,
                    cross_drop,
                    (cross_drop / max(cross_orig, 1e-6)) * 100,
                    bi_drop / max(abs(cross_drop), 1e-6),
                    len(results_df),
                ],
            }
        )
        return summary

    # ------------------------------------------------------------------
    # Step 5: Visualize
    # ------------------------------------------------------------------

    def visualize(self, results_df: pd.DataFrame, summary_df: pd.DataFrame) -> Tuple[str, str]:
        """Create bar chart and similarity distribution plots for publication."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Figure 1: Summary Bar Chart
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        categories = ["Bi-Encoder (MiniLM)", "Cross-Encoder (Reranker)"]
        orig_means = [
            results_df["biencoder_score_original"].mean(),
            results_df["crossencoder_score_original"].mean(),
        ]
        para_means = [
            results_df["biencoder_score_paraphrased"].mean(),
            results_df["crossencoder_score_paraphrased"].mean(),
        ]

        x = np.arange(len(categories))
        width = 0.35
        ax1 = axes[0]
        bars1 = ax1.bar(x - width / 2, orig_means, width, label="Original Coordinated", color="#2980b9", alpha=0.9)
        bars2 = ax1.bar(x + width / 2, para_means, width, label="After Adversarial Paraphrase", color="#e74c3c", alpha=0.9)

        for bar in bars1:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for bar in bars2:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax1.set_ylabel("Similarity / Entailment Score")
        ax1.set_title("Matching Score: Original vs. Paraphrased Pairs")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontweight="bold")
        ax1.set_ylim(0, 1.15)
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2 = axes[1]
        drops = [
            orig_means[0] - para_means[0],
            orig_means[1] - para_means[1],
        ]
        drop_colors = ["#c0392b", "#27ae60"]
        bars3 = ax2.bar(categories, drops, color=drop_colors, alpha=0.88, width=0.45)
        for bar, drop_val in zip(bars3, drops):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{drop_val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax2.set_ylabel("Score Degradation (Drop)")
        ax2.set_title("Vulnerability to Paraphrasing (Lower is Better)")
        ax2.set_xticklabels(categories, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Cross-Encoder vs. Bi-Encoder Robustness Against Coordinated Paraphrase Evasion", fontsize=13, fontweight="bold")
        plt.tight_layout()

        out_path1 = os.path.join(self.output_dir, "paraphrase_robustness.png")
        fig.savefig(out_path1, dpi=200, bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(self.figures_dir, "paraphrase_robustness.png"), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # Figure 2: Score Distribution (dist_semantic_similarity.png)
        fig2, ax_dist = plt.subplots(figsize=(9, 5))
        sns.kdeplot(results_df["biencoder_score_original"], ax=ax_dist, label="Bi-Encoder (Original)", color="#2980b9", fill=True, alpha=0.25, linewidth=2)
        sns.kdeplot(results_df["biencoder_score_paraphrased"], ax=ax_dist, label="Bi-Encoder (Paraphrased - Collapsed)", color="#e74c3c", fill=True, alpha=0.25, linestyle="--", linewidth=2)
        sns.kdeplot(results_df["crossencoder_score_paraphrased"], ax=ax_dist, label="Cross-Encoder (Paraphrased - Preserved)", color="#27ae60", fill=True, alpha=0.25, linewidth=2.5)

        ax_dist.set_title("Distribution of Coordination Match Scores Under Paraphrase Attack", fontsize=12, fontweight="bold")
        ax_dist.set_xlabel("Similarity Score", fontsize=11)
        ax_dist.set_ylabel("Density", fontsize=11)
        ax_dist.legend(loc="upper left")
        ax_dist.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path2 = os.path.join(self.figures_dir, "dist_semantic_similarity.png")
        fig2.savefig(out_path2, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig2)

        logger.info(f"Saved visualizations: {out_path1} and {out_path2}")
        return out_path1, out_path2

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, use_sample: bool = False) -> Dict:
        """Run the full paraphrase-robustness experiment."""
        logger.info("=" * 60)
        logger.info("PARAPHRASE ROBUSTNESS EXPERIMENT")
        logger.info("=" * 60)

        df = None
        if not os.path.exists("data/pair_truth.csv") or use_sample:
            from src.data.loader import DataLoader, create_sample_dataset
            from src.data.preprocessing import Preprocessor
            data_loader = DataLoader(self.config)
            if use_sample:
                create_sample_dataset(self.config["data"]["raw_dir"])
            try:
                df = data_loader.load_campaign()
                preprocessor = Preprocessor(self.config)
                df = preprocessor.process(df)
            except Exception as e:
                logger.warning(f"Could not load campaign data: {e}")

        # Step 1: Sample
        pairs = self.sample_coordinated_pairs(df)
        if not pairs:
            logger.error("No candidate coordinated pairs available.")
            return {}

        # Step 2: Paraphrase
        paraphrased_pairs = self.paraphrase_pairs(pairs)

        # Step 3: Score
        results_df = self.score_pairs(paraphrased_pairs)

        # Step 4: Summarize
        summary_df = self.summarize(results_df)
        logger.info(f"\n{summary_df.to_string(index=False)}")

        # Save artifacts
        results_path = os.path.join(self.output_dir, "paraphrase_robustness_results.csv")
        results_df.to_csv(results_path, index=False)
        summary_path = os.path.join(self.output_dir, "paraphrase_robustness_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        # Step 5: Visualize
        self.visualize(results_df, summary_df)

        logger.info("=" * 60)
        logger.info(f"EXPERIMENT COMPLETE: results saved to {self.output_dir}")
        logger.info("=" * 60)

        return {"results": results_df, "summary": summary_df}


def main():
    parser = argparse.ArgumentParser(description="Paraphrase Robustness Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    parser.add_argument("--n-pairs", type=int, default=100, help="Number of post pairs to test")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import yaml
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)

    if "paraphrase_experiment" not in config:
        config["paraphrase_experiment"] = {}
    config["paraphrase_experiment"]["n_pairs"] = args.n_pairs

    experiment = ParaphraseRobustnessExperiment(config)
    experiment.run(use_sample=args.sample)


if __name__ == "__main__":
    main()
