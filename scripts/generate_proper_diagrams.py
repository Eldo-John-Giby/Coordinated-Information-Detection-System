"""
scripts/generate_proper_diagrams.py

Generates publication-quality, professional architecture diagrams for Section IV:
1. results/paper/figures/system_architecture.png (Clean 6-Stage Modular Pipeline)
2. results/paper/figures/retrieve_then_rerank.png (Detailed Two-Stage Retrieve-then-Rerank Flow)
3. results/paper/figures/multisignal_fusion_diagram.png (The 6-Channel Coordination Fusion Engine)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

os.makedirs("results/paper/figures", exist_ok=True)

# Set global typography to Calibri
plt.rcParams['font.sans-serif'] = 'Calibri'
plt.rcParams['font.family'] = 'sans-serif'


def draw_card(ax, x, y, w, h, bg_color, border_color, title, badge_text, badge_color="#2563EB"):
    """Draws a modern rounded card with header badge and title."""
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.8, edgecolor=border_color, facecolor=bg_color
    )
    ax.add_patch(card)

    # Badge Pill
    badge_w = len(badge_text) * 0.009 + 0.04
    badge = patches.FancyBboxPatch(
        (x + 0.015, y + h - 0.042), badge_w, 0.03,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=0, facecolor=badge_color
    )
    ax.add_patch(badge)
    ax.text(x + 0.015 + badge_w/2, y + h - 0.027, badge_text,
            ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

    # Card Title
    ax.text(x + 0.025 + badge_w, y + h - 0.027, title,
            ha='left', va='center', fontsize=11, fontweight='bold', color='#0F172A')


def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Stage 1: Ingestion & Splitting
    draw_card(ax, 0.02, 0.55, 0.29, 0.38, "#F1F5F9", "#64748B", "Data Ingestion & Integrity", "STAGE 1", "#475569")
    # Sub-boxes
    b1 = patches.FancyBboxPatch((0.035, 0.74), 0.26, 0.12, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.2, edgecolor='#CBD5E1', facecolor='white')
    ax.add_patch(b1)
    ax.text(0.165, 0.82, "Zenodo IO Corpus (Honduras & UAE)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E293B')
    ax.text(0.165, 0.77, "Honduras: 4,704 posts (90.8% Spanish)\nUAE: 4,293 posts (82.6% Arabic)", ha='center', va='center', fontsize=8.2, color='#64748B')

    b2 = patches.FancyBboxPatch((0.035, 0.58), 0.26, 0.13, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.2, edgecolor='#3B82F6', facecolor='#EFF6FF')
    ax.add_patch(b2)
    ax.text(0.165, 0.66, "Canonical Account-Level Split (70/15/15)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1D4ED8')
    ax.text(0.165, 0.61, "Strict ID assignment before edge generation\nZero cross-partition identity leakage", ha='center', va='center', fontsize=8.2, color='#3B82F6')

    # Stage 2: Retrieve-then-Rerank NLP
    draw_card(ax, 0.35, 0.55, 0.30, 0.38, "#ECFDF5", "#10B981", "Retrieve-then-Rerank NLP", "STAGE 2", "#059669")
    b3 = patches.FancyBboxPatch((0.365, 0.74), 0.27, 0.12, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.2, edgecolor='#A7F3D0', facecolor='white')
    ax.add_patch(b3)
    ax.text(0.50, 0.82, "Stage 1: Dense Bi-Encoder Indexing", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#065F46')
    ax.text(0.50, 0.77, "all-MiniLM-L6-v2 (384d Dense Vectors)\nFAISS IndexFlatIP (Retrieval Floor: cos >= 0.30)", ha='center', va='center', fontsize=8.2, color='#047857')

    b4 = patches.FancyBboxPatch((0.365, 0.58), 0.27, 0.13, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.2, edgecolor='#059669', facecolor='#D1FAE5')
    ax.add_patch(b4)
    ax.text(0.50, 0.66, "Stage 2: Cross-Encoder Precision Rerank", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#064E3B')
    ax.text(0.50, 0.61, "ms-marco-MiniLM-L-6-v2 / DeBERTa-v3\nBudget Cap: N <= 20,000 pairs (Pareto gate)", ha='center', va='center', fontsize=8.2, color='#065F46')

    # Stage 3: Multi-Signal Behavioral Fusion
    draw_card(ax, 0.69, 0.55, 0.29, 0.38, "#FFFBEB", "#F59E0B", "6-Signal Behavioral Fusion", "STAGE 3", "#D97706")
    b5 = patches.FancyBboxPatch((0.705, 0.74), 0.26, 0.12, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.2, edgecolor='#FDE68A', facecolor='white')
    ax.add_patch(b5)
    ax.text(0.835, 0.82, "Multi-Channel Edge Tensor", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#92400E')
    ax.text(0.835, 0.77, "Semantic (0.25) + Temporal Burst (0.20)\nHashtags (0.20) + Reposts (0.15)\nURLs (0.10) + Mentions (0.10)", ha='center', va='center', fontsize=8.0, color='#B45309')

    b6 = patches.FancyBboxPatch((0.705, 0.58), 0.26, 0.13, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.2, edgecolor='#D97706', facecolor='#FEF3C7')
    ax.add_patch(b6)
    ax.text(0.835, 0.66, "Calibrated Composite Weight W(u,v)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#78350F')
    ax.text(0.835, 0.61, "W(u,v) = sum(w_k * S_k(u,v)) >= 0.30\nDegree Pruning <= 50, No Self-Loops", ha='center', va='center', fontsize=8.2, color='#92400E')

    # Stage 4: Coordination Graph Construction
    draw_card(ax, 0.02, 0.08, 0.29, 0.39, "#FAF5FF", "#A855F7", "Graph Construction & Topology", "STAGE 4", "#9333EA")
    b7 = patches.FancyBboxPatch((0.035, 0.28), 0.26, 0.12, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.2, edgecolor='#E9D5FF', facecolor='white')
    ax.add_patch(b7)
    ax.text(0.165, 0.36, "Undirected Latent Network G=(V,E,W)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#6B21A8')
    ax.text(0.165, 0.31, "V: Accounts with 7 behavioral node features\nE: Dyads with composite weight W >= 0.30", ha='center', va='center', fontsize=8.2, color='#7E22CE')

    b8 = patches.FancyBboxPatch((0.035, 0.12), 0.26, 0.13, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.2, edgecolor='#9333EA', facecolor='#F3E8FF')
    ax.add_patch(b8)
    ax.text(0.165, 0.20, "Dual Graph Export", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#581C87')
    ax.text(0.165, 0.15, "NetworkX Graph (Topological analytics)\nPyG Data(x, edge_index, edge_attr, y)", ha='center', va='center', fontsize=8.2, color='#6B21A8')

    # Stage 5: Dual-Engine Modeling
    draw_card(ax, 0.35, 0.08, 0.30, 0.39, "#EFF6FF", "#3B82F6", "Dual-Engine Detection & Learning", "STAGE 5", "#2563EB")
    b9 = patches.FancyBboxPatch((0.365, 0.28), 0.27, 0.12, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.2, edgecolor='#BFDBFE', facecolor='white')
    ax.add_patch(b9)
    ax.text(0.50, 0.36, "Unsupervised Community Recovery", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E40AF')
    ax.text(0.50, 0.31, "Louvain Modularity Partitioning\nClusters coordinated brigades without label bias", ha='center', va='center', fontsize=8.2, color='#1D4ED8')

    b10 = patches.FancyBboxPatch((0.365, 0.12), 0.27, 0.13, boxstyle="round,pad=0.01,rounding_size=0.015",
                                 linewidth=1.2, edgecolor='#2563EB', facecolor='#DBEAFE')
    ax.add_patch(b10)
    ax.text(0.50, 0.20, "Supervised Graph Neural Networks", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E3A8A')
    ax.text(0.50, 0.15, "Full-Batch GraphSAGE (F1=0.962, AUC=0.988)\n2-Layer Spectral GCN (F1=0.832, AUC=0.932)", ha='center', va='center', fontsize=8.2, color='#1E40AF')

    # Stage 6: Forensic Explainability & Serving
    draw_card(ax, 0.69, 0.08, 0.29, 0.39, "#F0FDF4", "#22C55E", "Forensic Audit & Evaluation", "STAGE 6", "#16A34A")
    b11 = patches.FancyBboxPatch((0.705, 0.28), 0.26, 0.12, boxstyle="round,pad=0.01,rounding_size=0.015",
                                 linewidth=1.2, edgecolor='#BBF7D0', facecolor='white')
    ax.add_patch(b11)
    ax.text(0.835, 0.36, "Adversarial Stress Testing & Transfer", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#14532D')
    ax.text(0.835, 0.31, "Adversary Tiers A1-A3 (Paraphrase + Jitter)\nZero-Shot Transfer: Honduras -> UAE (AUC=0.910)", ha='center', va='center', fontsize=8.2, color='#15803D')

    b12 = patches.FancyBboxPatch((0.705, 0.12), 0.26, 0.13, boxstyle="round,pad=0.01,rounding_size=0.015",
                                 linewidth=1.2, edgecolor='#16A34A', facecolor='#DCFCE7')
    ax.add_patch(b12)
    ax.text(0.835, 0.20, "Forensic Narratives & Dashboard", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#052E16')
    ax.text(0.835, 0.15, "Automated 6-channel audit trail synthesis\nInteractive Streamlit network exploration", ha='center', va='center', fontsize=8.2, color='#14532D')

    # Connecting Flow Arrows
    arrow = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color="#475569")
    ax.annotate("", xy=(0.35, 0.74), xytext=(0.31, 0.74), arrowprops=arrow)
    ax.annotate("", xy=(0.69, 0.74), xytext=(0.65, 0.74), arrowprops=arrow)
    
    # Curved connector Stage 3 -> Stage 4
    ax.annotate("", xy=(0.165, 0.47), xytext=(0.835, 0.55),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.0, color="#475569", connectionstyle="arc3,rad=-0.28"))
    
    ax.annotate("", xy=(0.35, 0.275), xytext=(0.31, 0.275), arrowprops=arrow)
    ax.annotate("", xy=(0.69, 0.275), xytext=(0.65, 0.275), arrowprops=arrow)

    plt.tight_layout()
    output_path = "results/paper/figures/system_architecture.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated sleek {output_path}")


def generate_retrieve_then_rerank_diagram():
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    steps = [
        {
            "badge": "STEP 1", "badge_color": "#2563EB",
            "title": "Dense Bi-Encoder Embedding",
            "bg": "#EFF6FF", "border": "#3B82F6",
            "bullets": [
                "Input: Raw post text c_i",
                "Model: all-MiniLM-L6-v2",
                "Output: 384d normalized vector e_i",
                "Throughput: >1,000 posts/sec (CPU)"
            ],
            "x": 0.03
        },
        {
            "badge": "STEP 2", "badge_color": "#D97706",
            "title": "FAISS High-Recall Candidate Search",
            "bg": "#FFFBEB", "border": "#F59E0B",
            "bullets": [
                "Index: IndexFlatIP (Inner Product)",
                "Query: Retrieve Top-20 nearest posts",
                "Recall Floor: cos(e_i, e_j) >= 0.30",
                "Pruning: Discards 99.4% unrelated pairs"
            ],
            "x": 0.275
        },
        {
            "badge": "STEP 3", "badge_color": "#7C3AED",
            "title": "Account-Dyad Aggregation",
            "bg": "#FAF5FF", "border": "#A855F7",
            "bullets": [
                "Map posts (p_i, p_j) to account dyads (u, v)",
                "Filter: Cross-account pairs only (u != v)",
                "Dyad Selection: Keep Top-3 post pairs",
                "Prevents single account dominance"
            ],
            "x": 0.52
        },
        {
            "badge": "STEP 4", "badge_color": "#059669",
            "title": "Cross-Encoder Precision Rerank",
            "bg": "#ECFDF5", "border": "#10B981",
            "bullets": [
                "Model: ms-marco-MiniLM / DeBERTa-v3",
                "Full token cross-attention [c_i ; c_j]",
                "Budget Gate: Hard cap at 20,000 pairs",
                "Output: s_cross in [0, 1] per pair"
            ],
            "x": 0.765
        }
    ]

    for s in steps:
        card = patches.FancyBboxPatch(
            (s["x"], 0.22), 0.21, 0.68,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            linewidth=1.8, edgecolor=s["border"], facecolor=s["bg"]
        )
        ax.add_patch(card)

        # Badge
        badge = patches.FancyBboxPatch(
            (s["x"] + 0.015, 0.81), 0.07, 0.04,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=0, facecolor=s["badge_color"]
        )
        ax.add_patch(badge)
        ax.text(s["x"] + 0.05, 0.83, s["badge"], ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

        # Title
        ax.text(s["x"] + 0.105, 0.73, s["title"], ha='center', va='center', fontsize=10.2, fontweight='bold', color='#0F172A')

        # Bullets box
        inner = patches.FancyBboxPatch(
            (s["x"] + 0.015, 0.26), 0.18, 0.41,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=1.0, edgecolor='#CBD5E1', facecolor='white'
        )
        ax.add_patch(inner)

        y_bullet = 0.62
        for b in s["bullets"]:
            ax.text(s["x"] + 0.025, y_bullet, f"• {b}", ha='left', va='center', fontsize=8.5, color='#334155')
            y_bullet -= 0.095

    # Bottom Equation Output Banner
    banner = patches.FancyBboxPatch(
        (0.12, 0.04), 0.76, 0.12,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.5, edgecolor='#059669', facecolor='#D1FAE5'
    )
    ax.add_patch(banner)
    ax.text(0.50, 0.10, "Final Semantic Score Formulation:   S_sem(u, v) = max_{(p_i, p_j) in P(u, v)} s_cross(p_i, p_j) in [0, 1]",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#065F46')
    ax.text(0.50, 0.065, "Pairs beyond N = 20,000 cap fall back to bi-encoder cosine (retains 91.2% edge recovery & 88.2% group recall)",
            ha='center', va='center', fontsize=8.5, color='#047857')

    # Flow arrows
    arrow = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color="#64748B")
    ax.annotate("", xy=(0.275, 0.56), xytext=(0.24, 0.56), arrowprops=arrow)
    ax.annotate("", xy=(0.52, 0.56), xytext=(0.485, 0.56), arrowprops=arrow)
    ax.annotate("", xy=(0.765, 0.56), xytext=(0.73, 0.56), arrowprops=arrow)

    plt.tight_layout()
    output_path = "results/paper/figures/retrieve_then_rerank.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated sleek {output_path}")


def generate_multisignal_fusion_diagram():
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    channels = [
        {"name": "1. Text Semantic Similarity", "weight": "w_1 = 0.25", "formula": "S_sem = max s_cross (Cross-Encoder)", "bg": "#EFF6FF", "border": "#3B82F6", "color": "#1D4ED8", "y": 0.82},
        {"name": "2. Temporal Synchronicity", "weight": "w_2 = 0.20", "formula": "S_temp = exp(-Δt / 3600s), burst <= 300s", "bg": "#FFFBEB", "border": "#F59E0B", "color": "#B45309", "y": 0.67},
        {"name": "3. Shared Hashtag Overlap", "weight": "w_3 = 0.20", "formula": "S_ht = Jaccard co-occurrence of deployed hashtags", "bg": "#ECFDF5", "border": "#10B981", "color": "#047857", "y": 0.52},
        {"name": "4. Shared Repost Targets", "weight": "w_4 = 0.15", "formula": "S_rep = Jaccard co-retweet cascade targets", "bg": "#FAF5FF", "border": "#A855F7", "color": "#6D28D9", "y": 0.37},
        {"name": "5. Shared Domain URLs", "weight": "w_5 = 0.10", "formula": "S_url = Jaccard overlap of shared domain links", "bg": "#FDF2F8", "border": "#EC4899", "color": "#BE185D", "y": 0.22},
        {"name": "6. Shared User Mentions", "weight": "w_6 = 0.10", "formula": "S_ment = Jaccard overlap of targeted handles", "bg": "#F0FDF4", "border": "#22C55E", "color": "#15803D", "y": 0.07}
    ]

    # Draw left channels
    for ch in channels:
        card = patches.FancyBboxPatch(
            (0.04, ch["y"]), 0.40, 0.11,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.4, edgecolor=ch["border"], facecolor=ch["bg"]
        )
        ax.add_patch(card)
        ax.text(0.06, ch["y"] + 0.075, ch["name"], ha='left', va='center', fontsize=9.5, fontweight='bold', color=ch["color"])
        ax.text(0.42, ch["y"] + 0.075, ch["weight"], ha='right', va='center', fontsize=9.0, fontweight='bold', color='#0F172A')
        ax.text(0.06, ch["y"] + 0.03, ch["formula"], ha='left', va='center', fontsize=8.5, color='#475569')

    # Draw Fusion Hub in the middle
    hub = patches.FancyBboxPatch(
        (0.53, 0.28), 0.20, 0.45,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=2.0, edgecolor='#0284C7', facecolor='#F0F9FF'
    )
    ax.add_patch(hub)
    ax.text(0.63, 0.65, "Weighted Fusion Engine", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0369A1')
    ax.text(0.63, 0.54, "W(u, v) = ∑ w_k · S_k(u, v)", ha='center', va='center', fontsize=11, fontweight='bold', color='#0C4A6E')
    ax.text(0.63, 0.46, "Constraint: ∑ w_k = 1.0", ha='center', va='center', fontsize=8.5, color='#0284C7')
    ax.text(0.63, 0.36, "Edge Filter: W(u, v) >= 0.30\nDegree Cap: k_max <= 50", ha='center', va='center', fontsize=8.5, color='#334155')

    # Draw Final Output Graph Card on the right
    out_card = patches.FancyBboxPatch(
        (0.81, 0.28), 0.16, 0.45,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=2.0, edgecolor='#7C3AED', facecolor='#FAF5FF'
    )
    ax.add_patch(out_card)
    ax.text(0.89, 0.65, "Coordination Graph", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#6D28D9')
    ax.text(0.89, 0.54, "G = (V, E, W)", ha='center', va='center', fontsize=12, fontweight='bold', color='#4C1D95')
    ax.text(0.89, 0.42, "V: 1,208 Accounts\nE: Latent Edges\nW: Coordination Weights\nZero Self-Loops", ha='center', va='center', fontsize=8.5, color='#5B21B6')

    # Connecting arrows from channels to hub
    arrow = dict(arrowstyle="->,head_width=0.3,head_length=0.5", lw=1.5, color="#94A3B8")
    for ch in channels:
        ax.annotate("", xy=(0.53, 0.50), xytext=(0.44, ch["y"] + 0.055), arrowprops=arrow)

    # Connecting arrow from hub to output
    ax.annotate("", xy=(0.81, 0.50), xytext=(0.73, 0.50),
                arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.7", lw=2.5, color="#0284C7"))

    plt.tight_layout()
    output_path = "results/paper/figures/multisignal_fusion_diagram.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated sleek {output_path}")


if __name__ == "__main__":
    generate_system_architecture()
    generate_retrieve_then_rerank_diagram()
    generate_multisignal_fusion_diagram()
