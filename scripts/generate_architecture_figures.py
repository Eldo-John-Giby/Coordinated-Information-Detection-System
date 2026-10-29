"""
scripts/generate_architecture_figures.py

Generates professional, publication-quality architecture figures for the research paper:
1. results/paper/figures/system_architecture.png
2. results/paper/figures/retrieve_then_rerank.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure output directory exists
os.makedirs("results/paper/figures", exist_ok=True)

def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis('off')
    
    # Background color
    fig.patch.set_facecolor('#FAFAFB')
    ax.set_facecolor('#FAFAFB')

    # Draw Stage Containers
    stages = [
        {"title": "1. Data Ingestion & Partitioning", "x": 0.03, "y": 0.58, "w": 0.28, "h": 0.35, "color": "#E8F0FE", "border": "#4285F4"},
        {"title": "2. Retrieve-then-Rerank NLP", "x": 0.35, "y": 0.58, "w": 0.30, "h": 0.35, "color": "#E6F4EA", "border": "#34A853"},
        {"title": "3. Multi-Signal Graph Fusion", "x": 0.69, "y": 0.58, "w": 0.28, "h": 0.35, "color": "#FEF7E0", "border": "#FBBC04"},
        {"title": "4. Graph Construction & Topology", "x": 0.03, "y": 0.10, "w": 0.28, "h": 0.38, "color": "#F3E8FD", "border": "#A142F4"},
        {"title": "5. Dual-Engine Modeling", "x": 0.35, "y": 0.10, "w": 0.30, "h": 0.38, "color": "#FCE8E6", "border": "#EA4335"},
        {"title": "6. Forensic Analysis & Serving", "x": 0.69, "y": 0.10, "w": 0.28, "h": 0.38, "color": "#E0F2F1", "border": "#009688"}
    ]

    for s in stages:
        # Container box
        rect = patches.FancyBboxPatch((s["x"], s["y"]), s["w"], s["h"],
                                      boxstyle="round,pad=0.015,rounding_size=0.02",
                                      linewidth=1.8, edgecolor=s["border"], facecolor=s["color"])
        ax.add_patch(rect)
        # Header text
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"] - 0.035, s["title"],
                ha='center', va='center', fontsize=11, fontweight='bold', color='#202124')

    # Stage 1 Sub-items
    ax.text(0.17, 0.81, "Zenodo Archives\n(Honduras & UAE Campaigns)", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.17, 0.70, "Preprocessing & Schema Audit\nString ID Canonicalization", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.17, 0.61, "Canonical 70/15/15 Account Split\n(Zero Cross-Partition Identity Leakage)", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#E8EAED', edgecolor='#5F6368', linestyle='--'))

    # Stage 2 Sub-items
    ax.text(0.50, 0.82, "MiniLM Bi-Encoder (384d Dense Embeddings)", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.50, 0.72, "FAISS IndexFlatIP Retrieval Floor (cos >= 0.30)\nAggregates Top-3 Candidate Post Pairs per Dyad", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.50, 0.61, "DeBERTa-v3 / MS-MARCO Cross-Encoder\nPrecision Reranking (Budget Cap: 20k pairs)", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#E6F4EA', edgecolor='#137333', linewidth=1.2))

    # Stage 3 Sub-items
    ax.text(0.83, 0.82, "Temporal Synchronicity\nExp Decay (tau=3600s, 5m bursts)", ha='center', va='center', fontsize=8.2,
            bbox=dict(boxstyle='square,pad=0.25', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.83, 0.72, "Digital Asset Jaccard Sharing\n(URLs, Hashtags, Mentions, Reposts)", ha='center', va='center', fontsize=8.2,
            bbox=dict(boxstyle='square,pad=0.25', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.83, 0.61, "Composite Calibration\nW(u,v) = sum(w_k * S_k), sum(w_k)=1.0", ha='center', va='center', fontsize=8.2,
            bbox=dict(boxstyle='square,pad=0.25', facecolor='#FEF7E0', edgecolor='#B06000', linewidth=1.2))

    # Stage 4 Sub-items
    ax.text(0.17, 0.38, "Coordination Edge Thresholding\nFilter: W(u,v) >= 0.30", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.17, 0.27, "Degree Pruning & Sanitization\nMax Degree <= 50, No Self-Loops", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.17, 0.15, "Dual Output Graph\nNetworkX Graph + PyTorch Geometric Data", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#F3E8FD', edgecolor='#7627BB'))

    # Stage 5 Sub-items
    ax.text(0.50, 0.38, "Unsupervised Modularity Partitioning\nLouvain Community Detection", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.50, 0.27, "Spatial Inductive GNNs\nFull-Batch GraphSAGE (F1=0.962) & 2-Layer GCN", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#FCE8E6', edgecolor='#C5221F', linewidth=1.2))
    ax.text(0.50, 0.15, "Empirical Evaluation Protocol\n5 Seeds, 95% Bootstrap CIs, Adversarial Stress A1-A3", ha='center', va='center', fontsize=8.2,
            bbox=dict(boxstyle='square,pad=0.25', facecolor='white', edgecolor='#BDC1C6'))

    # Stage 6 Sub-items
    ax.text(0.83, 0.38, "Hybrid Suspiciousness Scoring\nDensity + Coordination + GNN Probability", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.83, 0.27, "Forensic Audit Narratives\nInterpretable Breakdown across 6 Channels", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#BDC1C6'))
    ax.text(0.83, 0.15, "Serving & Visualization\nInteractive Streamlit Dashboard + Exported Tables", ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#E0F2F1', edgecolor='#00796B'))

    # Connecting Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#5F6368", shrinkA=4, shrinkB=4)
    # Stage 1 -> Stage 2
    ax.annotate("", xy=(0.35, 0.75), xytext=(0.31, 0.75), arrowprops=arrow_props)
    # Stage 2 -> Stage 3
    ax.annotate("", xy=(0.69, 0.75), xytext=(0.65, 0.75), arrowprops=arrow_props)
    # Stage 3 -> Stage 4
    ax.annotate("", xy=(0.17, 0.48), xytext=(0.83, 0.58),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#5F6368", connectionstyle="arc3,rad=-0.25"))
    # Stage 4 -> Stage 5
    ax.annotate("", xy=(0.35, 0.29), xytext=(0.31, 0.29), arrowprops=arrow_props)
    # Stage 5 -> Stage 6
    ax.annotate("", xy=(0.69, 0.29), xytext=(0.65, 0.29), arrowprops=arrow_props)

    plt.tight_layout()
    output_path = "results/paper/figures/system_architecture.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

def generate_retrieve_then_rerank_diagram():
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Steps
    steps = [
        {"num": "Step 1", "title": "Dense Bi-Encoder Embedding", "desc": "Map post texts to 384d vectors\nusing all-MiniLM-L6-v2", "x": 0.04, "color": "#E8F0FE", "border": "#1A73E8"},
        {"num": "Step 2", "title": "FAISS High-Recall Indexing", "desc": "IndexFlatIP inner product search\nFloor: cosine >= 0.30 (Top-20)", "x": 0.28, "color": "#FEF7E0", "border": "#F29900"},
        {"num": "Step 3", "title": "Dyad Candidate Selection", "desc": "Aggregate top-3 post pairs\nper unique account dyad (u, v)", "x": 0.52, "color": "#F3E8FD", "border": "#9334E6"},
        {"num": "Step 4", "title": "Cross-Encoder Precision Rerank", "desc": "Full token cross-attention\nCap: 20,000 pairs (Pareto gate)", "x": 0.76, "color": "#E6F4EA", "border": "#188038"}
    ]

    for s in steps:
        rect = patches.FancyBboxPatch((s["x"], 0.22), 0.20, 0.65,
                                      boxstyle="round,pad=0.015,rounding_size=0.03",
                                      linewidth=1.8, edgecolor=s["border"], facecolor=s["color"])
        ax.add_patch(rect)
        ax.text(s["x"] + 0.10, 0.78, s["num"], ha='center', va='center', fontsize=9.5, fontweight='bold', color=s["border"])
        ax.text(s["x"] + 0.10, 0.67, s["title"], ha='center', va='center', fontsize=10.5, fontweight='bold', color='#202124')
        ax.text(s["x"] + 0.10, 0.45, s["desc"], ha='center', va='center', fontsize=9, color='#3C4043', linespacing=1.4)

    # Output banner below
    banner = patches.FancyBboxPatch((0.15, 0.05), 0.70, 0.12,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    linewidth=1.2, edgecolor='#137333', facecolor='#E6F4EA')
    ax.add_patch(banner)
    ax.text(0.50, 0.11, "Output: Semantic Coordination Score S_sem(u,v) = max(s_cross) in [0, 1] for all candidate dyads",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#137333')

    # Connecting arrows
    arrow_props = dict(arrowstyle="->", lw=2.2, color="#5F6368", shrinkA=5, shrinkB=5)
    ax.annotate("", xy=(0.28, 0.55), xytext=(0.24, 0.55), arrowprops=arrow_props)
    ax.annotate("", xy=(0.52, 0.55), xytext=(0.48, 0.55), arrowprops=arrow_props)
    ax.annotate("", xy=(0.76, 0.55), xytext=(0.72, 0.55), arrowprops=arrow_props)

    plt.tight_layout()
    output_path = "results/paper/figures/retrieve_then_rerank.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_system_architecture()
    generate_retrieve_then_rerank_diagram()
