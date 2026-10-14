#!/usr/bin/env python3
"""
Streamlit Dashboard for Coordinated Misinformation Network Detection.

Interactive visualization of:
- Suspicious account groups
- Network graphs
- Coordination scores
- Feature explanations
- Example posts
"""

import json
import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Coordinated Network Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 20px;
        border-radius: 10px;
        color: #ffffff !important;
        text-align: center;
        margin: 5px;
    }
    .suspicious-group {
        background-color: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid #f59e0b;
        color: inherit !important;
        padding: 12px;
        margin: 8px 0;
        border-radius: 6px;
    }
    .evidence-item {
        background-color: rgba(148, 163, 184, 0.12);
        color: inherit !important;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 13px;
        line-height: 1.5;
        word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)


def load_results(results_dir: str = "results"):
    """Load pipeline results from disk."""
    results_path = os.path.join(results_dir, "pipeline_results.json")
    if not os.path.exists(results_path):
        return None

    with open(results_path, "r") as f:
        return json.load(f)


def load_coordination_weights(config_path: str = "configs/config.yaml"):
    """Read coordination weights from config.yaml (item 18).

    Previously Tab 5 plotted a hardcoded dict, so editing the config's
    coordination weights did not change what the dashboard showed.
    """
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        weights = (config.get("coordination", {}) or {}).get("weights", {})
        if weights:
            return {
                "Semantic": float(weights.get("semantic", 0.0)),
                "Temporal": float(weights.get("temporal", 0.0)),
                "URL": float(weights.get("url", 0.0)),
                "Hashtag": float(weights.get("hashtag", 0.0)),
                "Mention": float(weights.get("mention", 0.0)),
                "Repost": float(weights.get("repost", 0.0)),
            }
    except FileNotFoundError:
        st.warning(f"Config not found at {config_path}; cannot display live weights")
    except Exception as e:
        st.warning(f"Could not load weights from config: {e}")
    return None


def load_graph(results_dir: str = "results"):
    """Load the persisted coordination graph (item 16 artifact)."""
    path = os.path.join(results_dir, "artifacts", "coordination_graph.pkl")
    if not os.path.exists(path):
        return None
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load persisted graph: {e}")
        return None


def load_data(processed_dir: str = "data/processed"):
    """Load processed data."""
    csv_path = os.path.join(processed_dir, "processed_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, low_memory=False)
    return None


def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert hex color string to rgba CSS format."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(100, 116, 139, {alpha})"


def create_network_plot(
    G: nx.Graph,
    title: str = "Coordination Network",
    node_size_scale: float = 1.0,
    spacing_factor: float = 2.0,
    hide_isolates: bool = False,
    show_louvain_rings: bool = True,
    show_cluster_boundaries: bool = True
):
    """Create an interactive Plotly network visualization with Louvain community rings."""
    if G.number_of_nodes() == 0:
        return go.Figure(), {}

    # Filter isolates if requested
    plot_G = G.copy()
    if hide_isolates:
        connected_nodes = [n for n in plot_G.nodes() if plot_G.degree(n) > 0]
        if connected_nodes:
            plot_G = plot_G.subgraph(connected_nodes).copy()

    n_nodes = plot_G.number_of_nodes()
    if n_nodes == 0:
        return go.Figure(), {}

    # Detect Louvain communities
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(plot_G, weight="coordination_score")
    except Exception:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comm_list = list(greedy_modularity_communities(plot_G, weight="coordination_score"))
            partition = {node: i for i, c in enumerate(comm_list) for node in c}
        except Exception:
            partition = {node: 0 for node in plot_G.nodes()}

    # Community palette and color assignment
    COMMUNITY_PALETTE = [
        "#3b82f6", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899",
        "#10b981", "#f97316", "#84cc16", "#6366f1", "#14b8a6",
        "#e11d48", "#eab308", "#d946ef", "#0ea5e9", "#a855f7", "#22c55e"
    ]
    comm_sizes = {}
    comm_io_counts = {}
    for n, c in partition.items():
        comm_sizes[c] = comm_sizes.get(c, 0) + 1
        if plot_G.nodes[n].get("is_io", False):
            comm_io_counts[c] = comm_io_counts.get(c, 0) + 1

    sorted_comms = sorted(comm_sizes.keys(), key=lambda c: comm_sizes[c], reverse=True)
    comm_to_color = {
        c: COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)]
        for i, c in enumerate(sorted_comms)
    }

    # Dynamic spring layout for optimal spacing and responsiveness
    # Adjust iterations based on node count to ensure smooth rendering
    iterations = 60 if n_nodes <= 150 else (40 if n_nodes <= 400 else 25)
    k_val = (spacing_factor / np.sqrt(max(n_nodes, 4))) * 1.5
    pos = nx.spring_layout(plot_G, k=k_val, iterations=iterations, seed=42)

    # Louvain Community cluster boundary hulls
    hull_traces = []
    if show_cluster_boundaries:
        from scipy.spatial import ConvexHull
        comm_nodes_map = {}
        for n, c in partition.items():
            comm_nodes_map.setdefault(c, []).append(n)

        for c, members in comm_nodes_map.items():
            if len(members) >= 3:
                pts = np.array([pos[m] for m in members])
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    centroid = np.mean(hull_pts, axis=0)
                    diffs = hull_pts - centroid
                    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    padded = hull_pts + diffs * 0.18 + (diffs / norms) * 0.04
                    padded = np.vstack([padded, padded[0]])

                    color = comm_to_color.get(c, "#94a3b8")
                    hull_traces.append(
                        go.Scatter(
                            x=padded[:, 0].tolist(),
                            y=padded[:, 1].tolist(),
                            mode="lines",
                            fill="toself",
                            fillcolor=hex_to_rgba(color, 0.08),
                            line=dict(color=color, width=2, dash="dash"),
                            hoverinfo="text",
                            hovertext=f"<b>Louvain Community #{c} Ring</b><br>Members in view: {len(members)} accounts",
                            name=f"Community #{c} Ring",
                            showlegend=False
                        )
                    )
                except Exception:
                    pass

    # Node attributes
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_border_colors = []
    node_border_widths = []
    node_text = []

    degrees = dict(plot_G.degree())
    max_deg = max(degrees.values()) if degrees and max(degrees.values()) > 0 else 1

    for node in plot_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        is_io = plot_G.nodes[node].get("is_io", False)
        deg = degrees.get(node, 0)
        comm_id = partition.get(node, 0)
        comm_color = comm_to_color.get(comm_id, "#3b82f6")
        comm_size = comm_sizes.get(comm_id, 1)

        # Core node fill color: Crimson for IO, Emerald for Control
        color = "#ef4444" if is_io else "#10b981"
        node_colors.append(color)

        # Scaled, bounded node size (between 6px and 16px base)
        norm_deg = deg / max_deg
        size = (7 + norm_deg * 11) * node_size_scale
        node_sizes.append(size)

        # Louvain Community outer ring border
        if show_louvain_rings:
            node_border_colors.append(comm_color)
            node_border_widths.append(3.2)
        else:
            node_border_colors.append("#ffffff")
            node_border_widths.append(1.2)

        node_text.append(
            f"<b>Account:</b> {node}<br>"
            f"<b>Status:</b> {'[FLAGGED] Inauthentic / IO' if is_io else '[OK] Organic / Control'}<br>"
            f"<b>Louvain Community:</b> Community #{comm_id} ({comm_size} accounts in view)<br>"
            f"<b>Connections:</b> {deg}<br>"
            f"<b>Posts:</b> {plot_G.nodes[node].get('post_count', 'N/A')}"
        )

    # Node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=node_border_widths, color=node_border_colors),
            opacity=0.92
        ),
        name="Accounts"
    )

    # Edge trace
    edge_x = []
    edge_y = []
    edge_text = []

    for u, v, data in plot_G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        score = data.get("coordination_score", 0)
        edge_text.append(f"Coordination: {score:.3f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(
            width=1.0,
            color="rgba(148, 163, 184, 0.45)"
        ),
        name="Coordination Links"
    )

    all_traces = hull_traces + [edge_trace, node_trace]

    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest",
            dragmode="pan",  # Allows free dragging/panning across the screen
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
            height=650,
            uirevision="constant_view"
        )
    )

    comm_summary = {
        "sorted_comms": sorted_comms,
        "comm_sizes": comm_sizes,
        "comm_io_counts": comm_io_counts,
        "comm_to_color": comm_to_color,
    }

    return fig, comm_summary


def main():
    st.title("Coordinated Misinformation Network Detection")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Dashboard Controls")

    # Load results
    results_dir = st.sidebar.text_input("Results Directory", "results")
    results = load_results(results_dir)

    if results is None:
        st.warning(
            "No results found. Please run the pipeline first:\n\n"
            "```bash\npython train.py --sample\n```"
        )
        st.info(
            "This will create a sample dataset and run the full pipeline."
        )
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Suspicious Groups", "Network Graph",
        "Model Comparison", "Explanations"
    ])

    # Tab 1: Overview
    with tab1:
        st.header("System Overview")

        col1, col2, col3, col4 = st.columns(4)

        data_stats = results.get("data_stats", {})
        with col1:
            st.metric("Total Posts", f"{data_stats.get('total_posts', 0):,}")
        with col2:
            st.metric("Unique Accounts", f"{data_stats.get('unique_accounts', 0):,}")
        with col3:
            graph_stats = results.get("graph_stats", {})
            st.metric("Graph Nodes", f"{graph_stats.get('num_nodes', 0):,}")
        with col4:
            st.metric("Graph Edges", f"{graph_stats.get('num_edges', 0):,}")

        # Model comparison table
        st.subheader("Model Performance Comparison")
        comparison = results.get("model_comparison", {})
        if comparison:
            comp_df = pd.DataFrame([
                {"Model": k, **{mk: f"{mv:.4f}" if isinstance(mv, float) else mv
                               for mk, mv in v.items() if isinstance(mv, (int, float))}}
                for k, v in comparison.items()
            ])
            st.dataframe(comp_df, use_container_width=True)

    # Tab 2: Suspicious Groups
    with tab2:
        st.header("Suspicious Account Groups")

        groups = results.get("suspicious_groups", [])
        if not groups:
            st.info("No suspicious groups detected")
            return

        st.write(f"**{len(groups)} suspicious groups detected**")

        for i, group in enumerate(groups):
            with st.expander(
                f"[FLAGGED] Group {group.get('group_id', i)}: "
                f"{group.get('num_accounts', 0)} accounts, "
                f"Score: {group.get('coordination_score', 0):.3f}",
                expanded=(i == 0)
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write("**Accounts:**")
                    accounts = group.get("accounts", [])
                    for acc in accounts:
                        detail = next(
                            (d for d in group.get("account_details", [])
                             if d["accountid"] == acc),
                            {}
                        )
                        is_io = detail.get("is_io", False)
                        icon = "[IO]" if is_io else "[Control]"
                        st.write(f"{icon} {acc} "
                                f"(Posts: {detail.get('post_count', 'N/A')})")

                with col2:
                    st.write("**Metrics:**")
                    st.write(f"Coordination Score: "
                            f"{group.get('coordination_score', 0):.3f}")
                    st.write(f"Suspiciousness: "
                            f"{group.get('suspiciousness_score', 0):.3f}")
                    st.write(f"IO Ratio: "
                            f"{group.get('io_ratio', 0):.1%}")

                st.write("**Evidence:**")
                evidence = group.get("evidence", [])
                if evidence:
                    for ev in evidence:
                        st.markdown(
                            f"<div class='evidence-item'>"
                            f"{ev.get('description', 'N/A')}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No strong pairwise evidence found")

    # Tab 3: Network Graph
    with tab3:
        st.header("Coordination Network Graph")

        # Interactive controls for graph exploration
        with st.expander("Graph Display Controls", expanded=True):
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
            with ctrl_col1:
                node_limit = st.slider(
                    "Max Nodes to Show",
                    min_value=20,
                    max_value=1000,
                    value=120,
                    step=20,
                    help="Controls how many accounts to render (sorted by connectivity). Nodes > 400 may require a few extra seconds for force-directed layout computation."
                )
            with ctrl_col2:
                node_size_scale = st.slider("Node Size Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            with ctrl_col3:
                spacing_factor = st.slider("Layout Spacing", min_value=1.0, max_value=4.0, value=2.2, step=0.2)

            opt_col1, opt_col2, opt_col3 = st.columns(3)
            with opt_col1:
                show_louvain_rings = st.checkbox(
                    "Show Louvain Community Rings",
                    value=True,
                    help="Draws high-visibility colored outer rings on account nodes representing their Louvain community."
                )
            with opt_col2:
                show_cluster_boundaries = st.checkbox(
                    "Show Community Boundary Rings",
                    value=True,
                    help="Draws dashed boundary rings and soft cluster hulls around multi-account Louvain communities."
                )
            with opt_col3:
                hide_isolates = st.checkbox("Hide Isolated Accounts", value=True)

        G = load_graph(results_dir)
        if G is not None and G.number_of_nodes() > 0:
            total_graph_nodes = results.get("graph_stats", {}).get("num_nodes", G.number_of_nodes())
            if G.number_of_nodes() > node_limit:
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:node_limit]
                G = G.subgraph(top_nodes).copy()
                st.caption(
                    f"Showing top {len(top_nodes)} most-connected accounts (out of {total_graph_nodes:,} total in graph)."
                )

            fig, comm_summary = create_network_plot(
                G=G,
                title="Interactive Account Coordination Network",
                node_size_scale=node_size_scale,
                spacing_factor=spacing_factor,
                hide_isolates=hide_isolates,
                show_louvain_rings=show_louvain_rings,
                show_cluster_boundaries=show_cluster_boundaries
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "modeBarButtonsToAdd": ["drawline", "drawcircle", "eraseshape"],
                    "displaylogo": False,
                },
                key="interactive_network",
            )
            st.info(
                "**Legend & Navigation:** Node **inner fill** = Red (Inauthentic / IO) vs Green (Organic / Control). "
                "Node **outer ring** = Louvain Community. Click & drag anywhere to **pan**. Scroll/pinch to **zoom**."
            )

            # Display Louvain community badges
            if show_louvain_rings and comm_summary.get("sorted_comms"):
                top_comms = comm_summary["sorted_comms"][:8]
                comm_badges_html = " ".join([
                    f"<span style='display:inline-block; margin:2px 6px; padding:3px 9px; "
                    f"border-radius:12px; background:{comm_summary['comm_to_color'][c]}22; "
                    f"border:1.5px solid {comm_summary['comm_to_color'][c]}; font-size:12px;'>"
                    f"<b>Community #{c}</b>: {comm_summary['comm_sizes'][c]} accounts "
                    f"({comm_summary['comm_io_counts'].get(c, 0)} IO)</span>"
                    for c in top_comms
                ])
                st.markdown(
                    f"<div style='margin-top:4px; margin-bottom:12px;'>"
                    f"<b>Top Louvain Communities in View:</b><br>{comm_badges_html}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info(
                "Persisted graph not found (results/artifacts/coordination_graph.pkl). "
                "Falling back to the static image."
            )
            graph_path = os.path.join(results_dir, "figures", "network_graph.png")
            if os.path.exists(graph_path):
                st.image(graph_path, caption="Coordination Network (static)")

        # Graph statistics
        graph_stats = results.get("graph_stats", {})
        if graph_stats:
            st.subheader("Graph Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Density:** {graph_stats.get('density', 0):.4f}")
                st.write(f"**Avg Degree:** {graph_stats.get('avg_degree', 0):.2f}")
            with col2:
                st.write(f"**Components:** {graph_stats.get('num_components', 0)}")
                st.write(f"**Avg Clustering:** {graph_stats.get('avg_clustering', 0):.4f}")
            with col3:
                st.write(f"**Max Degree:** {graph_stats.get('max_degree', 0)}")
                st.write(f"**Median Degree:** {graph_stats.get('median_degree', 0):.1f}")

    # Tab 4: Model Comparison
    with tab4:
        st.header("Model Comparison")

        comparison = results.get("model_comparison", {})
        if comparison:
            # ROC curves
            st.subheader("Performance Metrics")

            models = list(comparison.keys())
            metrics_to_show = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

            chart_data = []
            for model_name, metrics in comparison.items():
                for metric in metrics_to_show:
                    if metric in metrics:
                        chart_data.append({
                            "Model": model_name,
                            "Metric": metric.upper(),
                            "Value": metrics[metric]
                        })

            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                fig = px.bar(
                    chart_df, x="Model", y="Value", color="Metric",
                    barmode="group", title="Model Performance Comparison"
                )
                st.plotly_chart(fig, use_container_width=True, key="model_comparison_chart")
        else:
            st.info("No model comparison data available")

    # Tab 5: Explanations
    with tab5:
        st.header("Explanations")

        groups = results.get("suspicious_groups", [])
        if not groups:
            st.info("No groups to explain")
            return

        # Explanation for each group
        for group in groups:
            st.subheader(f"Group {group.get('group_id', 'N/A')}")

            st.write("**Summary:**")
            st.write(group.get("summary", "No summary available"))

            # Feature weights (item 18: live from config.yaml, not hardcoded)
            st.write("**Detection Weights (from configs/config.yaml):**")
            weights = st.session_state.get("coordination_weights")
            if not weights:
                weights = load_coordination_weights()
                st.session_state["coordination_weights"] = weights

            # Per-group signal breakdown from results (item 12)
            breakdown = group.get("signal_breakdown", {})
            if breakdown:
                labels = list(breakdown.keys())
                mean_scores = [breakdown[k]["mean_score"] for k in labels]
                contributions = [breakdown[k]["weighted_contribution"] for k in labels]
                fig = go.Figure(go.Bar(
                    x=contributions,
                    y=labels,
                    orientation="h",
                    marker_color="#e67e22",
                    customdata=list(zip(labels, mean_scores)),
                    hovertemplate=(
                        "%{customdata[0]}: mean %{customdata[1]:.3f}, "
                        "contribution %{x:.3f}<extra></extra>"
                    ),
                ))
                fig.update_layout(
                    title="Per-Signal Contribution to This Group's Score",
                    xaxis_title="Weighted contribution",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"signal_chart_{group.get('group_id', 'N/A')}")

            if weights:
                fig = go.Figure(go.Bar(
                    x=list(weights.values()),
                    y=list(weights.keys()),
                    orientation="h",
                    marker_color="#3498db"
                ))
                fig.update_layout(
                    title="Feature Weights for Coordination Score",
                    xaxis_title="Weight",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"weights_chart_{group.get('group_id', 'N/A')}")

            st.markdown("---")


if __name__ == "__main__":
    main()
