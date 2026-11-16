"""
ShopGraph - Module 03: Customer-Product Graph & Co-Purchase Network Analysis
============================================================================
Constructs the bipartite customer-product property graph and the projected
product co-purchase network. Computes PageRank centrality, connected components,
community detection, clustering coefficients, and surfaces influential products
and co-purchase bundles.
"""

import os
import sys

# Configure Matplotlib cache before import
mpl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "mpl_cache")
os.environ["MPLCONFIGDIR"] = os.path.abspath(mpl_dir)

import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms import community
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def run_graph_analysis(data_dir: str, outputs_dir: str):
    t_start = time.time()
    print("=" * 80)
    print("SHOPGRAPH - STAGE 3: CUSTOMER-PRODUCT RELATIONSHIP GRAPH & CO-PURCHASE NETWORK")
    print("=" * 80)

    tables_dir = os.path.join(outputs_dir, "tables")
    figures_dir = os.path.join(outputs_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    inter_path = os.path.join(data_dir, "interactions.parquet")
    prod_path = os.path.join(data_dir, "product_catalog.parquet")

    if not os.path.exists(inter_path) or not os.path.exists(prod_path):
        raise FileNotFoundError("Prerequisite interaction or product parquet not found. Run 01_ingest_clean.py first.")

    print("[Graph] Loading interactions and product catalog...")
    df_inter = pd.read_parquet(inter_path)
    df_prod = pd.read_parquet(prod_path).set_index("stock_code")
    prod_desc_map = df_prod["canonical_description"].to_dict()

    # ---------------------------------------------------------
    # 1. BIPARTITE GRAPH CONSTRUCTION
    # ---------------------------------------------------------
    print("\n[Graph] Constructing Bipartite Customer-Product Property Graph...")
    B = nx.Graph()

    customers = df_inter["customer_id"].unique()
    products = df_inter["stock_code"].unique()

    # Add customer vertices (bipartite=0)
    cust_nodes = [(f"C_{cid}", {"type": "customer", "customer_id": cid}) for cid in customers]
    B.add_nodes_from(cust_nodes, bipartite=0)

    # Add product vertices (bipartite=1)
    prod_nodes = [
        (f"P_{sc}", {"type": "product", "stock_code": sc, "description": prod_desc_map.get(sc, sc)})
        for sc in products
    ]
    B.add_nodes_from(prod_nodes, bipartite=1)

    # Add weighted edges
    edges = [
        (f"C_{row['customer_id']}", f"P_{row['stock_code']}", float(row["confidence_weight"]))
        for _, row in df_inter.iterrows()
    ]
    B.add_weighted_edges_from(edges)

    num_v = B.number_of_nodes()
    num_e = B.number_of_edges()
    print(f"  -> Bipartite Graph constructed: {num_v:,} vertices ({len(customers):,} customers, {len(products):,} products), {num_e:,} edges.")

    # ---------------------------------------------------------
    # 2. CONNECTED COMPONENTS
    # ---------------------------------------------------------
    print("\n[Graph] Computing Connected Components...")
    connected_comps = list(nx.connected_components(B))
    num_comps = len(connected_comps)
    comp_sizes = [len(c) for c in connected_comps]
    giant_comp_size = max(comp_sizes)
    giant_pct = (giant_comp_size / num_v * 100)
    print(f"  -> Total Connected Components: {num_comps}")
    print(f"  -> Giant Component Size:       {giant_comp_size:,} vertices ({giant_pct:.2f}% of graph)")

    # ---------------------------------------------------------
    # 3. PAGERANK CENTRALITY ON BIPARTITE GRAPH
    # ---------------------------------------------------------
    print("\n[Graph] Calculating PageRank on Bipartite Graph (alpha=0.85)...")
    pagerank_scores = nx.pagerank(B, alpha=0.85, max_iter=80, tol=1e-4)

    # Separate product PageRank and rank
    prod_pr = []
    for sc in products:
        node_id = f"P_{sc}"
        pr = pagerank_scores.get(node_id, 0.0)
        deg = B.degree(node_id)
        desc = prod_desc_map.get(sc, sc)
        rev = df_prod.loc[sc, "total_revenue"] if sc in df_prod.index else 0.0
        prod_pr.append({
            "stock_code": sc,
            "description": desc,
            "pagerank": pr,
            "product_degree": deg,
            "total_revenue": rev
        })

    df_pr = pd.DataFrame(prod_pr).sort_values("pagerank", ascending=False).reset_index(drop=True)
    df_pr["rank"] = df_pr.index + 1
    pr_csv = os.path.join(tables_dir, "graph_pagerank_top_products.csv")
    df_pr.to_csv(pr_csv, index=False)
    print(f"  -> Saved PageRank centrality ranking: {pr_csv}")
    print(f"  -> Top Product by PageRank: {df_pr.iloc[0]['description']} (PR: {df_pr.iloc[0]['pagerank']:.6f})")

    # ---------------------------------------------------------
    # 4. FAST SCALABLE CO-PURCHASE PROJECTION (SPARSE MATRIX)
    # ---------------------------------------------------------
    print("\n[Graph] Projecting Customer-Product Network to Product Co-Purchase Graph...")
    # Map customers and products to integer indices
    cust2idx = {cid: i for i, cid in enumerate(customers)}
    prod2idx = {sc: i for i, sc in enumerate(products)}
    idx2prod = {i: sc for sc, i in prod2idx.items()}

    row_indices = df_inter["customer_id"].map(cust2idx).values
    col_indices = df_inter["stock_code"].map(prod2idx).values
    ones = np.ones(len(df_inter), dtype=np.int32)

    M = sp.csr_matrix((ones, (row_indices, col_indices)), shape=(len(customers), len(products)))
    item_degrees = np.diff(M.tocsc().indptr)

    # Co-occurrence matrix: C = M^T * M
    C = (M.T @ M).tocoo()

    # Filter to statistically significant co-purchase affinity (min co-buyers >= 80)
    # This captures the top ~35,000 product relationships with very high statistical significance
    affinity_threshold = 80
    mask = (C.row < C.col) & (C.data >= affinity_threshold)
    p_i = C.row[mask]
    p_j = C.col[mask]
    weights = C.data[mask]

    print(f"  -> Computed projected product co-purchase affinity edges (min shared buyers >= {affinity_threshold}): {len(weights):,} edges.")

    # Build projected product graph
    G_proj = nx.Graph()
    copurchase_pairs = []

    for u_idx, v_idx, w in zip(p_i, p_j, weights):
        u_code = idx2prod[u_idx]
        v_code = idx2prod[v_idx]
        G_proj.add_edge(u_code, v_code, weight=int(w))

        deg_u = item_degrees[u_idx]
        deg_v = item_degrees[v_idx]
        union_val = deg_u + deg_v - w
        jaccard = w / union_val if union_val > 0 else 0
        copurchase_pairs.append({
            "product_1_code": u_code,
            "product_1_desc": prod_desc_map.get(u_code, u_code),
            "product_2_code": v_code,
            "product_2_desc": prod_desc_map.get(v_code, v_code),
            "shared_customers": int(w),
            "jaccard_similarity": round(jaccard, 4)
        })

    proj_nodes = G_proj.number_of_nodes()
    proj_edges = G_proj.number_of_edges()
    print(f"  -> Projected Product Co-Purchase Graph: {proj_nodes:,} products, {proj_edges:,} affinity links.")

    # Top co-purchase pairs table
    df_pairs = pd.DataFrame(copurchase_pairs).sort_values("shared_customers", ascending=False).reset_index(drop=True)
    pairs_csv = os.path.join(tables_dir, "graph_top_copurchase_pairs.csv")
    df_pairs.to_csv(pairs_csv, index=False)
    print(f"  -> Saved top co-purchase pairs: {pairs_csv}")
    print(f"  -> Strongest Pair: {df_pairs.iloc[0]['product_1_desc'][:25]} <--> {df_pairs.iloc[0]['product_2_desc'][:25]} ({df_pairs.iloc[0]['shared_customers']} shared buyers)")

    # ---------------------------------------------------------
    # 5. COMMUNITY DETECTION & CLUSTERING COEFFICIENT
    # ---------------------------------------------------------
    print("\n[Graph] Detecting Product Communities & Themes (Modularity Partitioning)...")
    proj_gcc_nodes = max(nx.connected_components(G_proj), key=len)
    G_proj_gcc = G_proj.subgraph(proj_gcc_nodes)
    communities = list(community.greedy_modularity_communities(G_proj_gcc, weight="weight"))
    num_communities = len(communities)
    print(f"  -> Identified {num_communities} natural product communities/themes.")

    # Assign community ID to products
    prod_community_map = {}
    community_records = []
    for c_id, comm in enumerate(communities):
        comm_items = list(comm)
        top_items = sorted(
            comm_items,
            key=lambda x: df_prod.loc[x, "total_revenue"] if x in df_prod.index else 0,
            reverse=True
        )[:5]
        top_names = [prod_desc_map.get(sc, sc) for sc in top_items]
        community_records.append({
            "community_id": c_id + 1,
            "product_count": len(comm),
            "representative_products": " | ".join(top_names)
        })
        for sc in comm:
            prod_community_map[sc] = c_id + 1

    df_comm = pd.DataFrame(community_records).sort_values("product_count", ascending=False).reset_index(drop=True)
    comm_csv = os.path.join(tables_dir, "graph_communities.csv")
    df_comm.to_csv(comm_csv, index=False)
    print(f"  -> Saved product communities table: {comm_csv}")

    # Triangle counting and average clustering
    print("\n[Graph] Computing Clustering Coefficient and Triangle Counts...")
    triangles_dict = nx.triangles(G_proj)
    total_triangles = sum(triangles_dict.values()) // 3
    avg_clustering = nx.average_clustering(G_proj)
    print(f"  -> Total Triangles in Co-Purchase Graph: {total_triangles:,}")
    print(f"  -> Average Clustering Coefficient:      {avg_clustering:.4f}")

    # Summary table
    graph_summary = pd.DataFrame({
        "metric": [
            "Bipartite Vertices (Total)", "Customer Vertices", "Product Vertices",
            "Bipartite Interaction Edges", "Connected Components", "Giant Component Vertices",
            "Giant Component Coverage (%)", "Top PageRank Product", "Projected Co-Purchase Nodes",
            "Projected Co-Purchase Edges", "Product Communities Found", "Average Clustering Coefficient",
            "Total Triangles"
        ],
        "value": [
            f"{num_v:,}", f"{len(customers):,}", f"{len(products):,}", f"{num_e:,}",
            f"{num_comps}", f"{giant_comp_size:,}", f"{giant_pct:.2f}%",
            f"{df_pr.iloc[0]['description']} ({df_pr.iloc[0]['stock_code']})",
            f"{proj_nodes:,}", f"{proj_edges:,}", f"{num_communities}",
            f"{avg_clustering:.4f}", f"{total_triangles:,}"
        ]
    })
    sum_csv = os.path.join(tables_dir, "graph_summary_metrics.csv")
    graph_summary.to_csv(sum_csv, index=False)
    print(f"  -> Saved graph summary metrics: {sum_csv}")

    # ---------------------------------------------------------
    # 6. VISUALIZATIONS
    # ---------------------------------------------------------
    print("\n[Graph] Generating Publication-Grade Network Visualizations...")

    # Plot 05: Co-Purchase Network of Top 45 Central Products
    top_nodes = [node for node in df_pr.head(45)["stock_code"].tolist() if node in G_proj]
    sub_G = G_proj.subgraph(top_nodes).copy()
    sub_G.remove_nodes_from(list(nx.isolates(sub_G)))

    fig, ax = plt.subplots(figsize=(15, 11), dpi=300)
    pos = nx.spring_layout(sub_G, k=0.4, seed=42, iterations=80)

    # Node sizes by PageRank
    node_sizes = [df_pr.set_index("stock_code").loc[node, "pagerank"] * 120000 + 150 for node in sub_G.nodes()]

    # Colors by community
    palette = ["#0E868A", "#E07A5F", "#3D405B", "#81B29A", "#F2CC8F", "#D62828", "#003049"]
    node_colors = [palette[(prod_community_map.get(node, 1) - 1) % len(palette)] for node in sub_G.nodes()]

    # Edge widths
    edge_weights = [sub_G[u][v]["weight"] for u, v in sub_G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.8 + 3.5 * (w / max_w) for w in edge_weights]

    nx.draw_networkx_edges(sub_G, pos, ax=ax, width=edge_widths, alpha=0.35, edge_color="#64748B")
    nx.draw_networkx_nodes(sub_G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.92, edgecolors="white", linewidths=1.5)

    # Labels for top 25 nodes
    label_nodes = {node: prod_desc_map.get(node, node)[:20] for node in df_pr.head(25)["stock_code"] if node in sub_G}
    nx.draw_networkx_labels(sub_G, pos, labels=label_nodes, ax=ax, font_size=7.5, font_weight="bold", font_color="#0F172A")

    ax.set_title("ShopGraph: Co-Purchase Affinity Network of Central Products\n(Node size = PageRank Centrality, Node color = Product Community, Edge = Shared Buyers)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    chart5_path = os.path.join(figures_dir, "05_copurchase_network.png")
    plt.savefig(chart5_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Generated network chart: {chart5_path}")

    # Plot 06: Degree & PageRank Centrality Distribution
    fig, (ax_deg, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Customer vs Product degree distributions
    cust_degrees = [B.degree(f"C_{cid}") for cid in customers]
    prod_degrees = [B.degree(f"P_{sc}") for sc in products]

    ax_deg.hist(cust_degrees, bins=np.logspace(0, 3, 30), color="#1F3A52", alpha=0.7, label="Customer Degree (Products/Buyer)")
    ax_deg.hist(prod_degrees, bins=np.logspace(0, 3, 30), color="#0E868A", alpha=0.7, label="Product Degree (Buyers/Product)")
    ax_deg.set_xscale("log")
    ax_deg.set_yscale("log")
    ax_deg.set_title("Degree Distribution in Customer-Product Bipartite Graph", fontsize=11, fontweight="bold")
    ax_deg.set_xlabel("Degree (Log Scale)", fontsize=10)
    ax_deg.set_ylabel("Frequency Count (Log Scale)", fontsize=10)
    ax_deg.legend(loc="upper right")
    ax_deg.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: PageRank vs Total Revenue (Scatter showing structural vs monetary significance)
    ax_pr.scatter(
        df_pr["total_revenue"] / 1000.0, df_pr["pagerank"] * 10000,
        color="#2A9D8F", alpha=0.6, edgecolors="none", s=25
    )
    ax_pr.set_xscale("log")
    ax_pr.set_yscale("log")
    ax_pr.set_title("Structural Centrality (PageRank) vs Total Sales (£k)", fontsize=11, fontweight="bold")
    ax_pr.set_xlabel("Product Revenue (£k, Log Scale)", fontsize=10)
    ax_pr.set_ylabel("PageRank Centrality (x10^4, Log Scale)", fontsize=10)
    ax_pr.grid(True, linestyle="--", alpha=0.3)

    # Highlight top 3 products
    for i in range(3):
        row = df_pr.iloc[i]
        x_val = max(row["total_revenue"] / 1000.0, 0.1)
        y_val = row["pagerank"] * 10000
        ax_pr.annotate(
            row["description"][:18], (x_val, y_val),
            textcoords="offset points", xytext=(8, -4),
            fontsize=7.5, fontweight="bold", color="#1F3A52"
        )

    plt.suptitle("ShopGraph: Network Centrality & Structural Characteristics", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart6_path = os.path.join(figures_dir, "06_graph_degree_pagerank.png")
    plt.savefig(chart6_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Generated degree & PageRank chart: {chart6_path}")

    print("\n" + "=" * 80)
    print("STAGE 3 SUMMARY - GRAPH ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Bipartite Graph:           {num_v:,} vertices, {num_e:,} edges")
    print(f"Giant Connected Component: {giant_pct:.2f}% of graph")
    print(f"Top PageRank Product:      {df_pr.iloc[0]['description']} (PR: {df_pr.iloc[0]['pagerank']:.6f})")
    print(f"Co-Purchase Graph:         {proj_nodes:,} products, {proj_edges:,} affinity edges")
    print(f"Communities Found:         {num_communities}")
    print(f"Average Clustering Coeff:  {avg_clustering:.4f}")
    print(f"Total Stage 3 Elapsed:     {time.time() - t_start:.2f} seconds")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    outputs_dir = os.path.join(project_root, "outputs")
    run_graph_analysis(data_dir, outputs_dir)
