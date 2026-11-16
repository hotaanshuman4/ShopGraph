"""
ShopGraph - Module 04: Alternating Least Squares (ALS) Implicit Feedback Recommender
===================================================================================
Implements the distributed implicit feedback collaborative filtering model
(Hu, Koren, Volinsky). Factorizes customer-product interactions into latent
preference vectors, generates personalized Top-10 recommendations, and
produces comparative recommendation profiles for representative customers.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import scipy.sparse as sp

class ImplicitALSRecommender:
    """
    Alternating Least Squares for Implicit Feedback Datasets.
    Minimizes: sum_{u,i} c_{ui} * (p_{ui} - x_u^T y_i)^2 + lambda * (||x_u||^2 + ||y_i||^2)
    where c_{ui} = 1 + alpha * r_{ui}, p_{ui} = 1(r_{ui} > 0).
    """
    def __init__(self, factors: int = 25, regularization: float = 0.08, alpha: float = 30.0, iterations: int = 12, seed: int = 42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors = None
        self.item_factors = None
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}

    def fit(self, df_train: pd.DataFrame, verbose: bool = True):
        t0 = time.time()
        np.random.seed(self.seed)

        users = df_train["customer_id"].unique()
        items = df_train["stock_code"].unique()
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = {i: it for it, i in self.item2idx.items()}

        n_users = len(users)
        n_items = len(items)

        # Build sparse confidence matrix C - I (where non-zero entries are alpha * r_ui)
        u_idx = df_train["customer_id"].map(self.user2idx).values
        i_idx = df_train["stock_code"].map(self.item2idx).values
        weights = df_train["total_quantity"].values

        # Confidence: c_ui = 1 + alpha * r_ui => (c_ui - 1) = alpha * r_ui
        c_minus_1 = self.alpha * np.log1p(weights)

        # Sparse interaction matrices
        C_u = sp.csr_matrix((c_minus_1, (u_idx, i_idx)), shape=(n_users, n_items))
        C_i = sp.csr_matrix((c_minus_1, (i_idx, u_idx)), shape=(n_items, n_users))

        # Initialize latent factor matrices with small random values
        self.user_factors = np.random.normal(0, 0.05, (n_users, self.factors)).astype(np.float64)
        self.item_factors = np.random.normal(0, 0.05, (n_items, self.factors)).astype(np.float64)

        reg_I = self.regularization * np.eye(self.factors, dtype=np.float64)

        if verbose:
            print(f"[ALS] Fitting model: {n_users:,} users x {n_items:,} items | Factors={self.factors}, Reg={self.regularization}, Alpha={self.alpha}, Iterations={self.iterations}")

        # ALS Optimization loop
        for it in range(1, self.iterations + 1):
            t_iter = time.time()

            # 1. Update User Factors
            YtY = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                start = C_u.indptr[u]
                end = C_u.indptr[u + 1]
                if start == end:
                    # User has no interactions (unlikely in filtered train set)
                    continue

                u_items = C_u.indices[start:end]
                u_c_minus_1 = C_u.data[start:end]
                Y_u = self.item_factors[u_items, :]

                # A_u = YtY + Y_u.T @ diag(c_minus_1) @ Y_u + lambda * I
                # Optimized vectorized diagonal multiplication:
                Y_weighted = Y_u * u_c_minus_1[:, np.newaxis]
                A_u = YtY + (Y_u.T @ Y_weighted) + reg_I

                # b_u = Y_u.T @ c_u * p_u. Since p_u = 1 for purchased items:
                # c_u = 1 + c_minus_1
                b_u = Y_u.T @ (1.0 + u_c_minus_1)

                self.user_factors[u, :] = np.linalg.solve(A_u, b_u)

            # 2. Update Item Factors
            XtX = self.user_factors.T @ self.user_factors
            for i in range(n_items):
                start = C_i.indptr[i]
                end = C_i.indptr[i + 1]
                if start == end:
                    continue

                i_users = C_i.indices[start:end]
                i_c_minus_1 = C_i.data[start:end]
                X_i = self.user_factors[i_users, :]

                X_weighted = X_i * i_c_minus_1[:, np.newaxis]
                A_i = XtX + (X_i.T @ X_weighted) + reg_I
                b_i = X_i.T @ (1.0 + i_c_minus_1)

                self.item_factors[i, :] = np.linalg.solve(A_i, b_i)

            if verbose:
                print(f"  Iteration {it:2d}/{self.iterations} completed in {time.time() - t_iter:.2f}s")

        fit_time = time.time() - t0
        if verbose:
            print(f"[ALS] Training successfully finished in {fit_time:.2f}s.")
        return fit_time

    def recommend(self, customer_id: int, train_history: set, top_k: int = 10):
        """Recommend top_k unpurchased products for a given customer."""
        if customer_id not in self.user2idx:
            return []

        u_idx = self.user2idx[customer_id]
        u_factor = self.user_factors[u_idx, :]

        # Score all items: score_i = u_factor @ item_factor_i
        scores = self.item_factors @ u_factor

        # Mask items already purchased in training history
        for item in train_history:
            if item in self.item2idx:
                scores[self.item2idx[item]] = -np.inf

        # Get top K indices
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        recommendations = []
        for idx in top_indices:
            if scores[idx] > -np.inf:
                recommendations.append((self.idx2item[idx], float(scores[idx])))
        return recommendations

def run_als_recommender(data_dir: str, outputs_dir: str):
    t_start = time.time()
    print("=" * 80)
    print("SHOPGRAPH - STAGE 4A: PERSONALIZED COLLABORATIVE FILTERING (ALS)")
    print("=" * 80)

    tables_dir = os.path.join(outputs_dir, "tables")
    figures_dir = os.path.join(outputs_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train_interactions.parquet")
    prod_path = os.path.join(data_dir, "product_catalog.parquet")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train interactions not found at {train_path}. Run 01_ingest_clean.py first.")

    df_train = pd.read_parquet(train_path)
    df_prod = pd.read_parquet(prod_path).set_index("stock_code")
    prod_desc_map = df_prod["canonical_description"].to_dict()

    # Instantiate and fit ALS model
    model = ImplicitALSRecommender(factors=25, regularization=0.08, alpha=30.0, iterations=12)
    train_time = model.fit(df_train, verbose=True)

    # Precompute training histories per customer
    train_history_map = df_train.groupby("customer_id")["stock_code"].apply(set).to_dict()

    # ---------------------------------------------------------
    # GENERATE SAMPLE CUSTOMER PROFILES (Report Result 10.3)
    # ---------------------------------------------------------
    print("\n[ALS] Generating Top-10 Recommendations for Representative Customer Personas...")

    # Select 3 representative customers:
    # 1. High-frequency wholesale buyer
    # 2. Consistent gift / home decor buyer
    # 3. Moderate seasonal buyer
    cust_order_counts = df_train.groupby("customer_id")["stock_code"].count().sort_values(ascending=False)
    # Pick top 20th percentile, median, and frequent
    sample_cids = [
        cust_order_counts.index[5],   # High volume
        cust_order_counts.index[120], # Medium volume
        cust_order_counts.index[450]  # Moderate volume
    ]

    sample_records = []
    print("\n" + "=" * 80)
    print("SAMPLE CUSTOMER RECOMMENDATIONS COMPARISON (RESULT 10.3)")
    print("=" * 80)

    for cid in sample_cids:
        history = list(train_history_map.get(cid, set()))
        history_names = [f"{prod_desc_map.get(sc, sc)} ({sc})" for sc in history[:5]]
        recs = model.recommend(cid, set(history), top_k=10)

        print(f"\n[Customer ID: {cid}]")
        print(f"  Previous Purchases Sample ({len(history)} total items bought):")
        for h in history_names:
            print(f"    - {h}")

        print("  Personalized Top-10 ALS Recommendations:")
        for rank, (sc, score) in enumerate(recs, 1):
            desc = prod_desc_map.get(sc, sc)
            print(f"    {rank:2d}. {desc:<45} [{sc}] (score: {score:.3f})")
            sample_records.append({
                "customer_id": cid,
                "customer_type": "High Volume" if cid == sample_cids[0] else ("Medium Volume" if cid == sample_cids[1] else "Moderate Volume"),
                "rank": rank,
                "recommended_stock_code": sc,
                "recommended_description": desc,
                "confidence_score": round(score, 3),
                "history_sample": " | ".join(history_names[:3])
            })

    df_samples = pd.DataFrame(sample_records)
    samples_csv = os.path.join(tables_dir, "sample_recommendations.csv")
    df_samples.to_csv(samples_csv, index=False)
    print(f"\n[Output] Saved sample customer recommendations: {samples_csv}")

    return model, train_time

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    outputs_dir = os.path.join(project_root, "outputs")
    run_als_recommender(data_dir, outputs_dir)
