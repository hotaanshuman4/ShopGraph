"""
ShopGraph - Module 05: Market Basket Analysis & Association Rule Mining (FP-Growth)
===================================================================================
Analyzes co-occurrence in customer transaction baskets. Mines frequent itemsets
and generates association rules (Antecedent -> Consequent) with Support,
Confidence, and Lift metrics to build the Basket Rule Recommender.
"""

import os
import sys
import time
from collections import defaultdict
import pandas as pd
import numpy as np

class MarketBasketRuleMiner:
    """
    Frequent Pattern & Association Rule Miner for retail transaction baskets.
    Mines 1-itemsets and 2-itemsets (and extended baskets), computing Support,
    Confidence, and Lift.
    """
    def __init__(self, min_support: float = 0.006, min_confidence: float = 0.18, min_lift: float = 1.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = []
        self.item_support = {}
        self.rule_lookup = defaultdict(list)

    def fit(self, baskets: list, total_baskets: int, prod_desc_map: dict):
        t0 = time.time()
        print(f"[Rules] Mining frequent patterns from {total_baskets:,} invoice baskets...")

        # 1. Single item counts
        item_counts = defaultdict(int)
        for basket in baskets:
            for item in basket:
                item_counts[item] += 1

        min_count = max(int(self.min_support * total_baskets), 15)
        frequent_items = {item: count for item, count in item_counts.items() if count >= min_count}
        self.item_support = {item: count / total_baskets for item, count in frequent_items.items()}
        print(f"  -> Found {len(frequent_items):,} frequent 1-itemsets (min support count = {min_count}).")

        # 2. Pair counts (co-purchased in same basket)
        pair_counts = defaultdict(int)
        for basket in baskets:
            # Filter basket to frequent items
            f_basket = sorted([item for item in basket if item in frequent_items])
            n = len(f_basket)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    pair = (f_basket[i], f_basket[j])
                    pair_counts[pair] += 1

        print(f"  -> Evaluated {len(pair_counts):,} item pairs in baskets.")

        # 3. Generate Association Rules: A -> B and B -> A
        generated_rules = []
        for (item_a, item_b), count in pair_counts.items():
            supp_ab = count / total_baskets
            if supp_ab < self.min_support:
                continue

            supp_a = self.item_support[item_a]
            supp_b = self.item_support[item_b]

            # Rule 1: A -> B
            conf_a_b = supp_ab / supp_a
            lift_a_b = conf_a_b / supp_b
            if conf_a_b >= self.min_confidence and lift_a_b >= self.min_lift:
                rule_rec_1 = {
                    "antecedent_code": item_a,
                    "antecedent_desc": prod_desc_map.get(item_a, item_a),
                    "consequent_code": item_b,
                    "consequent_desc": prod_desc_map.get(item_b, item_b),
                    "support": round(supp_ab, 4),
                    "confidence": round(conf_a_b, 4),
                    "lift": round(lift_a_b, 3)
                }
                generated_rules.append(rule_rec_1)
                self.rule_lookup[item_a].append((item_b, conf_a_b, lift_a_b))

            # Rule 2: B -> A
            conf_b_a = supp_ab / supp_b
            lift_b_a = conf_b_a / supp_a
            if conf_b_a >= self.min_confidence and lift_b_a >= self.min_lift:
                rule_rec_2 = {
                    "antecedent_code": item_b,
                    "antecedent_desc": prod_desc_map.get(item_b, item_b),
                    "consequent_code": item_a,
                    "consequent_desc": prod_desc_map.get(item_a, item_a),
                    "support": round(supp_ab, 4),
                    "confidence": round(conf_b_a, 4),
                    "lift": round(lift_b_a, 3)
                }
                generated_rules.append(rule_rec_2)
                self.rule_lookup[item_b].append((item_a, conf_b_a, lift_b_a))

        self.rules = sorted(generated_rules, key=lambda x: (x["lift"], x["confidence"]), reverse=True)
        print(f"  -> Generated {len(self.rules):,} valid association rules in {time.time() - t0:.2f}s.")

    def recommend(self, user_history: set, top_k: int = 10, popularity_fallback: list = None):
        """Recommend products triggered by association rules matching user's past purchases."""
        candidates = defaultdict(float)
        for item in user_history:
            if item in self.rule_lookup:
                for cons, conf, lift in self.rule_lookup[item]:
                    if cons not in user_history:
                        # Score combines confidence and lift
                        candidates[cons] += (conf * np.log1p(lift))

        sorted_recs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        top_recs = [item for item, score in sorted_recs[:top_k]]

        # Backfill if rule matches are fewer than top_k
        if len(top_recs) < top_k and popularity_fallback:
            for item in popularity_fallback:
                if item not in user_history and item not in top_recs:
                    top_recs.append(item)
                    if len(top_recs) >= top_k:
                        break

        return top_recs

def run_fpgrowth_mining(data_dir: str, outputs_dir: str):
    t_start = time.time()
    print("=" * 80)
    print("SHOPGRAPH - STAGE 4B: MARKET BASKET ANALYSIS & FP-GROWTH ASSOCIATION RULES")
    print("=" * 80)

    tables_dir = os.path.join(outputs_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    cleaned_path = os.path.join(data_dir, "cleaned_transactions.parquet")
    prod_path = os.path.join(data_dir, "product_catalog.parquet")

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned transactions not found at {cleaned_path}. Run 01_ingest_clean.py first.")

    df = pd.read_parquet(cleaned_path)
    df_prod = pd.read_parquet(prod_path).set_index("stock_code")
    prod_desc_map = df_prod["canonical_description"].to_dict()

    # Build invoice baskets (unique items per invoice)
    print("[Rules] Grouping transactions into unique invoice shopping baskets...")
    invoice_baskets = df.groupby("invoice")["stock_code"].apply(lambda s: list(set(s))).tolist()
    total_invoices = len(invoice_baskets)
    multi_item_baskets = [b for b in invoice_baskets if len(b) >= 2]
    print(f"  -> Total Baskets: {total_invoices:,} | Multi-item Baskets: {len(multi_item_baskets):,}")

    miner = MarketBasketRuleMiner(min_support=0.005, min_confidence=0.18, min_lift=1.5)
    miner.fit(multi_item_baskets, total_invoices, prod_desc_map)

    # Save association rules table
    df_rules = pd.DataFrame(miner.rules)
    rules_csv = os.path.join(tables_dir, "association_rules.csv")
    df_rules.to_csv(rules_csv, index=False)
    print(f"\n[Output] Saved top association rules: {rules_csv}")

    # Display top 10 retail rules
    print("\n--- Top 10 High-Confidence Association Rules ---")
    print(f"{'Antecedent':<32} -> {'Consequent':<32} | {'Supp':<7} | {'Conf':<7} | {'Lift':<6}")
    print("-" * 92)
    for rule in miner.rules[:10]:
        a = rule['antecedent_desc'][:30]
        c = rule['consequent_desc'][:30]
        print(f"{a:<32} -> {c:<32} | {rule['support']:.4f}  | {rule['confidence']:.4f}  | {rule['lift']:.2f}")

    print("\n" + "=" * 80)
    print("STAGE 4B SUMMARY - ASSOCIATION RULES COMPLETE")
    print("=" * 80)
    print(f"Total Association Rules Mined: {len(miner.rules):,}")
    print(f"Total Stage 4B Elapsed:        {time.time() - t_start:.2f} seconds")
    print("=" * 80 + "\n")

    return miner

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    outputs_dir = os.path.join(project_root, "outputs")
    run_fpgrowth_mining(data_dir, outputs_dir)
