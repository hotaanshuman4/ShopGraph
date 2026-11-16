"""
ShopGraph - Module 01: Ingestion, Cleaning & Interaction Signal Engineering
==========================================================================
Part of the ShopGraph Distributed E-Commerce Analytics & Recommendation System.
Ingests the UCI Online Retail II dataset, applies 7 rigorous cleaning and
validation rules, builds derived customer and product tables, computes implicit
interaction confidence weights, and produces a temporal train/test split.
"""

import os
import sys
import time
import openpyxl
import pandas as pd
import numpy as np

def run_ingest_and_clean(data_dir: str, outputs_dir: str):
    t_start = time.time()
    print("=" * 80)
    print("SHOPGRAPH - STAGE 1: DATA INGESTION, CLEANING & INTERACTION ENGINEERING")
    print("=" * 80)

    tables_dir = os.path.join(outputs_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    raw_parquet_path = os.path.join(data_dir, "online_retail_raw.parquet")
    xlsx_path = os.path.join(data_dir, "online_retail_II.xlsx")

    # ---------------------------------------------------------
    # 1. INGESTION
    # ---------------------------------------------------------
    if os.path.exists(raw_parquet_path):
        print(f"[Ingestion] Found existing pre-converted raw parquet: {raw_parquet_path}")
        print("[Ingestion] Loading raw parquet file...")
        df_raw = pd.read_parquet(raw_parquet_path)
    elif os.path.exists(xlsx_path):
        print(f"[Ingestion] Found raw Excel dataset: {xlsx_path}")
        print("[Ingestion] Streaming sheets 'Year 2009-2010' and 'Year 2010-2011' into parquet...")
        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        all_rows = []
        headers = None

        for sheet_name in wb.sheetnames:
            print(f"  -> Reading sheet: {sheet_name}...")
            ws = wb[sheet_name]
            first = True
            for row in ws.iter_rows(values_only=True):
                if first:
                    if headers is None:
                        headers = [str(c).strip() for c in row]
                    first = False
                    continue
                if row[0] is not None:
                    all_rows.append(row)
        wb.close()

        print(f"[Ingestion] Extracted {len(all_rows):,} rows across all sheets.")
        df_raw = pd.DataFrame(all_rows, columns=headers)
        # Normalize column names
        col_map = {
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country"
        }
        df_raw.rename(columns=col_map, inplace=True)
        # Type conversions
        df_raw["quantity"] = pd.to_numeric(df_raw["quantity"], errors="coerce")
        df_raw["price"] = pd.to_numeric(df_raw["price"], errors="coerce")
        df_raw["customer_id"] = pd.to_numeric(df_raw["customer_id"], errors="coerce")
        df_raw["invoice_date"] = pd.to_datetime(df_raw["invoice_date"], errors="coerce")
        df_raw["invoice"] = df_raw["invoice"].astype(str).str.strip()
        df_raw["stock_code"] = df_raw["stock_code"].astype(str).str.strip()
        df_raw["description"] = df_raw["description"].astype(str).str.strip()
        df_raw["country"] = df_raw["country"].astype(str).str.strip()

        print(f"[Ingestion] Saving cached raw parquet to: {raw_parquet_path}...")
        df_raw.to_parquet(raw_parquet_path, index=False)
    else:
        print("[Ingestion] No Online Retail II raw file found! Calling backup sample generator...")
        try:
            from src.sample_generator import generate_sample_retail_data
        except ImportError:
            from sample_generator import generate_sample_retail_data
        sample_path = os.path.join(data_dir, "sample_retail_data.parquet")
        df_raw = generate_sample_retail_data(sample_path, n_records=75000)
        col_map = {
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country"
        }
        df_raw.rename(columns=col_map, inplace=True)

    initial_rows = len(df_raw)
    print(f"\n[Cleaning] Starting Data Validation & Cleaning Pipeline on {initial_rows:,} records.")

    # ---------------------------------------------------------
    # 2. SEVEN CLEANING & VALIDATION RULES (Table 7.3)
    # ---------------------------------------------------------
    cleaning_log = []
    current_df = df_raw.copy()

    def log_step(rule_num, rule_name, reason, before_count, after_count):
        dropped = before_count - after_count
        pct = (dropped / before_count * 100) if before_count > 0 else 0
        cumulative_pct = ((initial_rows - after_count) / initial_rows * 100)
        cleaning_log.append({
            "step": rule_num,
            "rule": rule_name,
            "reason": reason,
            "rows_before": before_count,
            "rows_after": after_count,
            "rows_dropped": dropped,
            "step_drop_pct": round(pct, 2),
            "cumulative_drop_pct": round(cumulative_pct, 2)
        })
        print(f"  Step {rule_num} [{rule_name}]: {before_count:,} -> {after_count:,} "
              f"(-{dropped:,} | -{pct:.2f}%)")

    # Step 0: Raw
    cleaning_log.append({
        "step": 0,
        "rule": "Raw Ingestion",
        "reason": "Original unfiltered invoice rows",
        "rows_before": initial_rows,
        "rows_after": initial_rows,
        "rows_dropped": 0,
        "step_drop_pct": 0.0,
        "cumulative_drop_pct": 0.0
    })

    # Rule 1: Drop rows with null Customer ID
    b_count = len(current_df)
    current_df = current_df[current_df["customer_id"].notna() & (current_df["customer_id"] > 0)]
    current_df["customer_id"] = current_df["customer_id"].astype(np.int64)
    log_step(1, "Drop Null Customer ID", "Interactions must belong to an identifiable customer", b_count, len(current_df))

    # Rule 2: Remove cancelled invoices ('C' prefix) and Quantity <= 0
    b_count = len(current_df)
    is_cancelled = current_df["invoice"].str.upper().str.startswith("C") | (current_df["quantity"] <= 0)
    current_df = current_df[~is_cancelled]
    log_step(2, "Remove Cancellations & Non-positive Qty", "Cancellations and returns are not purchases", b_count, len(current_df))

    # Rule 3: Remove lines with Price <= 0
    b_count = len(current_df)
    current_df = current_df[current_df["price"] > 0]
    log_step(3, "Remove Price <= 0", "Free or erroneous lines carry no buying intent", b_count, len(current_df))

    # Rule 4: Exclude non-product service stock codes
    b_count = len(current_df)
    non_product_codes = {
        "POST", "DOT", "M", "BANK CHARGES", "D", "CRUK", "PADS", "AMAZONFEE",
        "S", "TEST", "C2", "GIFT_0001_10", "GIFT_0001_20", "GIFT_0001_30",
        "GIFT_0001_40", "GIFT_0001_50", "ADJUST", "ADJUST2"
    }
    # Check upper code against non-product set or manual pattern
    code_upper = current_df["stock_code"].str.upper().str.strip()
    is_non_prod = code_upper.isin(non_product_codes) | code_upper.str.startswith("POST")
    current_df = current_df[~is_non_prod]
    log_step(4, "Exclude Non-Product Codes", "Postage, fees and manual adjustments are not products", b_count, len(current_df))

    # Rule 5: Drop exact duplicate lines
    b_count = len(current_df)
    current_df = current_df.drop_duplicates(subset=["invoice", "stock_code", "quantity", "price", "customer_id", "invoice_date"])
    log_step(5, "Drop Exact Duplicate Lines", "Prevent double-counting of identical transaction lines", b_count, len(current_df))

    # Rule 6: Cap extreme quantities at a high percentile
    b_count = len(current_df)
    p999 = current_df["quantity"].quantile(0.999)
    current_df["raw_quantity"] = current_df["quantity"]
    current_df["quantity"] = current_df["quantity"].clip(upper=p999)
    log_step(6, f"Cap Extreme Quantities (P99.9 = {int(p999)})", "Limit pull of extreme wholesale outliers on confidence weights", b_count, len(current_df))

    # Rule 7: Map each stock code to one canonical description
    b_count = len(current_df)
    valid_desc = current_df[current_df["description"].str.len() > 2]
    desc_mode = valid_desc.groupby("stock_code")["description"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    ).to_dict()
    current_df["canonical_description"] = current_df["stock_code"].map(desc_mode)
    # Fill any missing canonical descriptions with stock_code
    current_df["canonical_description"] = current_df["canonical_description"].fillna(current_df["stock_code"])
    log_step(7, "Standardize Canonical Descriptions", "Resolve inconsistencies in free-text product names", b_count, len(current_df))

    # Add line total revenue
    current_df["line_total"] = current_df["quantity"] * current_df["price"]

    # ---------------------------------------------------------
    # 3. SAVE CLEANED TRANSACTIONS & LOG
    # ---------------------------------------------------------
    cleaned_parquet_path = os.path.join(data_dir, "cleaned_transactions.parquet")
    current_df.to_parquet(cleaned_parquet_path, index=False)
    print(f"\n[Output] Cleaned transactions saved to: {cleaned_parquet_path}")

    log_df = pd.DataFrame(cleaning_log)
    log_csv_path = os.path.join(tables_dir, "cleaning_log.csv")
    log_df.to_csv(log_csv_path, index=False)
    print(f"[Output] Cleaning log saved to: {log_csv_path}")

    # ---------------------------------------------------------
    # 4. DERIVED TABLES (Customers, Products, Interactions)
    # ---------------------------------------------------------
    print("\n[Derived Tables] Computing Customers, Products, and Interaction Signals...")
    max_date = current_df["invoice_date"].max()

    # Customer RFM Base Table
    customer_df = current_df.groupby("customer_id").agg(
        order_count=("invoice", "nunique"),
        total_items=("quantity", "sum"),
        total_spend=("line_total", "sum"),
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        country=("country", "first")
    ).reset_index()
    customer_df["recency_days"] = (max_date - customer_df["last_purchase"]).dt.total_seconds() / (24 * 3600)
    customer_df["recency_days"] = customer_df["recency_days"].round(1)
    customer_df["total_spend"] = customer_df["total_spend"].round(2)
    cust_parquet_path = os.path.join(data_dir, "customer_rfm.parquet")
    customer_df.to_parquet(cust_parquet_path, index=False)

    # Product Catalog Table
    product_df = current_df.groupby("stock_code").agg(
        canonical_description=("canonical_description", "first"),
        total_units_sold=("quantity", "sum"),
        total_revenue=("line_total", "sum"),
        distinct_customers=("customer_id", "nunique"),
        distinct_orders=("invoice", "nunique"),
        avg_unit_price=("price", "mean")
    ).reset_index()
    product_df["total_revenue"] = product_df["total_revenue"].round(2)
    product_df["avg_unit_price"] = product_df["avg_unit_price"].round(2)
    prod_parquet_path = os.path.join(data_dir, "product_catalog.parquet")
    product_df.to_parquet(prod_parquet_path, index=False)

    # Interaction Table (Customer x Product)
    # Implicit feedback confidence weight: c_ui = 1 + alpha * log(1 + quantity)
    alpha = 40.0
    interactions_df = current_df.groupby(["customer_id", "stock_code"]).agg(
        total_quantity=("quantity", "sum"),
        invoice_count=("invoice", "nunique"),
        last_purchase=("invoice_date", "max"),
        first_purchase=("invoice_date", "min")
    ).reset_index()
    interactions_df["confidence_weight"] = 1.0 + alpha * np.log1p(interactions_df["total_quantity"])
    interactions_df["confidence_weight"] = interactions_df["confidence_weight"].round(3)
    inter_parquet_path = os.path.join(data_dir, "interactions.parquet")
    interactions_df.to_parquet(inter_parquet_path, index=False)

    # ---------------------------------------------------------
    # 5. TEMPORAL TRAIN / TEST SPLIT (Section 7.4)
    # ---------------------------------------------------------
    # 80/20 chronological split based on invoice timestamp
    date_cutoff = current_df["invoice_date"].quantile(0.80)
    print(f"\n[Train/Test Split] Temporal cutoff date: {date_cutoff.strftime('%Y-%m-%d %H:%M:%S')}")

    train_tx = current_df[current_df["invoice_date"] < date_cutoff]
    test_tx = current_df[current_df["invoice_date"] >= date_cutoff]

    train_inter = train_tx.groupby(["customer_id", "stock_code"]).agg(
        total_quantity=("quantity", "sum"),
        invoice_count=("invoice", "nunique"),
        last_purchase=("invoice_date", "max")
    ).reset_index()
    train_inter["confidence_weight"] = 1.0 + alpha * np.log1p(train_inter["total_quantity"])

    test_inter = test_tx.groupby(["customer_id", "stock_code"]).agg(
        total_quantity=("quantity", "sum"),
        invoice_count=("invoice", "nunique"),
        last_purchase=("invoice_date", "max")
    ).reset_index()
    test_inter["confidence_weight"] = 1.0 + alpha * np.log1p(test_inter["total_quantity"])

    train_path = os.path.join(data_dir, "train_interactions.parquet")
    test_path = os.path.join(data_dir, "test_interactions.parquet")
    train_inter.to_parquet(train_path, index=False)
    test_inter.to_parquet(test_path, index=False)

    # Metrics summary
    num_customers = len(customer_df)
    num_products = len(product_df)
    total_interactions = len(interactions_df)
    matrix_cells = num_customers * num_products
    density = (total_interactions / matrix_cells * 100) if matrix_cells > 0 else 0

    print("\n" + "=" * 80)
    print("STAGE 1 SUMMARY - DATASET COUNTS & MATRIX DENSITY")
    print("=" * 80)
    print(f"Raw Input Rows:            {initial_rows:,}")
    print(f"Cleaned Transactions:      {len(current_df):,} (kept {len(current_df)/initial_rows*100:.2f}%)")
    print(f"Distinct Customers:        {num_customers:,}")
    print(f"Distinct Products:         {num_products:,}")
    print(f"Customer-Product Pairs:    {total_interactions:,}")
    print(f"Interaction Matrix Size:   {num_customers:,} x {num_products:,} ({matrix_cells:,} potential cells)")
    print(f"Matrix Sparsity / Density: {density:.4f}% density ({100 - density:.4f}% sparse)")
    print(f"Train Interactions:        {len(train_inter):,} (from {len(train_tx):,} transactions)")
    print(f"Test Interactions:         {len(test_inter):,} (from {len(test_tx):,} transactions)")
    print(f"Total Stage 1 Elapsed:     {time.time() - t_start:.2f} seconds")
    print("=" * 80 + "\n")

    return {
        "initial_rows": initial_rows,
        "cleaned_rows": len(current_df),
        "num_customers": num_customers,
        "num_products": num_products,
        "interactions": total_interactions,
        "density_pct": density,
        "train_interactions": len(train_inter),
        "test_interactions": len(test_inter)
    }

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    outputs_dir = os.path.join(project_root, "outputs")
    run_ingest_and_clean(data_dir, outputs_dir)
