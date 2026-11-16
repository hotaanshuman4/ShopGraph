"""
ShopGraph - Module 02: Exploratory Data Analysis & Distributed Analytics
========================================================================
Analyzes revenue, order seasonality, country distribution, basket statistics,
and customer RFM segmentation. Produces executive summary tables and
publication-grade visualization charts.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure Matplotlib cache directory to avoid permissions issues
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "mpl_cache")

def run_eda(data_dir: str, outputs_dir: str):
    t_start = time.time()
    print("=" * 80)
    print("SHOPGRAPH - STAGE 2: CUSTOMER & SALES EXPLORATORY ANALYTICS")
    print("=" * 80)

    tables_dir = os.path.join(outputs_dir, "tables")
    figures_dir = os.path.join(outputs_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    cleaned_parquet = os.path.join(data_dir, "cleaned_transactions.parquet")
    if not os.path.exists(cleaned_parquet):
        raise FileNotFoundError(f"Cleaned transactions not found at {cleaned_parquet}. Run 01_ingest_clean.py first.")

    print(f"[EDA] Loading cleaned transaction records from: {cleaned_parquet}...")
    df = pd.read_parquet(cleaned_parquet)
    print(f"[EDA] Loaded {len(df):,} transactions for analytics.")

    # ---------------------------------------------------------
    # 1. MONTHLY REVENUE & SEASONALITY
    # ---------------------------------------------------------
    print("\n[EDA] Computing monthly revenue and order trends...")
    df["year_month"] = df["invoice_date"].dt.to_period("M").astype(str)

    monthly = df.groupby("year_month").agg(
        total_revenue=("line_total", "sum"),
        total_orders=("invoice", "nunique"),
        unique_customers=("customer_id", "nunique"),
        total_items_sold=("quantity", "sum"),
    ).reset_index()

    monthly["total_revenue"] = monthly["total_revenue"].round(2)
    monthly["avg_order_value"] = (monthly["total_revenue"] / monthly["total_orders"]).round(2)
    monthly_csv = os.path.join(tables_dir, "monthly_sales.csv")
    monthly.to_csv(monthly_csv, index=False)
    print(f"  -> Saved monthly sales table: {monthly_csv}")

    # Plot 01: Monthly Sales Trend (Combo Chart)
    fig, ax1 = plt.subplots(figsize=(13, 6), dpi=300)
    navy_color = "#1F3A52"
    teal_color = "#0E868A"

    x_indices = np.arange(len(monthly))
    bars = ax1.bar(
        x_indices, monthly["total_revenue"] / 1000.0,
        color=navy_color, width=0.6, alpha=0.9, label="Monthly Revenue (£k)"
    )
    ax1.set_xlabel("Year-Month", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Revenue (£ Thousands)", fontsize=11, fontweight="bold", color=navy_color)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("£%.0fk"))

    # Line overlay for order count
    ax2 = ax1.twinx()
    line = ax2.plot(
        x_indices, monthly["total_orders"],
        color=teal_color, marker="o", linewidth=2.5, markersize=6, label="Total Orders"
    )
    ax2.set_ylabel("Number of Orders", fontsize=11, fontweight="bold", color=teal_color)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

    # Title & styling
    plt.title("ShopGraph: Monthly Revenue & Order Volume Trends (2009–2011)", fontsize=14, fontweight="bold", pad=15)
    lines_labels = [bars, line[0]]
    labels = [l.get_label() for l in lines_labels]
    ax1.legend(lines_labels, labels, loc="upper left", frameon=True, framealpha=0.9)
    plt.tight_layout()
    chart1_path = os.path.join(figures_dir, "01_monthly_sales_trend.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"  -> Generated chart: {chart1_path}")

    # ---------------------------------------------------------
    # 2. TOP PRODUCTS (REVENUE & QUANTITY)
    # ---------------------------------------------------------
    print("\n[EDA] Analyzing top-selling and highest-revenue products...")
    top_revenue = df.groupby(["stock_code", "canonical_description"]).agg(
        total_revenue=("line_total", "sum"),
        total_quantity=("quantity", "sum"),
        order_count=("invoice", "nunique"),
        customer_count=("customer_id", "nunique"),
        avg_price=("price", "mean")
    ).reset_index().sort_values("total_revenue", ascending=False)

    top_revenue["total_revenue"] = top_revenue["total_revenue"].round(2)
    top_revenue["avg_price"] = top_revenue["avg_price"].round(2)
    top_revenue_csv = os.path.join(tables_dir, "top_products_revenue.csv")
    top_revenue.to_csv(top_revenue_csv, index=False)
    print(f"  -> Saved top products table: {top_revenue_csv}")

    # Plot 02: Top 15 Products by Revenue (Horizontal Bar Chart)
    top15 = top_revenue.head(15).iloc[::-1]  # Reverse for bottom-to-top display
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    y_pos = np.arange(len(top15))

    short_labels = [
        f"{row['canonical_description'][:32]}... ({row['stock_code']})"
        if len(row['canonical_description']) > 32
        else f"{row['canonical_description']} ({row['stock_code']})"
        for _, row in top15.iterrows()
    ]

    bars = ax.barh(y_pos, top15["total_revenue"] / 1000.0, color="#176B87", alpha=0.9, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Total Sales (£ Thousands)", fontsize=11, fontweight="bold")
    ax.set_title("ShopGraph: Top 15 Highest Revenue Products", fontsize=14, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("£%.1fk"))
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"£{w:.1f}k",
                va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1F3A52")

    plt.tight_layout()
    chart2_path = os.path.join(figures_dir, "02_top_products_revenue.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"  -> Generated chart: {chart2_path}")

    # ---------------------------------------------------------
    # 3. SALES BY COUNTRY
    # ---------------------------------------------------------
    print("\n[EDA] Summarizing geographic sales distribution...")
    country_sales = df.groupby("country").agg(
        total_revenue=("line_total", "sum"),
        total_orders=("invoice", "nunique"),
        unique_customers=("customer_id", "nunique"),
        total_units=("quantity", "sum")
    ).reset_index().sort_values("total_revenue", ascending=False)

    total_store_revenue = country_sales["total_revenue"].sum()
    country_sales["revenue_share_pct"] = (country_sales["total_revenue"] / total_store_revenue * 100).round(2)
    country_sales["total_revenue"] = country_sales["total_revenue"].round(2)
    country_csv = os.path.join(tables_dir, "country_sales.csv")
    country_sales.to_csv(country_csv, index=False)
    print(f"  -> Saved country sales table: {country_csv}")

    # Plot 03: Country Sales Distribution (Top 10 International + UK Share)
    fig, (ax_main, ax_intl) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, gridspec_kw={"width_ratios": [1, 1.4]})

    # UK vs International
    uk_rev = country_sales[country_sales["country"] == "United Kingdom"]["total_revenue"].sum()
    intl_rev = total_store_revenue - uk_rev
    colors_pie = ["#1F3A52", "#14B8A6"]
    ax_main.pie(
        [uk_rev, intl_rev],
        labels=["United Kingdom", "International"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors_pie,
        explode=(0.05, 0),
        textprops={"fontsize": 10, "fontweight": "bold"}
    )
    ax_main.set_title("Domestic vs International Revenue", fontsize=12, fontweight="bold", pad=10)

    # Top 8 International Markets
    intl_df = country_sales[country_sales["country"] != "United Kingdom"].head(8).iloc[::-1]
    y_intl = np.arange(len(intl_df))
    ax_intl.barh(y_intl, intl_df["total_revenue"] / 1000.0, color="#0E868A", alpha=0.85, height=0.6)
    ax_intl.set_yticks(y_intl)
    ax_intl.set_yticklabels(intl_df["country"], fontsize=9.5)
    ax_intl.set_xlabel("Revenue (£ Thousands)", fontsize=10, fontweight="bold")
    ax_intl.set_title("Top 8 International Export Markets", fontsize=12, fontweight="bold", pad=10)
    ax_intl.xaxis.set_major_formatter(ticker.FormatStrFormatter("£%.0fk"))
    ax_intl.grid(axis="x", linestyle="--", alpha=0.3)

    for i, (_, row) in enumerate(intl_df.iterrows()):
        val = row["total_revenue"] / 1000.0
        ax_intl.text(val + 3, i, f"£{val:.1f}k ({row['revenue_share_pct']}%)",
                     va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1F3A52")

    plt.suptitle("ShopGraph: Geographic Revenue Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart3_path = os.path.join(figures_dir, "03_country_sales_distribution.png")
    plt.savefig(chart3_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Generated chart: {chart3_path}")

    # ---------------------------------------------------------
    # 4. CUSTOMER PURCHASING BEHAVIOR (RFM SEGMENTATION)
    # ---------------------------------------------------------
    print("\n[EDA] Computing Customer RFM (Recency, Frequency, Monetary) Profiles...")
    max_date = df["invoice_date"].max()

    rfm = df.groupby("customer_id").agg(
        recency_days=("invoice_date", lambda x: (max_date - x.max()).total_seconds() / (24 * 3600)),
        frequency=("invoice", "nunique"),
        monetary=("line_total", "sum"),
        total_items=("quantity", "sum")
    ).reset_index()

    rfm["recency_days"] = rfm["recency_days"].round(1)
    rfm["monetary"] = rfm["monetary"].round(2)

    # Quintile scoring (1 to 5)
    # Recency: lower days = higher score
    rfm["r_score"] = pd.qcut(rfm["recency_days"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    # Frequency: higher orders = higher score (use rank method to handle ties)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Monetary: higher spend = higher score
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Segment mapping
    def assign_segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New / Promising"
        elif r == 3 and f <= 2:
            return "Needs Attention"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2:
            return "Hibernating / Inactive"
        else:
            return "Standard Customers"

    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    segment_summary = rfm.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
        total_segment_spend=("monetary", "sum")
    ).reset_index()

    segment_summary["pct_customers"] = (segment_summary["customer_count"] / len(rfm) * 100).round(2)
    segment_summary["pct_revenue"] = (segment_summary["total_segment_spend"] / rfm["monetary"].sum() * 100).round(2)
    segment_summary["avg_recency"] = segment_summary["avg_recency"].round(1)
    segment_summary["avg_frequency"] = segment_summary["avg_frequency"].round(1)
    segment_summary["avg_spend"] = segment_summary["avg_spend"].round(2)
    segment_summary["total_segment_spend"] = segment_summary["total_segment_spend"].round(2)
    segment_summary = segment_summary.sort_values("total_segment_spend", ascending=False)

    rfm_csv = os.path.join(tables_dir, "customer_rfm_segments.csv")
    segment_summary.to_csv(rfm_csv, index=False)
    print(f"  -> Saved RFM segment summary: {rfm_csv}")

    # Plot 04: RFM Distributions and Segments
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Panel A: Recency distribution
    axes[0, 0].hist(rfm["recency_days"], bins=40, color="#1F3A52", edgecolor="white", alpha=0.85)
    axes[0, 0].set_title("Customer Recency Distribution (Days Since Last Order)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Days Inactive", fontsize=9.5)
    axes[0, 0].set_ylabel("Customer Count", fontsize=9.5)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.3)

    # Panel B: Frequency distribution (log scale)
    axes[0, 1].hist(rfm["frequency"], bins=np.logspace(0, 2.5, 30), color="#0E868A", edgecolor="white", alpha=0.85)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Order Frequency Distribution (Log Scale)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Number of Orders", fontsize=9.5)
    axes[0, 1].set_ylabel("Customer Count", fontsize=9.5)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.3)

    # Panel C: Monetary Spend distribution (log scale)
    pos_monetary = rfm[rfm["monetary"] > 0]["monetary"]
    axes[1, 0].hist(pos_monetary, bins=np.logspace(1, 5, 35), color="#2A9D8F", edgecolor="white", alpha=0.85)
    axes[1, 0].set_xscale("log")
    axes[1, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("£%.0f"))
    axes[1, 0].set_title("Total Spend per Customer (Log Scale)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Total Spend (£)", fontsize=9.5)
    axes[1, 0].set_ylabel("Customer Count", fontsize=9.5)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.3)

    # Panel D: Segment Share (Customer Share vs Revenue Share)
    seg_sorted = segment_summary.sort_values("pct_revenue", ascending=True)
    y_s = np.arange(len(seg_sorted))
    bar_width = 0.38
    axes[1, 1].barh(y_s - bar_width/2, seg_sorted["pct_customers"], height=bar_width, color="#94A3B8", label="% Customers")
    axes[1, 1].barh(y_s + bar_width/2, seg_sorted["pct_revenue"], height=bar_width, color="#1F3A52", label="% Total Revenue")
    axes[1, 1].set_yticks(y_s)
    axes[1, 1].set_yticklabels(seg_sorted["segment"], fontsize=9)
    axes[1, 1].set_xlabel("Percentage (%)", fontsize=9.5, fontweight="bold")
    axes[1, 1].set_title("Customer Segments: Population vs Revenue Contribution", fontsize=11, fontweight="bold")
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(axis="x", linestyle="--", alpha=0.3)

    plt.suptitle("ShopGraph: Customer Purchasing Behavior & RFM Segmentation", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    chart4_path = os.path.join(figures_dir, "04_rfm_distribution.png")
    plt.savefig(chart4_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Generated chart: {chart4_path}")

    # ---------------------------------------------------------
    # 5. BASKET STATISTICS
    # ---------------------------------------------------------
    print("\n[EDA] Summarizing basket size and transaction statistics...")
    basket_stats = df.groupby("invoice").agg(
        items_in_basket=("quantity", "sum"),
        distinct_products=("stock_code", "nunique"),
        order_value=("line_total", "sum")
    ).reset_index()

    basket_summary = pd.DataFrame({
        "metric": [
            "Total Invoices", "Avg Items per Basket", "Median Items per Basket",
            "Avg Distinct Products per Basket", "Avg Spend per Basket (£)",
            "Median Spend per Basket (£)", "Max Spend on Single Basket (£)"
        ],
        "value": [
            f"{len(basket_stats):,}",
            f"{basket_stats['items_in_basket'].mean():.2f}",
            f"{basket_stats['items_in_basket'].median():.0f}",
            f"{basket_stats['distinct_products'].mean():.2f}",
            f"£{basket_stats['order_value'].mean():.2f}",
            f"£{basket_stats['order_value'].median():.2f}",
            f"£{basket_stats['order_value'].max():,.2f}"
        ]
    })
    basket_csv = os.path.join(tables_dir, "basket_summary.csv")
    basket_summary.to_csv(basket_csv, index=False)
    print(f"  -> Saved basket summary: {basket_csv}")

    print("\n" + "=" * 80)
    print("STAGE 2 SUMMARY - EXPLORATORY ANALYTICS COMPLETE")
    print("=" * 80)
    print(f"Total Revenue Analyzed:    £{df['line_total'].sum():,.2f}")
    print(f"Total Unique Invoices:     {df['invoice'].nunique():,}")
    print(f"Total Unique Customers:    {df['customer_id'].nunique():,}")
    print(f"UK Revenue Share:          {uk_rev / total_store_revenue * 100:.2f}%")
    print(f"Top Product:               {top_revenue.iloc[0]['canonical_description']} (£{top_revenue.iloc[0]['total_revenue']:,.2f})")
    print(f"Champions Segment Revenue: {segment_summary[segment_summary['segment']=='Champions']['pct_revenue'].values[0]:.1f}%")
    print(f"Total Stage 2 Elapsed:     {time.time() - t_start:.2f} seconds")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    outputs_dir = os.path.join(project_root, "outputs")
    run_eda(data_dir, outputs_dir)
