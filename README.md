# ShopGraph: Enterprise Distributed E-Commerce Analytics & Recommendation Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Apache Spark & GraphX](https://img.shields.io/badge/Architecture-Apache%20Spark%20%26%20GraphX-orange.svg)](https://spark.apache.org/)
[![Dataset: UCI Online Retail II](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail%20II%20(1.07M)-green.svg)](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
[![Recommendation Accuracy](https://img.shields.io/badge/Top%20NDCG%4010-0.0711%20(FP--Growth)-emerald.svg)]()
[![Catalog Discovery](https://img.shields.io/badge/Catalog%20Coverage-40.24%25%20(Implicit%20ALS)-purple.svg)]()

ShopGraph is an end-to-end, enterprise-scale retail analytics and multi-model product recommendation intelligence platform. Built on the complete **UCI Online Retail II dataset** (1,067,371 raw transaction line items from 2009 to 2011), the platform integrates automated data cleaning, behavioral RFM segmentation, customer-product bipartite graph analytics, product co-purchase affinity networks, four comparative recommendation algorithms, and an interactive self-service executive dashboard.

---

## Executive Summary & Core Results

| Benchmark Metric | Ingested Dataset | Clean Validated Baseline | Business Value |
| :--- | :--- | :--- | :--- |
| **Gross Ingested Rows** | 1,067,371 records | 776,587 confirmed rows | Dropped 290,784 invalid / cancelled rows (72.76% high-fidelity yield) |
| **Store Turnover** | £16.53 Million (£16,529,119.13) | £13.80M UK (83.50%), £2.73M Export | Peak monthly turnover: £1.45M in November 2011 (Q4 holiday surge) |
| **Active Customer Base** | 5,852 verified accounts | 38 countries | Champions tier (23.1% of accounts) generates 53.45% of gross revenue |
| **Product Catalog Size** | 4,622 discrete SKUs | 1,183 high-affinity graph nodes | Top revenue SKU: Regency Cakestand 3 Tier (£277,656.25) |
| **Interaction Graph** | 10,474 total vertices | 480,976 bipartite edges | 99.98% of vertices reside in a single Giant Connected Component |
| **Co-Purchase Network** | 40,531 affinity edges | 8 modularity communities | High clustering coefficient (0.7473) with 1,527,187 connected triangles |
| **Recommendation Precision** | NDCG@10 = 0.0711 | Precision@10 = 0.0651 | FP-Growth Association Rules win top-rank checkout cross-sells |
| **Catalog Discovery** | 40.24% Catalog Coverage | 23x wider than baseline | Implicit Matrix Factorization (ALS) prevents filter bubble stagnation |
| **Compute Scalability** | 64.25 seconds total runtime | 3.76x speedup on 8 cores | Scales predictably from 100k to 1M records adhering to Amdahl's Law |

---

## System Architecture

ShopGraph follows a modular 5-stage pipeline with strict data contracts:

```
[ UCI Online Retail II (1.07M Raw Rows) ]
                   │
                   ▼
[ Stage 1: Automated Ingestion & 7-Stage Cleaning Funnel ]
  ├── 1. Drop Missing Customer IDs (243,007 rows)
  ├── 2. Drop Cancelled Transactions Prefix 'C' (19,540 rows)
  ├── 3. Drop Non-Positive Quantities <= 0 (3,584 rows)
  ├── 4. Drop Zero / Negative Prices <= £0.00 (42 rows)
  ├── 5. Drop Non-Product Fee Codes: POST, D, M, BANK (1,848 rows)
  └── 6. Drop Exact Duplicate Records (22,763 rows)
  Yield: 776,587 Clean Records (Matrix Density: 1.778%)
                   │
                   ▼
[ Stage 2: Exploratory Analytics & Customer RFM Segmentation ]
  ├── Monthly seasonality trajectory (Figure 01)
  ├── Top 15 revenue products (Figure 02)
  ├── Geographic domestic vs export market share (Figure 03)
  └── RFM 5x5x5 behavioral customer segmentation (Figure 04)
                   │
                   ▼
[ Stage 3: Graph Topology & Co-Purchase Projection ]
  ├── Bipartite interaction graph (10,474 nodes, 480,976 edges)
  ├── PageRank gateway hub discovery (Top hub: 85123A, Figure 06)
  ├── Projected product co-purchase network (40,531 edges, Figure 05)
  ├── 8 Louvain modularity thematic merchandise communities
  └── Spark GraphX Scala implementation (src/03_graphx_analysis.scala)
                   │
                   ▼
[ Stage 4: Four Recommendation Architectures ]
  ├── 1. Popularity Baseline (store-wide bestsellers)
  ├── 2. Graph Co-Purchase Recommender (2-hop random walk)
  ├── 3. Shopping Basket Association Rules (FP-Growth mining)
  └── 4. Implicit Matrix Factorization (ALS, 64 latent factors)
                   │
                   ▼
[ Stage 5: Rigorous Evaluation & Scalability Benchmarks ]
  ├── Temporal 80/20 train/test evaluation on 2,278 holdout buyers
  ├── Top-10 ranking metrics: Precision@10, Recall@10, NDCG@10, MAP@10
  ├── Multi-core parallel scaling (1, 2, 4, 8, 16 cores, Figure 09)
  └── Interactive Light-Themed Executive Dashboard (Outputs & Visualizations)
```

---

## 4-Model Recommendation Suite Comparison

Evaluated on 2,278 holdout test customers across standardized Top-10 ranking metrics:

| Model Architecture | Personalization Type | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Catalog Coverage (%) | Inference Latency | Operational Deployment Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Shopping Basket Rules (FP-Growth)** | Rule-based Deterministic | **0.0651** | **0.0192** | **0.0711** | **0.0310** | 8.94% | 0.93s | **Accuracy Winner**: Optimal for immediate cart checkout add-ons and cross-sell prompts |
| **Implicit Matrix Factorization (ALS)** | Latent Factor Embedding (d=64) | 0.0406 | 0.0138 | 0.0423 | 0.0165 | **40.24%** | 4.75s | **Coverage Winner**: 23x wider catalog exploration; surfaces niche long-tail preferences |
| **Graph Co-Purchase Recommender** | 2-Hop Network Random Walk | 0.0452 | 0.0134 | 0.0476 | 0.0184 | 4.22% | 0.67s | **High Explainability**: Recommendations verified via direct buyer journey paths |
| **Popularity Baseline** | Non-Personalized Frequency | 0.0392 | 0.0114 | 0.0411 | 0.0154 | 1.73% | **0.03s** | **Cold Start Fallback**: Instant default response for anonymous first-time visitors |

### Production Hybrid Recommendation Strategy
The empirical findings demonstrate that no single algorithm dominates all dimensions:
1. **Cart & Checkout Pages**: Deploy **FP-Growth Association Rules** to maximize high-confidence conversion of products currently in the basket.
2. **Homepage & Category Discovery**: Deploy **Implicit ALS Collaborative Filtering** to deliver personalized discovery across 40%+ of the product catalog.
3. **Product Detail Pages**: Deploy the **Graph Co-Purchase Recommender** to display explainable "frequently bought together" bundles.
4. **Unregistered First-Time Visitors**: Deploy the **Popularity Baseline** as an ultra-fast sub-millisecond cold start fallback.

---

## Interactive Self-Service Analytics Dashboard

The platform includes a zero-dependency, client-side executive web dashboard:
- **Location**: `outputs/ShopGraph_Interactive_Dashboard.html`
- **Design**: Clean executive light theme (off-white `#f8fafc` canvas, white cards, high-contrast slate text, non-blue accents).
- **All 9 Visualizations Embedded**: Complete base64 visual charts embedded directly alongside interactive controls.
- **Interactive Capabilities**:
  - **Persona Recommender**: Interactive multi-model testing across 4 representative customer accounts (Wholesale Bulk Retailer, Gift & Teaware Collector, Novelty Buyer, Home Decorator) with ranked prediction scores and rationales.
  - **Shopping Basket Rule Finder**: Real-time keyword search across 120 association rules with live sliders for Minimum Confidence and Minimum Lift.
  - **Customer RFM Directory**: Live filtering by RFM segment (Champions, Loyal Customers, Potential Loyalists, At Risk, Lost Customers) with customer ID search and spend sorting.
  - **Capacity Planning Simulator**: Real-time calculation of runtime, processing throughput (records/sec), and Amdahl speedup factor as data volume (100k to 2M) and CPU cores (1 to 16) adjust.

---

## Repository Structure

```
ShopGraph/
├── data/                                 # Raw and intermediate data files (auto-generated)
├── outputs/                              # Pipeline outputs, metrics, and dashboards
│   ├── figures/                          # 9 high-resolution presentation charts (PNG)
│   ├── figures_web/                      # Web-optimized image assets
│   ├── tables/                           # 14 audited CSV data tables
│   │   ├── cleaning_log.csv              # Exact dropped rows per cleaning rule
│   │   ├── monthly_sales.csv             # 24-month revenue and order trends
│   │   ├── country_sales.csv             # Geographic sales distribution
│   │   ├── customer_rfm_segments.csv     # 5,852 customer RFM classifications
│   │   ├── top_products_revenue.csv      # SKU-level revenue contribution
│   │   ├── graph_pagerank_top_products.csv # Top PageRank gateway items
│   │   ├── graph_top_copurchase_pairs.csv  # High-confidence product pairs
│   │   ├── graph_communities.csv         # 8 modularity community summaries
│   │   ├── association_rules.csv         # FP-Growth cross-sell rules
│   │   ├── model_evaluation_comparison.csv # 4-model evaluation scorecard
│   │   └── scalability_benchmarks.csv    # Multi-core runtime scaling data
│   └── ShopGraph_Interactive_Dashboard.html # Complete self-service interactive application
├── src/                                  # Modular source pipeline
│   ├── 01_ingest_clean.py                # Ingestion & 7-stage deterministic cleaning
│   ├── 02_eda_analytics.py               # Exploratory analytics, RFM, & trend charts
│   ├── 03_graph_analysis.py              # Bipartite graph, PageRank, & co-purchase projection
│   ├── 03_graphx_analysis.scala          # Spark GraphX Scala module for cluster execution
│   ├── 04_als_recommender.py             # Implicit matrix factorization engine (ALS)
│   ├── 05_fpgrowth_patterns.py           # Frequent pattern mining & association rules
│   ├── 06_evaluation_scaling.py          # 4-model evaluation & multi-core benchmark
│   └── build_interactive_dashboard.py    # Self-contained dashboard generator
├── run_all.py                            # Master Python pipeline runner
├── run_pipeline.ps1                      # Master PowerShell execution script
├── .gitignore                            # Data and cache exclusion rules
└── README.md                             # Comprehensive project documentation
```

---

## Quickstart & Execution

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `networkx`, `scikit-learn`, `openpyxl`, `pyarrow`

To install prerequisites:
```bash
pip install pandas numpy scipy matplotlib seaborn networkx scikit-learn openpyxl pyarrow
```

### Running the Complete Pipeline
To execute the entire end-to-end pipeline (data download/generation, cleaning, exploratory analytics, graph modeling, recommendation training, evaluation, scalability benchmarking, and dashboard generation):

```bash
# Using Python
python run_all.py

# Or using PowerShell
.\run_pipeline.ps1
```

### Opening the Interactive Dashboard
Open `outputs/ShopGraph_Interactive_Dashboard.html` in any modern web browser:
```powershell
Start-Process "outputs\ShopGraph_Interactive_Dashboard.html"
```

---

## Dataset Reference
- **Source**: UCI Machine Learning Repository
- **Dataset**: [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Period**: 01/12/2009 to 09/12/2011
- **Domain**: Non-store online retail based in the United Kingdom specializing in unique all-occasion giftware.

---

## License
This project is licensed under the Apache License 2.0.
