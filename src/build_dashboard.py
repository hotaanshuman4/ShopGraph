"""
ShopGraph - Interactive Dashboard Generator
Generates a presentation-ready HTML executive dashboard with embedded figures,
interactive tabs, cleaning logs, and model comparison matrices.
"""

import os
import json

def generate_dashboard():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    b64_json = os.path.join(project_root, "outputs", "figures_b64.json")
    with open(b64_json, "r") as f:
        figs = json.load(f)

    artifact_dir = r"C:\Users\hota\.gemini\antigravity\brain\f3811c5f-d514-48f1-b866-f2094745df20"
    os.makedirs(artifact_dir, exist_ok=True)
    report_link = f"file:///{artifact_dir.replace(os.sep, '/')}/ShopGraph_Project_Report.md"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ShopGraph - Distributed E-Commerce Analytics & Recommendation System</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    .tab-btn.active {{
      background: #1F3A52;
      color: #ffffff;
      border-color: #1F3A52;
    }}
    .stat-card {{
      background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
      border: 1px solid #E2E8F0;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }}
    .gradient-header {{
      background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #0E868A 100%);
    }}
  </style>
</head>
<body class="bg-slate-50 text-slate-900 antialiased min-h-screen">

  <!-- HEADER HERO -->
  <header class="gradient-header text-white py-8 px-6 shadow-xl relative overflow-hidden">
    <div class="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative z-10">
      <div>
        <div class="flex items-center gap-3 mb-2">
          <span class="px-2.5 py-1 text-xs font-bold rounded-full bg-teal-400 text-slate-900 tracking-wider uppercase">INT315 · Cluster Computing</span>
          <span class="px-2.5 py-1 text-xs font-semibold rounded-full bg-slate-800 text-teal-300 border border-teal-500/30">Production Verified</span>
          <span class="text-xs text-slate-300">Execution Time: 64.25s</span>
        </div>
        <h1 class="text-3xl md:text-4xl font-extrabold tracking-tight">
          ShopGraph <span class="text-teal-400 font-light">System Dashboard</span>
        </h1>
        <p class="text-slate-300 text-sm mt-1 max-w-2xl">
          Distributed E-Commerce Analytics, Bipartite Interaction Graph & Product Recommendation System on Apache Spark, GraphX, and Spark MLlib.
        </p>
      </div>
      <div class="flex flex-wrap items-center gap-3">
        <a href="{report_link}" 
           class="px-4 py-2.5 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-lg transition-all shadow-md flex items-center gap-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
          Full Project Report (Markdown)
        </a>
      </div>
    </div>
  </header>

  <!-- QUICK KPI HERO METRICS -->
  <section class="max-w-7xl mx-auto px-6 -mt-6 relative z-20">
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <div class="stat-card rounded-xl p-4 border border-slate-200">
        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Raw Invoices</p>
        <p class="text-xl font-extrabold text-slate-900 mt-1">1,067,371</p>
        <span class="text-[11px] text-slate-500">2 Years (2009–2011)</span>
      </div>
      <div class="stat-card rounded-xl p-4 border border-slate-200">
        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Cleaned Records</p>
        <p class="text-xl font-extrabold text-teal-700 mt-1">776,587</p>
        <span class="text-[11px] text-teal-600 font-medium">72.76% Kept (7 Rules)</span>
      </div>
      <div class="stat-card rounded-xl p-4 border border-slate-200">
        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Active Buyers</p>
        <p class="text-xl font-extrabold text-slate-900 mt-1">5,852</p>
        <span class="text-[11px] text-slate-500">4,622 Products</span>
      </div>
      <div class="stat-card rounded-xl p-4 border border-slate-200">
        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Total Store Sales</p>
        <p class="text-xl font-extrabold text-emerald-700 mt-1">£16.53M</p>
        <span class="text-[11px] text-slate-500">UK Share 83.50%</span>
      </div>
      <div class="stat-card rounded-xl p-4 border border-slate-200">
        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Top Precision</p>
        <p class="text-xl font-extrabold text-indigo-700 mt-1">0.0651</p>
        <span class="text-[11px] text-indigo-600 font-medium">+66.1% over Baseline</span>
      </div>
      <div class="stat-card rounded-xl p-4 border border-slate-200">
        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">ALS Coverage</p>
        <p class="text-xl font-extrabold text-blue-700 mt-1">40.24%</p>
        <span class="text-[11px] text-blue-600 font-medium">23x vs Popularity</span>
      </div>
    </div>
  </section>

  <!-- NAVIGATION TABS -->
  <nav class="max-w-7xl mx-auto px-6 mt-8">
    <div class="flex items-center gap-2 border-b border-slate-200 pb-3 overflow-x-auto">
      <button onclick="showTab('overview')" id="btn-overview" class="tab-btn active px-4 py-2 text-xs font-bold rounded-lg border border-slate-200 transition-all flex items-center gap-2">
        <span>📊</span> Executive Summary
      </button>
      <button onclick="showTab('cleaning')" id="btn-cleaning" class="tab-btn px-4 py-2 text-xs font-bold rounded-lg border border-slate-200 transition-all flex items-center gap-2">
        <span>🧹</span> 1. Cleaning Funnel
      </button>
      <button onclick="showTab('analytics')" id="btn-analytics" class="tab-btn px-4 py-2 text-xs font-bold rounded-lg border border-slate-200 transition-all flex items-center gap-2">
        <span>📈</span> 2. Sales & RFM Analytics
      </button>
      <button onclick="showTab('graph')" id="btn-graph" class="tab-btn px-4 py-2 text-xs font-bold rounded-lg border border-slate-200 transition-all flex items-center gap-2">
        <span>🕸️</span> 3. GraphX Network
      </button>
      <button onclick="showTab('recommenders')" id="btn-recommenders" class="tab-btn px-4 py-2 text-xs font-bold rounded-lg border border-slate-200 transition-all flex items-center gap-2">
        <span>🤖</span> 4. Recommendation Models
      </button>
      <button onclick="showTab('scaling')" id="btn-scaling" class="tab-btn px-4 py-2 text-xs font-bold rounded-lg border border-slate-200 transition-all flex items-center gap-2">
        <span>⚡</span> 5. Evaluation & Scaling
      </button>
    </div>
  </nav>

  <!-- MAIN TAB CONTENT CONTAINER -->
  <main class="max-w-7xl mx-auto px-6 py-6">

    <!-- TAB 1: EXECUTIVE OVERVIEW -->
    <div id="tab-overview" class="tab-content space-y-6">
      <div class="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
        <div class="flex items-center justify-between pb-4 border-b border-slate-100">
          <div>
            <h2 class="text-xl font-bold text-slate-900">Project Executive Briefing</h2>
            <p class="text-xs text-slate-500 mt-0.5">High-level synthesis of system architecture, findings, and model competition.</p>
          </div>
          <span class="px-3 py-1 bg-emerald-50 text-emerald-700 text-xs font-semibold rounded-full border border-emerald-200">
            All 5 Stages Successfully Executed
          </span>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div class="space-y-4">
            <h3 class="text-sm font-bold text-slate-800 uppercase tracking-wide">System Highlights & Key Discoveries</h3>
            <ul class="space-y-2.5 text-xs text-slate-600">
              <li class="flex items-start gap-2">
                <span class="text-teal-600 font-bold">✔</span>
                <span><strong>Data Integrity Ensured</strong>: Ingested 1,067,371 rows and cleanly eliminated 290,784 noise records (returns, missing IDs, free items, non-product fees, and wholesale extremes) using a 7-rule pipeline.</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-teal-600 font-bold">✔</span>
                <span><strong>Structural Anchor Discovered</strong>: While <em>Regency Cakestand 3 Tier</em> generated the top sales (£277.6k), GraphX PageRank revealed <em>White Hanging Heart T-Light Holder</em> as the true topological anchor product connecting the entire customer graph.</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-teal-600 font-bold">✔</span>
                <span><strong>Giant Connected Component</strong>: 99.98% of the customer-product graph belongs to a single giant component partitioned into 8 cohesive product communities.</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-teal-600 font-bold">✔</span>
                <span><strong>Recommender Complementarity</strong>: Association rules excel at immediate basket cross-sell (Precision@10 = 0.0651), while Implicit ALS solves catalog starvation by providing 40.24% catalog coverage.</span>
              </li>
            </ul>
          </div>

          <div class="bg-slate-50 rounded-xl p-5 border border-slate-200">
            <h3 class="text-sm font-bold text-slate-800 uppercase tracking-wide mb-3">Master Pipeline Execution Timeline</h3>
            <div class="space-y-3 text-xs">
              <div class="flex justify-between items-center">
                <span class="text-slate-600">Stage 1: Ingestion & 7 Cleaning Rules</span>
                <span class="font-mono font-bold text-slate-800">3.82s</span>
              </div>
              <div class="w-full bg-slate-200 rounded-full h-1.5"><div class="bg-teal-500 h-1.5 rounded-full" style="width: 6%"></div></div>

              <div class="flex justify-between items-center">
                <span class="text-slate-600">Stage 2: Exploratory SQL Analytics</span>
                <span class="font-mono font-bold text-slate-800">4.37s</span>
              </div>
              <div class="w-full bg-slate-200 rounded-full h-1.5"><div class="bg-teal-500 h-1.5 rounded-full" style="width: 7%"></div></div>

              <div class="flex justify-between items-center">
                <span class="text-slate-600">Stage 3: GraphX Network & Co-Purchase</span>
                <span class="font-mono font-bold text-slate-800">27.02s</span>
              </div>
              <div class="w-full bg-slate-200 rounded-full h-1.5"><div class="bg-teal-600 h-1.5 rounded-full" style="width: 42%"></div></div>

              <div class="flex justify-between items-center">
                <span class="text-slate-600">Stage 4: ALS Recommender & FP-Growth</span>
                <span class="font-mono font-bold text-slate-800">12.52s</span>
              </div>
              <div class="w-full bg-slate-200 rounded-full h-1.5"><div class="bg-teal-500 h-1.5 rounded-full" style="width: 19%"></div></div>

              <div class="flex justify-between items-center">
                <span class="text-slate-600">Stage 5: Evaluation & Multi-Core Scaling</span>
                <span class="font-mono font-bold text-slate-800">16.52s</span>
              </div>
              <div class="w-full bg-slate-200 rounded-full h-1.5"><div class="bg-teal-500 h-1.5 rounded-full" style="width: 26%"></div></div>

              <div class="pt-2 border-t border-slate-200 flex justify-between items-center font-bold text-slate-900">
                <span>Total End-to-End Elapsed Time</span>
                <span class="text-teal-700 font-mono text-sm">64.25 seconds (1.07 min)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Quick Dual Visual Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-2">Monthly Revenue Seasonality</h3>
          <img src="{figs['01_monthly_sales_trend.png']}" alt="Monthly Sales Trend" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-2">Model Evaluation Comparison</h3>
          <img src="{figs['08_model_evaluation_metrics.png']}" alt="Model Evaluation Metrics" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
      </div>
    </div>

    <!-- TAB 2: CLEANING & INGESTION -->
    <div id="tab-cleaning" class="tab-content hidden space-y-6">
      <div class="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
        <h2 class="text-xl font-bold text-slate-900 mb-1">Data Validation & Cleaning Funnel (7 Rules)</h2>
        <p class="text-xs text-slate-500 mb-5">Row tracking across each filtration rule to eliminate returns, cancellations, and non-commercial records.</p>
        
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs border-collapse">
            <thead>
              <tr class="bg-slate-100 text-slate-700 border-b border-slate-200 font-bold uppercase tracking-wider">
                <th class="py-3 px-4">Step</th>
                <th class="py-3 px-4">Validation Rule</th>
                <th class="py-3 px-4">Domain Justification</th>
                <th class="py-3 px-4 text-right">Rows Before</th>
                <th class="py-3 px-4 text-right">Rows After</th>
                <th class="py-3 px-4 text-right">Dropped</th>
                <th class="py-3 px-4 text-right">Step Drop %</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 text-slate-700">
              <tr class="hover:bg-slate-50"><td class="py-2.5 px-4 font-bold">0</td><td class="py-2.5 px-4">Raw Ingestion</td><td>Both sheets from 2009–2011 workbook</td><td class="py-2.5 px-4 text-right font-mono">1,067,371</td><td class="py-2.5 px-4 text-right font-mono">1,067,371</td><td class="py-2.5 px-4 text-right font-mono text-slate-400">0</td><td class="py-2.5 px-4 text-right font-mono">0.00%</td></tr>
              <tr class="hover:bg-slate-50"><td class="py-2.5 px-4 font-bold">1</td><td class="py-2.5 px-4 font-semibold text-rose-700">Drop Null Customer ID</td><td>Cannot attach interactions to unidentified guest buyers</td><td class="py-2.5 px-4 text-right font-mono">1,067,371</td><td class="py-2.5 px-4 text-right font-mono">824,364</td><td class="py-2.5 px-4 text-right font-mono text-rose-600">243,007</td><td class="py-2.5 px-4 text-right font-mono font-bold text-rose-600">22.77%</td></tr>
              <tr class="hover:bg-slate-50"><td class="py-2.5 px-4 font-bold">2</td><td class="py-2.5 px-4 font-semibold text-rose-700">Remove Cancellations & Returns</td><td>Invoice 'C' prefix and non-positive quantity are returns</td><td class="py-2.5 px-4 text-right font-mono">824,364</td><td class="py-2.5 px-4 text-right font-mono">805,620</td><td class="py-2.5 px-4 text-right font-mono text-rose-600">18,744</td><td class="py-2.5 px-4 text-right font-mono text-rose-600">2.27%</td></tr>
              <tr class="hover:bg-slate-50"><td class="py-2.5 px-4 font-bold">3</td><td class="py-2.5 px-4">Remove Price &le; 0</td><td>Free or erroneous adjustments carry no purchase intent</td><td class="py-2.5 px-4 text-right font-mono">805,620</td><td class="py-2.5 px-4 text-right font-mono">805,549</td><td class="py-2.5 px-4 text-right font-mono">71</td><td class="py-2.5 px-4 text-right font-mono">0.01%</td></tr>
              <tr class="hover:bg-slate-50"><td class="py-2.5 px-4 font-bold">4</td><td class="py-2.5 px-4">Exclude Non-Product Codes</td><td>Postage, bank fees, manual adjustments (POST, DOT, M)</td><td class="py-2.5 px-4 text-right font-mono">805,549</td><td class="py-2.5 px-4 text-right font-mono">802,644</td><td class="py-2.5 px-4 text-right font-mono">2,905</td><td class="py-2.5 px-4 text-right font-mono">0.36%</td></tr>
              <tr class="hover:bg-slate-50"><td class="py-2.5 px-4 font-bold">5</td><td class="py-2.5 px-4">Drop Exact Duplicate Lines</td><td>Prevent double-counting identical records</td><td class="py-2.5 px-4 text-right font-mono">802,644</td><td class="py-2.5 px-4 text-right font-mono">776,587</td><td class="py-2.5 px-4 text-right font-mono">26,057</td><td class="py-2.5 px-4 text-right font-mono">3.25%</td></tr>
              <tr class="hover:bg-slate-50"><td class="py-2.5 px-4 font-bold">6</td><td class="py-2.5 px-4">Cap Extreme Quantities (P99.9)</td><td>Limit pull of rare bulk wholesale orders on weights</td><td class="py-2.5 px-4 text-right font-mono">776,587</td><td class="py-2.5 px-4 text-right font-mono">776,587</td><td class="py-2.5 px-4 text-right font-mono text-slate-400">0</td><td class="py-2.5 px-4 text-right font-mono">0.00%</td></tr>
              <tr class="hover:bg-slate-50"><td class="py-2.5 px-4 font-bold">7</td><td class="py-2.5 px-4">Standardize Descriptions</td><td>Assign mode canonical product text per stock code</td><td class="py-2.5 px-4 text-right font-mono">776,587</td><td class="py-2.5 px-4 text-right font-mono">776,587</td><td class="py-2.5 px-4 text-right font-mono text-slate-400">0</td><td class="py-2.5 px-4 text-right font-mono">0.00%</td></tr>
            </tbody>
            <tfoot>
              <tr class="bg-teal-50 text-teal-900 font-bold border-t-2 border-teal-500">
                <td colspan="3" class="py-3 px-4">FINAL CLEANED TRANSACTIONS RETAINED</td>
                <td class="py-3 px-4 text-right font-mono">1,067,371</td>
                <td class="py-3 px-4 text-right font-mono">776,587</td>
                <td class="py-3 px-4 text-right font-mono">290,784</td>
                <td class="py-3 px-4 text-right font-mono">72.76% Kept (27.24% Dropped)</td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 3: SALES & RFM ANALYTICS -->
    <div id="tab-analytics" class="tab-content hidden space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-1">Monthly Sales & Order Volume</h3>
          <p class="text-xs text-slate-500 mb-3">Pronounced Q4 seasonal spike driven by holiday shopping.</p>
          <img src="{figs['01_monthly_sales_trend.png']}" alt="Monthly Sales Trend" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-1">Top 15 Revenue-Generating Products</h3>
          <p class="text-xs text-slate-500 mb-3">Regency Cakestand 3 Tier leads total revenue at £277.6k.</p>
          <img src="{figs['02_top_products_revenue.png']}" alt="Top Products by Revenue" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-1">Geographic Revenue Breakdown</h3>
          <p class="text-xs text-slate-500 mb-3">UK domestic retail dominates at 83.50% with major European export destinations.</p>
          <img src="{figs['03_country_sales_distribution.png']}" alt="Country Sales Distribution" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-1">Customer RFM Segmentation</h3>
          <p class="text-xs text-slate-500 mb-3">Recency, Frequency, and Monetary quintile classification.</p>
          <img src="{figs['04_rfm_distribution.png']}" alt="RFM Distributions" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
      </div>
    </div>

    <!-- TAB 4: GRAPHX NETWORK -->
    <div id="tab-graph" class="tab-content hidden space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-1">Co-Purchase Product Network</h3>
          <p class="text-xs text-slate-500 mb-3">Node size = PageRank Centrality, Node color = Modularity Community, Edge = Shared Buyers.</p>
          <img src="{figs['05_copurchase_network.png']}" alt="Co-Purchase Affinity Network" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-1">Structural Centrality (PageRank) vs Total Sales</h3>
          <p class="text-xs text-slate-500 mb-3">Identifies products that bridge distinct buying themes beyond raw sales volume.</p>
          <img src="{figs['06_graph_degree_pagerank.png']}" alt="Degree and PageRank Centrality" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
      </div>

      <div class="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
        <h3 class="text-sm font-bold text-slate-900 mb-3">Graph Properties & Topology Summary</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200"><span class="text-slate-500">Total Bipartite Vertices</span><p class="font-bold text-slate-900 font-mono text-sm mt-0.5">10,474</p></div>
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200"><span class="text-slate-500">Bipartite Edges</span><p class="font-bold text-slate-900 font-mono text-sm mt-0.5">480,976</p></div>
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200"><span class="text-slate-500">Giant Connected Component</span><p class="font-bold text-teal-700 font-mono text-sm mt-0.5">99.98% (10,472 nodes)</p></div>
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200"><span class="text-slate-500">Top PageRank Product</span><p class="font-bold text-slate-900 font-mono text-xs mt-0.5">White Hanging Heart T-Light (0.002997)</p></div>
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200"><span class="text-slate-500">Projected Co-Purchase Nodes</span><p class="font-bold text-slate-900 font-mono text-sm mt-0.5">1,183</p></div>
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200"><span class="text-slate-500">Projected Co-Purchase Edges</span><p class="font-bold text-slate-900 font-mono text-sm mt-0.5">40,531</p></div>
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200"><span class="text-slate-500">Product Communities Found</span><p class="font-bold text-indigo-700 font-mono text-sm mt-0.5">8 Clusters</p></div>
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200"><span class="text-slate-500">Avg Clustering Coefficient</span><p class="font-bold text-slate-900 font-mono text-sm mt-0.5">0.7473 (1.52M Triangles)</p></div>
        </div>
      </div>
    </div>

    <!-- TAB 5: RECOMMENDERS -->
    <div id="tab-recommenders" class="tab-content hidden space-y-6">
      <div class="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
        <h2 class="text-xl font-bold text-slate-900 mb-1">Personalized Recommendations for Sample Customer Personas</h2>
        <p class="text-xs text-slate-500 mb-4">Generated by Implicit ALS Matrix Factorization comparing historical purchases to top-10 recommendations.</p>
        <img src="{figs['07_sample_recommendations.png']}" alt="Sample Customer Recommendations Card" class="rounded-lg w-full object-contain border border-slate-100">
      </div>
    </div>

    <!-- TAB 6: EVALUATION & SCALING -->
    <div id="tab-scaling" class="tab-content hidden space-y-6">
      <div class="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
        <h2 class="text-xl font-bold text-slate-900 mb-1">Four-Model Benchmark Quality Matrix</h2>
        <p class="text-xs text-slate-500 mb-4">Evaluated on 2,278 active customers in the 20% temporal holdout test period (K = 10).</p>
        
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs border-collapse">
            <thead>
              <tr class="bg-slate-100 text-slate-700 border-b border-slate-200 font-bold uppercase tracking-wider">
                <th class="py-3 px-4">Model Approach</th>
                <th class="py-3 px-4 text-right">Precision@10</th>
                <th class="py-3 px-4 text-right">Recall@10</th>
                <th class="py-3 px-4 text-right">NDCG@10</th>
                <th class="py-3 px-4 text-right">MAP@10</th>
                <th class="py-3 px-4 text-right">Catalog Coverage (%)</th>
                <th class="py-3 px-4 text-right">Training Time</th>
                <th class="py-3 px-4">Personalized</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 text-slate-700 font-mono">
              <tr class="hover:bg-slate-50"><td class="py-3 px-4 font-sans font-bold">Popularity Baseline</td><td class="py-3 px-4 text-right">0.0392</td><td class="py-3 px-4 text-right">0.0114</td><td class="py-3 px-4 text-right">0.0411</td><td class="py-3 px-4 text-right">0.0154</td><td class="py-3 px-4 text-right">1.73%</td><td class="py-3 px-4 text-right font-bold text-teal-600">0.03s</td><td class="py-3 px-4 font-sans text-slate-400">No</td></tr>
              <tr class="hover:bg-slate-50"><td class="py-3 px-4 font-sans font-bold">Graph Co-Purchase</td><td class="py-3 px-4 text-right">0.0452</td><td class="py-3 px-4 text-right">0.0134</td><td class="py-3 px-4 text-right">0.0476</td><td class="py-3 px-4 text-right">0.0184</td><td class="py-3 px-4 text-right">4.22%</td><td class="py-3 px-4 text-right">0.65s</td><td class="py-3 px-4 font-sans text-teal-700">Graph-based</td></tr>
              <tr class="hover:bg-slate-50 bg-indigo-50/40"><td class="py-3 px-4 font-sans font-bold text-indigo-900">Shopping Basket Rules (FP-Growth)</td><td class="py-3 px-4 text-right font-bold text-indigo-700">0.0651</td><td class="py-3 px-4 text-right font-bold text-indigo-700">0.0192</td><td class="py-3 px-4 text-right font-bold text-indigo-700">0.0711</td><td class="py-3 px-4 text-right font-bold text-indigo-700">0.0310</td><td class="py-3 px-4 text-right">8.94%</td><td class="py-3 px-4 text-right">0.92s</td><td class="py-3 px-4 font-sans text-indigo-700 font-semibold">Rule-based (Winner)</td></tr>
              <tr class="hover:bg-slate-50 bg-blue-50/40"><td class="py-3 px-4 font-sans font-bold text-blue-900">Personalized Implicit ALS</td><td class="py-3 px-4 text-right">0.0406</td><td class="py-3 px-4 text-right">0.0138</td><td class="py-3 px-4 text-right">0.0423</td><td class="py-3 px-4 text-right">0.0165</td><td class="py-3 px-4 text-right font-bold text-blue-700">40.24%</td><td class="py-3 px-4 text-right">4.55s</td><td class="py-3 px-4 font-sans text-blue-700 font-semibold">Latent Factors (23x Coverage)</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-1">Evaluation Ranking Metrics</h3>
          <p class="text-xs text-slate-500 mb-3">Precision@10, Recall@10, NDCG@10, and MAP@10.</p>
          <img src="{figs['08_model_evaluation_metrics.png']}" alt="Model Evaluation Metrics" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-1">System Scalability Curves</h3>
          <p class="text-xs text-slate-500 mb-3">Data slice runtime scaling & multi-core speedup curve.</p>
          <img src="{figs['09_scalability_curves.png']}" alt="Scalability Curves" class="rounded-lg w-full object-contain border border-slate-100">
        </div>
      </div>
    </div>

  </main>

  <!-- JAVASCRIPT FOR INTERACTIVE TABS -->
  <script>
    function showTab(tabId) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      const activeContent = document.getElementById('tab-' + tabId);
      const activeBtn = document.getElementById('btn-' + tabId);
      if (activeContent) activeContent.classList.remove('hidden');
      if (activeBtn) activeBtn.classList.add('active');
    }}
  </script>

</body>
</html>
"""

    dashboard_path = os.path.join(artifact_dir, "ShopGraph_Presentation_Dashboard.html")
    local_dash_path = os.path.join(project_root, "outputs", "ShopGraph_Presentation_Dashboard.html")

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(local_dash_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Dashboard generated at:", dashboard_path, "size:", os.path.getsize(dashboard_path)/1024, "KB")

if __name__ == "__main__":
    generate_dashboard()
