"""
================================================================================
SHOPGRAPH: MASTER PIPELINE RUNNER
================================================================================
A Distributed E-Commerce Analytics and Recommendation System
Using Apache Spark, GraphX, and Spark MLlib (PySpark) on the Online Retail II Dataset.

Executes all 5 stages end-to-end:
  1. Ingestion, 7 Cleaning Rules & Interaction Signal Engineering
  2. Exploratory Analytics (Revenue, RFM, Countries, Basket Metrics)
  3. Bipartite Interaction Graph & Product Co-Purchase Network Analysis
  4. Machine Learning Recommenders (Personalized ALS, Graph, Basket Rules, Baseline)
  5. Four-Model Evaluation (P@10, R@10, NDCG@10, MAP) & Scalability Benchmarking
================================================================================
"""

import os
import sys
import time
import datetime
import importlib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, "..", "mpl_cache")

# Dynamic module loaders to cleanly load numbered files
mod_ingest = importlib.import_module("src.01_ingest_clean")
mod_eda = importlib.import_module("src.02_eda_analytics")
mod_graph = importlib.import_module("src.03_graph_analysis")
mod_als = importlib.import_module("src.04_als_recommender")
mod_fpgrowth = importlib.import_module("src.05_fpgrowth_patterns")
mod_eval = importlib.import_module("src.06_evaluation_scaling")
mod_dash = importlib.import_module("src.build_interactive_dashboard")

# Tee stdout to log file
class LoggerTee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

def main():
    log_file_path = os.path.join(LOGS_DIR, f"run_pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = LoggerTee(log_file_path)

    t_global_start = time.time()
    print("=" * 85)
    print("                SHOPGRAPH: END-TO-END SYSTEM EXECUTION                ")
    print(f"       Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"       Dataset: Online Retail II (~1.07 Million Transactions, 2009–2011)")
    print(f"       Log File: {log_file_path}")
    print("=" * 85 + "\n")

    # STAGE 1: Ingestion & Cleaning
    t1 = time.time()
    stage1_res = mod_ingest.run_ingest_and_clean(DATA_DIR, OUTPUTS_DIR)
    dur1 = time.time() - t1

    # STAGE 2: Exploratory Analytics
    t2 = time.time()
    mod_eda.run_eda(DATA_DIR, OUTPUTS_DIR)
    dur2 = time.time() - t2

    # STAGE 3: Graph Analysis
    t3 = time.time()
    mod_graph.run_graph_analysis(DATA_DIR, OUTPUTS_DIR)
    dur3 = time.time() - t3

    # STAGE 4A & 4B: Models & Patterns
    t4a = time.time()
    als_model, als_time = mod_als.run_als_recommender(DATA_DIR, OUTPUTS_DIR)
    dur4a = time.time() - t4a

    t4b = time.time()
    miner = mod_fpgrowth.run_fpgrowth_mining(DATA_DIR, OUTPUTS_DIR)
    dur4b = time.time() - t4b

    # STAGE 5: Evaluation & Scaling
    t5 = time.time()
    df_comp, df_scaling = mod_eval.run_evaluation_and_scaling(DATA_DIR, OUTPUTS_DIR)
    dur5 = time.time() - t5

    # STAGE 6: Build Self-Service Interactive Dashboard with all embedded figures
    t6 = time.time()
    dash_path_1, dash_path_2 = mod_dash.build_interactive_dashboard()
    dur6 = time.time() - t6

    total_pipeline_time = time.time() - t_global_start

    print("\n" + "=" * 85)
    print("                     SHOPGRAPH PIPELINE EXECUTION SUMMARY                    ")
    print("=" * 85)
    print(f"  Stage 1 (Ingest & 7 Cleaning Rules):     {dur1:6.2f} seconds")
    print(f"  Stage 2 (Exploratory Data Analytics):    {dur2:6.2f} seconds")
    print(f"  Stage 3 (Graph & Co-Purchase Network):   {dur3:6.2f} seconds")
    print(f"  Stage 4A (ALS Matrix Factorization):     {dur4a:6.2f} seconds")
    print(f"  Stage 4B (FP-Growth Association Rules):  {dur4b:6.2f} seconds")
    print(f"  Stage 5 (4-Model Evaluation & Scaling):  {dur5:6.2f} seconds")
    print(f"  Stage 6 (Interactive Visual Dashboard):  {dur6:6.2f} seconds")
    print("-" * 85)
    print(f"  Total Pipeline Wall-Clock Duration:      {total_pipeline_time:6.2f} seconds ({total_pipeline_time/60.0:.2f} minutes)")
    print(f"  All outputs generated and saved in:      {OUTPUTS_DIR}")
    print(f"  Interactive Dashboard Artifact:          {dash_path_2}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
