"""Canopy customer segmentation pipeline.

Tasks:
  evaluate_k        writes output/k_evaluation.csv
  assign_clusters   writes output/cluster_assignments.csv
  profile_clusters  writes output/cluster_profile.csv
  train_surrogate   writes output/tree_rules.txt and output/tree_agreement.csv

Dependencies: evaluate_k runs first; assign_clusters runs next; profile_clusters and
train_surrogate both need the assignments and run in parallel.
"""
import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
DATA = PROJECT / "data" / "canopy_customers.csv"
OUT = PROJECT / "output"

K = 5            # TODO: set to the number of clusters the team chooses (4 to 7)
TREE_DEPTH = 4   # TODO: set to the tree depth the team chooses (3 or 4)


@dag(schedule=None, start_date=datetime(2025, 1, 1), catchup=False, tags=["canopy"])
def canopy_segments():

    @task
    def evaluate_k():
        """Write the elbow and silhouette measurements for every k in K_RANGE."""
        import segments as s
        # TODO: load the customers, scale the features, evaluate k, and write output/k_evaluation.csv
        customers = s.load_customers(DATA)
        X = s.scale_features(customers)
        k_evaluation = s.evaluate_k(X)
        s.write_csv(k_evaluation, OUT / "k_evaluation.csv")


    @task
    def assign_clusters():
        """Write each customer's cluster for k = K."""
        import segments as s
        # TODO: write output/cluster_assignments.csv
        customers = s.load_customers(DATA)
        X = s.scale_features(customers)
        cluster_result = s.assign_clusters(customers, X, K)
        s.write_csv(cluster_result, OUT / "cluster_assignments.csv")
    
    @task
    def profile_clusters():
        """Write the size and feature averages of each cluster."""
        import segments as s
        # TODO: read the assignments and write output/cluster_profile.csv
        import pandas as pd

        customers = s.load_customers(DATA)
        assignments = pd.read_csv(OUT / "cluster_assignments.csv")

        profile = s.profile_clusters(customers, assignments)
        s.write_csv(profile, OUT / "cluster_profile.csv")


    @task
    def train_surrogate():
        """Write the tree rules and the tree's agreement with the clusters."""
        import segments as s
        # TODO: write output/tree_rules.txt and output/tree_agreement.csv
        # tree_agreement.csv has one row with two columns: tree_depth, agreement (rounded to 4 decimals)
        import pandas as pd

        customers = s.load_customers(DATA)
        assignments = pd.read_csv(OUT / "cluster_assignments.csv")

        tree = s.train_surrogate(customers, assignments, TREE_DEPTH)

        rules = s.tree_rules(tree)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "tree_rules.txt").write_text(rules, encoding="utf-8")

        agreement_result = pd.DataFrame([{
            "tree_depth": TREE_DEPTH,
            "agreement": round(s.agreement(tree, customers, assignments), 4),
        }])
        s.write_csv(agreement_result, OUT / "tree_agreement.csv")
    # TODO: create the four tasks and set the dependencies described at the top of the file
    # Create the four task instances.
    evaluation_task = evaluate_k()
    assignment_task = assign_clusters()
    profile_task = profile_clusters()
    surrogate_task = train_surrogate()

    # Evaluate k first, assign clusters next, then allow both final tasks to run.
    evaluation_task >> assignment_task
    assignment_task >> [profile_task, surrogate_task]

canopy_segments()
