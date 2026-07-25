"""
Run the final clustering step after feature extraction.

This file was split out from `04_cluster_analysis.ipynb`.

Inputs expected from the previous feature-extraction steps:
    outputs/feat_rep_e.pkl
    outputs/feat_rep_s.pkl
    outputs/feat_rep_u.pkl
    outputs/feat_rep_h.pkl
    datasets/full_dataframe.pkl

Outputs:
    clustering_results/analysis_df.pkl
    clustering_results/analysis_df.csv

The clustering choices and hyperparameters are kept the same as the notebook:
    AgglomerativeClustering(n_clusters=44) on vect4
    DBSCAN(eps=9.4, min_samples=8) on vect4
    KMeans(n_clusters=70, init='k-means++', n_init=20, max_iter=100, random_state=0) on vect4
"""

import os
import pickle
import sys

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


sys.setrecursionlimit(10000)


def load_feature_representations():
    """Read the input vectors for clustering."""
    feat_rep_e = pd.DataFrame(pickle.load(open("outputs/feat_rep_e.pkl", "rb")))
    feat_rep_s = pd.DataFrame(pickle.load(open("outputs/feat_rep_s.pkl", "rb")))
    feat_rep_u = pd.read_pickle("outputs/feat_rep_u.pkl")
    feat_rep_h = pd.read_pickle("outputs/feat_rep_h.pkl")

    print("input data read ...")
    return feat_rep_e, feat_rep_s, feat_rep_u, feat_rep_h


def build_input_vectors(feat_rep_e, feat_rep_s, feat_rep_u, feat_rep_h):
    """Define the same scaled input vectors used in the notebook."""
    scaler = StandardScaler()

    vect1 = pd.DataFrame(scaler.fit_transform(feat_rep_e.to_numpy())).dropna()
    vect2 = pd.DataFrame(
        scaler.fit_transform(
            pd.concat([feat_rep_e, feat_rep_h], ignore_index=True, axis=1).to_numpy()
        )
    ).dropna()
    vect3 = pd.DataFrame(
        scaler.fit_transform(
            pd.concat([feat_rep_e, feat_rep_u], ignore_index=True, axis=1).to_numpy()
        )
    ).dropna()
    vect4 = pd.DataFrame(
        scaler.fit_transform(
            pd.concat([feat_rep_e, feat_rep_s], ignore_index=True, axis=1).to_numpy()
        )
    ).dropna()
    vect5 = pd.DataFrame(
        scaler.fit_transform(
            pd.concat([feat_rep_e, feat_rep_s, feat_rep_u], ignore_index=True, axis=1).to_numpy()
        )
    ).dropna()
    vect6 = pd.DataFrame(
        scaler.fit_transform(
            pd.concat([feat_rep_e, feat_rep_s, feat_rep_h], ignore_index=True, axis=1).to_numpy()
        )
    ).dropna()
    vect7 = pd.DataFrame(
        scaler.fit_transform(
            pd.concat([feat_rep_e, feat_rep_s, feat_rep_u, feat_rep_h], ignore_index=True, axis=1).to_numpy()
        )
    ).dropna()

    base_line_1 = pd.DataFrame(
        scaler.fit_transform(
            pd.concat([feat_rep_u, feat_rep_h], ignore_index=True, axis=1).to_numpy()
        )
    )
    base_line_2 = pd.DataFrame(
        scaler.fit_transform(
            pd.concat([feat_rep_s, feat_rep_u, feat_rep_h], ignore_index=True, axis=1).to_numpy()
        )
    )

    print("input vectors defined ...")

    return {
        "vect1": vect1,
        "vect2": vect2,
        "vect3": vect3,
        "vect4": vect4,
        "vect5": vect5,
        "vect6": vect6,
        "vect7": vect7,
        "base_line_1": base_line_1,
        "base_line_2": base_line_2,
    }


def run_final_clustering(input_vect):
    """Run the final clustering models using the original notebook settings."""
    agg_results = AgglomerativeClustering(n_clusters=44).fit_predict(input_vect)
    db_results = DBSCAN(eps=9.4, min_samples=8).fit_predict(input_vect)
    km_results = KMeans(
        n_clusters=70,
        init="k-means++",
        n_init=20,
        max_iter=100,
        random_state=0,
    ).fit_predict(input_vect)

    return agg_results, db_results, km_results


def build_analysis_dataframe(full_df, agg_results, db_results, km_results):
    """Create the dataframe containing message IDs, true tags, and cluster labels."""
    analysis_df = pd.DataFrame()
    analysis_df["msg_id"] = full_df["msg_id"]
    analysis_df["tag"] = full_df["Tag"]
    analysis_df["agg_labels"] = agg_results
    analysis_df["db_labels"] = db_results
    analysis_df["km_labels"] = km_results

    return analysis_df


def main():
    full_df = pd.read_pickle("datasets/full_dataframe.pkl")

    feat_rep_e, feat_rep_s, feat_rep_u, feat_rep_h = load_feature_representations()
    input_vectors = build_input_vectors(feat_rep_e, feat_rep_s, feat_rep_u, feat_rep_h)

    # The notebook uses vect4 for the final clustering results.
    input_vect = input_vectors["vect4"]

    agg_results, db_results, km_results = run_final_clustering(input_vect)
    analysis_df = build_analysis_dataframe(full_df, agg_results, db_results, km_results)

    os.makedirs("clustering_results", exist_ok=True)
    analysis_df.to_pickle("clustering_results/analysis_df.pkl")
    analysis_df.to_csv("clustering_results/analysis_df.csv", index=False)

    print("clustering complete ...")
    print("saved clustering_results/analysis_df.pkl")
    print("saved clustering_results/analysis_df.csv")


if __name__ == "__main__":
    main()
