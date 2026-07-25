### Import relevant packages
import os
import pickle
import sys
import json
import os

import numpy as np
import pandas as pd

from joblib import Memory, Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    v_measure_score,
    davies_bouldin_score,
    silhouette_score
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabulate import tabulate


sys.setrecursionlimit(10000)


### Settings for caching and parallel processing

# Store the agglomerative clustering trees on disk.
# Do not clear this folder between the coarse and fine sweeps.
memory = Memory(
    location='cache/agglomerative',
    verbose=0
)

# Evaluate up to seven vectors simultaneously.
# If fewer than seven CPU cores are available, use the available number.
N_JOBS = min(7, os.cpu_count() or 1)


### Read in the necessary datapoints

full_df = pd.read_pickle(
    'datasets/full_dataframe.pkl'
)

tagged_phish_df = pd.read_pickle(
    'datasets/tagged_phish_df.pkl'
)


### Prepare values used repeatedly during evaluation

full_tags = full_df['Tag'].to_numpy()

tagged_indices = tagged_phish_df.index[
    tagged_phish_df['content_tag'] != 0
].to_numpy()

labels_true = tagged_phish_df.loc[
    tagged_indices,
    'content_tag'
].to_numpy()


### Functions for Post-Clustering Evaluation

# Internal Evaluation metrics
def internal_eval(emb_vect, labels):
    # The complete dataset is used.
    # No Silhouette sampling is performed.
    sls = silhouette_score(
        emb_vect,
        labels
    )

    dbi = davies_bouldin_score(
        emb_vect,
        labels
    )

    return [sls, dbi]


# External Evaluation metrics
def external_eval(labels_pred):
    labels_pred = np.asarray(labels_pred)

    tagged_labels_pred = labels_pred[
        tagged_indices
    ]

    e1 = adjusted_rand_score(
        labels_true,
        tagged_labels_pred
    )

    e2 = adjusted_mutual_info_score(
        labels_true,
        tagged_labels_pred
    )

    e3 = v_measure_score(
        labels_true,
        tagged_labels_pred
    )

    return [e1, e2, e3]


# Phish/Benign Homogeneity
# Probability of the major class in each cluster
def homogeneity_eval(labels):
    labels = np.asarray(labels)

    _, inverse_labels = np.unique(
        labels,
        return_inverse=True
    )

    is_phish = (
        full_tags == 'Phish'
    ).astype(int)

    is_benign = (
        full_tags == 'Benign'
    ).astype(int)

    n_phish = np.bincount(
        inverse_labels,
        weights=is_phish
    )

    n_benign = np.bincount(
        inverse_labels,
        weights=is_benign
    )

    total = n_phish + n_benign

    if np.any(total == 0):
        raise ValueError(
            'A cluster contains no Phish or Benign emails.'
        )

    major_class = np.maximum(
        n_phish,
        n_benign
    ) / total

    score = np.mean(major_class)

    return [score]


# Cluster-wise analysis of homogeneity
def cluster_analysis(df, labels, full_df):
    df['clusters'] = labels

    for i in np.unique(labels):
        cluster_indices = list(
            df[df['clusters'] == i].index
        )

        temp = full_df.iloc[
            cluster_indices
        ]

        n_phish = temp[
            temp['Tag'] == 'Phish'
        ].shape[0]

        n_benign = temp[
            temp['Tag'] == 'Benign'
        ].shape[0]

        total = n_phish + n_benign

        print(
            "For cluster",
            i,
            temp.shape[0],
            ":"
        )

        print(
            "The percentage of phishing emails are:",
            100 * (n_phish / total)
        )

        print(
            "The percentage of benign emails are:",
            100 * (n_benign / total)
        )


# Distribution of campaign tags among the clusters
def tag_analysis(tag, labels):
    import collections

    temp_indices = tagged_phish_df.index[
        tagged_phish_df['content_tag'] == tag
    ].to_numpy()

    tag_cluster_labels = np.asarray(labels)[
        temp_indices
    ]

    cluster_dist = collections.Counter(
        tag_cluster_labels
    )

    for cluster, count in cluster_dist.items():
        print(
            count,
            'emails are in cluster',
            cluster
        )

    print()


### Key-based tags to analyse the distribution

with open(
    'datasets/keys_dict.pkl',
    'rb'
) as file:
    key_dict = pickle.load(file)

tags = [
    i + 1
    for i in range(len(key_dict))
]

keys = list(
    key_dict.keys()
)


### Read the input features for clustering

with open(
    'outputs/feat_rep_e.pkl',
    'rb'
) as file:
    feat_rep_e = np.asarray(
        pickle.load(file)
    )

with open(
    'outputs/feat_rep_s.pkl',
    'rb'
) as file:
    feat_rep_s = np.asarray(
        pickle.load(file)
    )

feat_rep_u = pd.read_pickle(
    'outputs/feat_rep_u.pkl'
).to_numpy()

feat_rep_h = pd.read_pickle(
    'outputs/feat_rep_h.pkl'
).to_numpy()

print('input data read ...')


### Check that all feature representations contain the same emails

feature_representations = [
    feat_rep_e,
    feat_rep_s,
    feat_rep_u,
    feat_rep_h
]

row_counts = {
    feature.shape[0]
    for feature in feature_representations
}

if len(row_counts) != 1:
    raise ValueError(
        'All feature representations must contain '
        'the same number of rows.'
    )


### Function to combine and standardize NumPy feature arrays

def scale_features(*feature_arrays):
    combined_features = np.hstack(
        feature_arrays
    )

    scaled_features = StandardScaler().fit_transform(
        combined_features
    )

    return scaled_features


### Define input vectors as NumPy arrays

vect1 = scale_features(
    feat_rep_e
)

vect2 = scale_features(
    feat_rep_e,
    feat_rep_h
)

vect3 = scale_features(
    feat_rep_e,
    feat_rep_u
)

vect4 = scale_features(
    feat_rep_e,
    feat_rep_s
)

vect5 = scale_features(
    feat_rep_e,
    feat_rep_s,
    feat_rep_u
)

vect6 = scale_features(
    feat_rep_e,
    feat_rep_s,
    feat_rep_h
)

vect7 = scale_features(
    feat_rep_e,
    feat_rep_s,
    feat_rep_u,
    feat_rep_h
)


### Baseline vectors

base_line_1 = scale_features(
    feat_rep_u,
    feat_rep_h
)

base_line_2 = scale_features(
    feat_rep_s,
    feat_rep_u,
    feat_rep_h
)


print('input vectors defined ...')

print()
print('vect1: email body text')
print('vect2: email body text + header')
print('vect3: email body text + URL')
print('vect4: email body text + subject')
print('vect5: email body text + subject + URL')
print('vect6: email body text + subject + header')
print('vect7: email body text + subject + URL + header')
print()
print('Number of parallel workers:', N_JOBS)
print()


### Store all seven vectors

input_vects = [
    vect1,
    vect2,
    vect3,
    vect4,
    vect5,
    vect6,
    vect7
]


### Run all requested cluster numbers for one vector

def cluster_single_vector(
    input_vect,
    cluster_numbers
):
    vector_scores = []

    for n in cluster_numbers:
        ag_model = AgglomerativeClustering(
            n_clusters=n,

            # Cache the hierarchical tree.
            memory=memory,

            # Build the complete tree so that it can be reused
            # for every requested number of clusters.
            compute_full_tree=True
        ).fit(input_vect)

        labels = ag_model.labels_

        eval_metrics = (
            external_eval(labels)
            + homogeneity_eval(labels)
            + internal_eval(input_vect, labels)
        )

        vector_scores.append(
            eval_metrics
        )

    return np.asarray(
        vector_scores,
        dtype=float
    )


### Run all seven vectors in parallel

def evaluate_cluster_range(
    cluster_numbers
):
    cluster_numbers = list(
        cluster_numbers
    )

    # Each parallel task receives one vector and evaluates
    # every requested number of clusters for that vector.
    scores_by_vector = Parallel(
        n_jobs=N_JOBS,
        backend='loky',
        verbose=0
    )(
        delayed(cluster_single_vector)(
            input_vect,
            cluster_numbers
        )
        for input_vect in input_vects
    )

    # Shape:
    # seven vectors × cluster numbers × six metrics
    scores_by_vector = np.stack(
        scores_by_vector,
        axis=0
    )

    # Average each evaluation metric across all seven vectors.
    average_scores = np.mean(
        scores_by_vector,
        axis=0
    )

    eval_scores = []

    for n, scores in zip(
        cluster_numbers,
        average_scores
    ):
        result_row = list(scores)
        result_row.append(n)

        eval_scores.append(
            result_row
        )

    return eval_scores


### Normalize and combine the six evaluation metrics

def combine_scores(eval_scores):
    names = [
        'ARI',
        'AMI',
        'V-M',
        'Purity',
        'Silhouette',
        'DB-Index',
        'n_cluster'
    ]

    results_df = pd.DataFrame(
        eval_scores,
        columns=names
    )

    metric_names = [
        'ARI',
        'AMI',
        'V-M',
        'Purity',
        'Silhouette',
        'DB-Index'
    ]

    # Normalize every metric between 0 and 1.
    normalized_scores = pd.DataFrame(
        MinMaxScaler().fit_transform(
            results_df[metric_names]
        ),
        columns=metric_names
    )

    # Lower DB-Index is better.
    # Reverse it so that higher is better for all six metrics.
    normalized_scores['DB-Index'] = (
        1 - normalized_scores['DB-Index']
    )

    # Give equal weight to all six normalized metrics.
    results_df['Combined'] = (
        normalized_scores.mean(axis=1)
    )

    return results_df



def save_result(name, value):
    path = 'outputs/best_clustering_parameters.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        with open(path, 'r') as file:
            results = json.load(file)
    except FileNotFoundError:
        results = {}

    results[name] = value

    with open(path, 'w') as file:
        json.dump(results, file, indent=4)


def main():
    ### Coarse-grained parameter sweep

    print(
        'Performing coarse-grained parameter sweep ...'
    )

    coarse_cluster_numbers = range(
        30,
        100,
        3
    )

    coarse_eval_scores = evaluate_cluster_range(
        coarse_cluster_numbers
    )

    coarse_results = combine_scores(
        coarse_eval_scores
    )


    ### Identify the peak of the coarse-grained sweep

    best_index = coarse_results[
        'Combined'
    ].idxmax()

    best_n = int(
        coarse_results.loc[
            best_index,
            'n_cluster'
        ]
    )

    print()
    print(
        'Best coarse-grained cluster number:',
        best_n
    )


    ### Automatically determine the fine-grained range

    best_position = coarse_results.index.get_loc(
        best_index
    )

    lower_position = max(
        0,
        best_position - 1
    )

    upper_position = min(
        len(coarse_results) - 1,
        best_position + 1
    )

    n1 = int(
        coarse_results.iloc[
            lower_position
        ]['n_cluster']
    )

    n2 = int(
        coarse_results.iloc[
            upper_position
        ]['n_cluster']
    )

    print(
        'Fine-grained sweep range:',
        n1,
        'to',
        n2
    )


    ### Fine-grained parameter sweep

    print()
    print(
        'Performing fine-grained parameter sweep ...'
    )

    fine_cluster_numbers = range(
        n1,
        n2 + 1
    )

    fine_eval_scores = evaluate_cluster_range(
        fine_cluster_numbers
    )

    fine_results = combine_scores(
        fine_eval_scores
    )


    ### Tabulate the results

    print()
    print(
        'Results for Agglomerative:'
    )

    print(
        tabulate(
            fine_results,
            headers='keys',
            tablefmt='fancy_grid',
            showindex=False
        )
    )


    ### Select the final best number of clusters

    best_index = fine_results[
        'Combined'
    ].idxmax()

    best_n = int(
        fine_results.loc[
            best_index,
            'n_cluster'
        ]
    )

    best_score = fine_results.loc[
        best_index,
        'Combined'
    ]

    print()
    print(
        'Best number of clusters:',
        best_n
    )

    print(
        'Combined evaluation score:',
        best_score
    )

    # Agglomerative
    save_result('agglomerative', int(best_n))



if __name__ == '__main__':
    main()