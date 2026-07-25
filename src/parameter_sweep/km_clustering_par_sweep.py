### Import relevant packages

import os
import pickle
import sys

import numpy as np
import pandas as pd
import json
import os

from joblib import Parallel, delayed, parallel_config

from sklearn.cluster import KMeans
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


### Parallel processing settings

# Process up to seven vectors simultaneously
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
].to_numpy(dtype=int)

labels_true = tagged_phish_df.loc[
    tagged_indices,
    'content_tag'
].to_numpy()


### Functions for Post-Clustering Evaluation


# Internal Evaluation metrics
def internal_eval(emb_vect, labels):

    # Complete dataset is used.
    # No Silhouette sampling is performed.
    sls = silhouette_score(
        emb_vect,
        labels
    )

    # Store the actual Davies-Bouldin Index.
    # Lower values are better.
    dbi = davies_bouldin_score(
        emb_vect,
        labels
    )

    return [sls, dbi]


# External Evaluation metrics
def external_eval(labels_pred):

    labels_pred = np.asarray(
        labels_pred
    )

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
# Probability of major class in each cluster
def homogeneity_eval(labels):

    labels = np.asarray(
        labels
    )

    _, inverse_labels = np.unique(
        labels,
        return_inverse=True
    )

    is_phish = (
        full_tags == 'Phish'
    ).astype(float)

    is_benign = (
        full_tags == 'Benign'
    ).astype(float)

    n_phish = np.bincount(
        inverse_labels,
        weights=is_phish
    )

    n_benign = np.bincount(
        inverse_labels,
        weights=is_benign
    )

    total = (
        n_phish
        + n_benign
    )

    if np.any(total == 0):
        raise ValueError(
            'A cluster contains no Phish or Benign emails.'
        )

    major_class = (
        np.maximum(
            n_phish,
            n_benign
        )
        / total
    )

    score = np.mean(
        major_class
    )

    return [score]


# Cluster-wise analysis of homogeneity
def cluster_analysis(labels, full_df):

    labels = np.asarray(
        labels
    )

    for cluster in np.unique(labels):

        temp = full_df[
            labels == cluster
        ]

        n_phish = temp[
            temp['Tag'] == 'Phish'
        ].shape[0]

        n_benign = temp[
            temp['Tag'] == 'Benign'
        ].shape[0]

        total = (
            n_phish
            + n_benign
        )

        print(
            "For cluster",
            cluster,
            temp.shape[0],
            ":"
        )

        print(
            "The percentage of phishing emails are:",
            100 * (
                n_phish / total
            )
        )

        print(
            "The percentage of benign emails are:",
            100 * (
                n_benign / total
            )
        )


# Distribution of campaign tags among the clusters
def tag_analysis(tag, labels):

    import collections

    labels = np.asarray(
        labels
    )

    temp_indices = tagged_phish_df.index[
        tagged_phish_df['content_tag'] == tag
    ].to_numpy(dtype=int)

    cluster_dist = collections.Counter(
        labels[temp_indices]
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

    key_dict = pickle.load(
        file
    )


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


print(
    'input data read ...'
)


### Check that all feature representations
### contain the same number of emails

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


if next(iter(row_counts)) != len(full_df):

    raise ValueError(
        'The feature representations and full_dataframe.pkl '
        'must contain the same number of rows.'
    )


### Function to combine and standardize NumPy features

def scale_features(*feature_arrays):

    combined_features = np.hstack(
        feature_arrays
    )

    scaled_features = StandardScaler().fit_transform(
        combined_features
    )

    return np.ascontiguousarray(
        scaled_features
    )


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


print(
    'input vectors defined ...'
)


print()

print(
    'Vector 1: email body text'
)

print(
    'Vector 2: email body text + header'
)

print(
    'Vector 3: email body text + URL'
)

print(
    'Vector 4: email body text + subject'
)

print(
    'Vector 5: email body text + subject + URL'
)

print(
    'Vector 6: email body text + subject + header'
)

print(
    'Vector 7: email body text + subject + URL + header'
)

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


### Run K-Means for one vector across
### all requested cluster numbers

def cluster_single_vector(
    input_vect,
    cluster_numbers
):

    vector_scores = []

    for n in cluster_numbers:

        kmeans_model = KMeans(
            n_clusters=n,
            init='k-means++',
            n_init=20,
            max_iter=100,
            random_state=0
        )

        labels = kmeans_model.fit_predict(
            input_vect
        )

        eval_metrics = (
            external_eval(labels)
            + homogeneity_eval(labels)
            + internal_eval(
                input_vect,
                labels
            )
        )

        vector_scores.append(
            eval_metrics
        )


    return np.asarray(
        vector_scores,
        dtype=float
    )


### Store previously computed cluster results
###
### This means values already evaluated during
### the coarse sweep are not calculated again
### during the fine-grained sweep.

cluster_score_cache = {}


### Run requested cluster numbers across
### all seven vectors in parallel

def evaluate_cluster_range(
    cluster_numbers
):

    cluster_numbers = [
        int(n)
        for n in cluster_numbers
    ]


    ### Determine which cluster numbers
    ### have not already been calculated

    missing_cluster_numbers = [
        n
        for n in cluster_numbers
        if n not in cluster_score_cache
    ]


    if missing_cluster_numbers:

        print(
            'Evaluating cluster numbers:',
            missing_cluster_numbers
        )


        ### Run one vector per parallel worker.
        ###
        ### inner_max_num_threads=1 prevents each K-Means
        ### worker from creating another large group of
        ### numerical threads.

        with parallel_config(
            backend='loky',
            n_jobs=N_JOBS,
            inner_max_num_threads=1,
            max_nbytes='1M',
            mmap_mode='r'
        ):

            scores_by_vector = Parallel()(
                delayed(
                    cluster_single_vector
                )(
                    input_vect,
                    missing_cluster_numbers
                )

                for input_vect
                in input_vects
            )


        ### Shape:
        ###
        ### 7 vectors
        ### x number of cluster values
        ### x 6 evaluation metrics

        scores_by_vector = np.stack(
            scores_by_vector,
            axis=0
        )


        ### Average each metric across
        ### all seven vectors

        average_scores = np.mean(
            scores_by_vector,
            axis=0
        )


        ### Add results to cache

        for n, scores in zip(
            missing_cluster_numbers,
            average_scores
        ):

            cluster_score_cache[n] = list(
                scores
            )


    ### Return requested cluster numbers
    ### in the requested order

    eval_scores = []

    for n in cluster_numbers:

        temp = list(
            cluster_score_cache[n]
        )

        temp.append(
            n
        )

        eval_scores.append(
            temp
        )


    return eval_scores


### Function to normalize and combine
### the six evaluation metrics

def combine_scores(
    eval_scores
):

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


    ### Normalize all evaluation measures
    ### between 0 and 1

    normalized_scores = pd.DataFrame(
        MinMaxScaler().fit_transform(
            results_df[
                metric_names
            ]
        ),
        columns=metric_names
    )


    ### Lower Davies-Bouldin Index is better.
    ###
    ### Reverse the normalized value so that
    ### higher = better for all six metrics.

    normalized_scores[
        'DB-Index'
    ] = (
        1
        - normalized_scores[
            'DB-Index'
        ]
    )


    ### Give equal weight to all six measures

    results_df[
        'Combined'
    ] = normalized_scores.mean(
        axis=1
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

    print(
        'Number of parallel workers:',
        N_JOBS
    )

    print()


    #######################################################
    ### Coarse-grained parameter sweep
    #######################################################

    print(
        'Performing coarse-grained parameter sweep ...'
    )


    coarse_cluster_numbers = list(
        range(
            30,
            100,
            3
        )
    )


    coarse_eval_scores = evaluate_cluster_range(
        coarse_cluster_numbers
    )


    coarse_results = combine_scores(
        coarse_eval_scores
    )


    #######################################################
    ### Identify peak of coarse-grained sweep
    #######################################################

    best_coarse_position = int(
        np.argmax(
            coarse_results[
                'Combined'
            ].to_numpy()
        )
    )


    best_coarse_n = int(
        coarse_results.iloc[
            best_coarse_position
        ][
            'n_cluster'
        ]
    )


    print()

    print(
        'Best coarse-grained cluster number:',
        best_coarse_n
    )


    #######################################################
    ### Automatically determine fine-grained range
    #######################################################

    lower_position = max(
        0,
        best_coarse_position - 1
    )


    upper_position = min(
        len(coarse_results) - 1,
        best_coarse_position + 1
    )


    n1 = int(
        coarse_results.iloc[
            lower_position
        ][
            'n_cluster'
        ]
    )


    n2 = int(
        coarse_results.iloc[
            upper_position
        ][
            'n_cluster'
        ]
    )


    print(
        'Fine-grained sweep range:',
        n1,
        'to',
        n2
    )


    if (
        best_coarse_position == 0
        or
        best_coarse_position
        == len(coarse_results) - 1
    ):

        print(
            'Warning: the best coarse result is at '
            'the edge of the search range.'
        )


    #######################################################
    ### Fine-grained parameter sweep
    #######################################################

    print()

    print(
        'Performing fine-grained parameter sweep ...'
    )


    fine_cluster_numbers = list(
        range(
            n1,
            n2 + 1,
            1
        )
    )


    fine_eval_scores = evaluate_cluster_range(
        fine_cluster_numbers
    )


    fine_results = combine_scores(
        fine_eval_scores
    )


    #######################################################
    ### Tabulate the results
    #######################################################

    print()

    print(
        'Results for K-Means:'
    )


    print(
        tabulate(
            fine_results,
            headers='keys',
            tablefmt='fancy_grid',
            showindex=False,
            floatfmt='.6f'
        )
    )


    #######################################################
    ### Select final best number of clusters
    #######################################################

    best_fine_position = int(
        np.argmax(
            fine_results[
                'Combined'
            ].to_numpy()
        )
    )


    best_n = int(
        fine_results.iloc[
            best_fine_position
        ][
            'n_cluster'
        ]
    )


    best_score = float(
        fine_results.iloc[
            best_fine_position
        ][
            'Combined'
        ]
    )


    print()

    print(
        'Best number of clusters:',
        best_n
    )


    print(
        'Combined evaluation score:',
        best_score
    )

    # K-Means
    save_result('kmeans', int(best_n))


if __name__ == '__main__':
    main()