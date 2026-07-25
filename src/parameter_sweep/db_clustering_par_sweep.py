
### Import relevant packages
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
import json
import os

from joblib import Memory, Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
    v_measure_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


sys.setrecursionlimit(10000)


### Parameter-sweep settings

# Fixed DBSCAN minimum number of samples
MIN_SAMPLES = 8

# Epsilon is expressed as a multiplier of each vector's
# automatically identified k-distance elbow.
COARSE_FACTOR_MIN = 0.50
COARSE_FACTOR_MAX = 1.50
COARSE_FACTOR_STEP = 0.10

# The fine sweep searches between the neighboring coarse values.
FINE_FACTOR_STEP = 0.01

# Process up to seven vectors simultaneously.
N_JOBS = min(7, os.cpu_count() or 1)

# Cache the nearest-neighbor graphs so that they can be reused
# during the fine-grained sweep and future script executions.
memory = Memory(
    location='cache/dbscan',
    verbose=0
)


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
    labels = np.asarray(labels)

    number_of_labels = np.unique(
        labels
    ).size

    # Silhouette and Davies-Bouldin require at least
    # two labels and fewer labels than samples.
    if (
        number_of_labels < 2
        or number_of_labels >= len(labels)
    ):
        return [np.nan, np.nan]

    # The complete dataset is used.
    # No Silhouette sampling is performed.
    sls = silhouette_score(
        emb_vect,
        labels
    )

    # Store the real Davies-Bouldin Index.
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
# Probability of the major class in each cluster
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

    total = n_phish + n_benign

    if np.any(total == 0):
        raise ValueError(
            'A cluster contains no Phish or Benign emails.'
        )

    major_class = (
        np.maximum(n_phish, n_benign)
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

        total = n_phish + n_benign

        print(
            "For cluster",
            cluster,
            temp.shape[0],
            ":"
        )

        if total == 0:
            print(
                "No Phish or Benign emails "
                "are present in this cluster."
            )
            continue

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
    labels = np.asarray(
        labels
    )

    temp_indices = tagged_phish_df.index[
        tagged_phish_df['content_tag'] == tag
    ].to_numpy(dtype=int)

    cluster_dist = Counter(
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


### Check that the feature arrays have matching rows

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

number_of_samples = next(
    iter(row_counts)
)

if number_of_samples != len(full_df):
    raise ValueError(
        'The feature arrays and full_dataframe.pkl '
        'must contain the same number of rows.'
    )


### Function to combine and standardize NumPy feature arrays

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


print('input vectors defined ...')

print()
print('Vector 1: email body text')
print('Vector 2: email body text + header')
print('Vector 3: email body text + URL')
print('Vector 4: email body text + subject')
print('Vector 5: email body text + subject + URL')
print('Vector 6: email body text + subject + header')
print('Vector 7: email body text + subject + URL + header')
print()


### Store the seven vectors

input_vects = [
    vect1,
    vect2,
    vect3,
    vect4,
    vect5,
    vect6,
    vect7
]

vector_names = [
    'Vector 1',
    'Vector 2',
    'Vector 3',
    'Vector 4',
    'Vector 5',
    'Vector 6',
    'Vector 7'
]


### Create an inclusive floating-point range

def float_range(start, stop, step):
    number_of_steps = int(
        round((stop - start) / step)
    )

    values = np.linspace(
        start,
        stop,
        number_of_steps + 1
    )

    return np.round(
        values,
        10
    )


### Automatically identify the elbow of a k-distance curve

def identify_k_distance_elbow(
    input_vect,
    min_samples
):
    nearest_neighbors = NearestNeighbors(
        n_neighbors=min_samples,
        n_jobs=1
    )

    nearest_neighbors.fit(
        input_vect
    )

    distances, _ = nearest_neighbors.kneighbors(
        input_vect
    )

    # DBSCAN counts each point as part of its own neighborhood.
    # Therefore, the final column represents the distance needed
    # to include min_samples points, including the point itself.
    k_distances = np.sort(
        distances[:, -1]
    )

    minimum_distance = k_distances[0]
    maximum_distance = k_distances[-1]

    if np.isclose(
        minimum_distance,
        maximum_distance
    ):
        if maximum_distance <= 0:
            raise ValueError(
                'The k-distance values are all zero.'
            )

        return float(
            maximum_distance
        )

    normalized_distances = (
        (k_distances - minimum_distance)
        / (maximum_distance - minimum_distance)
    )

    normalized_positions = np.linspace(
        0,
        1,
        len(k_distances)
    )

    # For a sorted, increasing k-distance curve, the elbow
    # is approximated by its largest distance below the
    # straight line joining the first and final points.
    differences = (
        normalized_positions
        - normalized_distances
    )

    elbow_index = int(
        np.argmax(differences)
    )

    elbow_epsilon = float(
        k_distances[elbow_index]
    )

    # Fall back to the 90th percentile for unusual curves
    # where the detected elbow is at an extreme endpoint.
    if (
        elbow_index == 0
        or elbow_index == len(k_distances) - 1
        or elbow_epsilon <= 0
    ):
        positive_distances = k_distances[
            k_distances > 0
        ]

        if len(positive_distances) == 0:
            raise ValueError(
                'No positive k-distance values were found.'
            )

        elbow_epsilon = float(
            np.percentile(
                positive_distances,
                90
            )
        )

    return elbow_epsilon


### Build and cache a sparse neighborhood graph

@memory.cache
def prepare_dbscan_graph(
    input_vect,
    min_samples,
    maximum_epsilon_factor
):
    base_epsilon = identify_k_distance_elbow(
        input_vect,
        min_samples
    )

    maximum_epsilon = (
        base_epsilon
        * maximum_epsilon_factor
    )

    nearest_neighbors = NearestNeighbors(
        radius=maximum_epsilon,
        n_jobs=1
    )

    nearest_neighbors.fit(
        input_vect
    )

    # Passing input_vect explicitly includes each point as
    # its own zero-distance neighbor.
    neighborhood_graph = (
        nearest_neighbors.radius_neighbors_graph(
            input_vect,
            radius=maximum_epsilon,
            mode='distance',
            sort_results=True
        )
    )

    return (
        base_epsilon,
        neighborhood_graph
    )


### Replace undefined internal metrics with the worst
### result observed for that vector

def replace_invalid_internal_scores(scores):
    scores = np.asarray(
        scores,
        dtype=float
    ).copy()

    # Silhouette: higher is better.
    silhouette_values = scores[:, 4]

    valid_silhouette = np.isfinite(
        silhouette_values
    )

    if np.any(valid_silhouette):
        worst_silhouette = np.min(
            silhouette_values[valid_silhouette]
        )
    else:
        worst_silhouette = -1.0

    silhouette_values[
        ~valid_silhouette
    ] = worst_silhouette

    scores[:, 4] = silhouette_values

    # Davies-Bouldin: lower is better.
    dbi_values = scores[:, 5]

    valid_dbi = np.isfinite(
        dbi_values
    )

    if np.any(valid_dbi):
        worst_dbi = np.max(
            dbi_values[valid_dbi]
        )
    else:
        worst_dbi = 1_000_000.0

    dbi_values[
        ~valid_dbi
    ] = worst_dbi

    scores[:, 5] = dbi_values

    return scores


### Run all requested epsilon factors for one vector

def evaluate_single_vector(
    input_vect,
    epsilon_factors
):
    base_epsilon, neighborhood_graph = (
        prepare_dbscan_graph(
            input_vect,
            MIN_SAMPLES,
            COARSE_FACTOR_MAX
        )
    )

    evaluation_scores = []
    cluster_counts = []
    noise_fractions = []

    for epsilon_factor in epsilon_factors:
        epsilon = (
            base_epsilon
            * epsilon_factor
        )

        labels = DBSCAN(
            eps=epsilon,
            min_samples=MIN_SAMPLES,
            metric='precomputed',

            # Parallelism is handled across the seven vectors.
            # Keeping this at one prevents CPU oversubscription.
            n_jobs=1
        ).fit_predict(
            neighborhood_graph
        )

        eval_metrics = (
            external_eval(labels)
            + homogeneity_eval(labels)
            + internal_eval(input_vect, labels)
        )

        evaluation_scores.append(
            eval_metrics
        )

        unique_labels = set(
            labels
        )

        number_of_clusters = len(
            unique_labels
        )

        if -1 in unique_labels:
            number_of_clusters -= 1

        cluster_counts.append(
            number_of_clusters
        )

        noise_fractions.append(
            np.mean(labels == -1)
        )

    evaluation_scores = (
        replace_invalid_internal_scores(
            evaluation_scores
        )
    )

    return (
        base_epsilon,
        evaluation_scores,
        np.asarray(cluster_counts),
        np.asarray(noise_fractions)
    )


### Run all seven vectors in parallel

def evaluate_factor_range(
    epsilon_factors
):
    epsilon_factors = np.asarray(
        epsilon_factors,
        dtype=float
    )

    parallel_results = Parallel(
        n_jobs=N_JOBS,
        backend='loky',
        verbose=0
    )(
        delayed(evaluate_single_vector)(
            input_vect,
            epsilon_factors
        )
        for input_vect in input_vects
    )

    base_epsilons = np.asarray([
        result[0]
        for result in parallel_results
    ])

    # Shape:
    # seven vectors × epsilon factors × six metrics
    scores_by_vector = np.stack([
        result[1]
        for result in parallel_results
    ])

    cluster_counts_by_vector = np.stack([
        result[2]
        for result in parallel_results
    ])

    noise_fractions_by_vector = np.stack([
        result[3]
        for result in parallel_results
    ])

    # Average each evaluation measure across all seven vectors.
    average_scores = np.mean(
        scores_by_vector,
        axis=0
    )

    average_cluster_counts = np.mean(
        cluster_counts_by_vector,
        axis=0
    )

    average_noise_percentages = (
        100
        * np.mean(
            noise_fractions_by_vector,
            axis=0
        )
    )

    vector_epsilons = (
        base_epsilons[:, np.newaxis]
        * epsilon_factors[np.newaxis, :]
    )

    average_epsilons = np.mean(
        vector_epsilons,
        axis=0
    )

    results = pd.DataFrame(
        average_scores,
        columns=[
            'ARI',
            'AMI',
            'V-M',
            'Purity',
            'Silhouette',
            'DB-Index'
        ]
    )

    results['eps_factor'] = epsilon_factors
    results['mean_epsilon'] = average_epsilons
    results['mean_clusters'] = average_cluster_counts
    results['mean_noise_pct'] = average_noise_percentages

    return (
        results,
        base_epsilons
    )


### Normalize one metric between zero and one

def normalize_metric(
    values,
    higher_is_better=True
):
    values = np.asarray(
        values,
        dtype=float
    )

    minimum = np.min(
        values
    )

    maximum = np.max(
        values
    )

    if np.isclose(
        minimum,
        maximum
    ):
        normalized = np.ones(
            len(values)
        )
    else:
        normalized = (
            (values - minimum)
            / (maximum - minimum)
        )

    if not higher_is_better:
        normalized = (
            1 - normalized
        )

    return normalized


### Normalize and combine the six evaluation measures

def combine_scores(results):
    results = results.copy()

    normalized_scores = pd.DataFrame({
        'ARI': normalize_metric(
            results['ARI']
        ),
        'AMI': normalize_metric(
            results['AMI']
        ),
        'V-M': normalize_metric(
            results['V-M']
        ),
        'Purity': normalize_metric(
            results['Purity']
        ),
        'Silhouette': normalize_metric(
            results['Silhouette']
        ),

        # Lower DB-Index values are better.
        'DB-Index': normalize_metric(
            results['DB-Index'],
            higher_is_better=False
        )
    })

    # Give equal weight to all six normalized measures.
    results['Combined'] = (
        normalized_scores.mean(axis=1)
    )

    return results

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

    print(
        'Minimum samples:',
        MIN_SAMPLES
    )

    print()


    ### Coarse-grained parameter sweep

    print(
        'Performing coarse-grained parameter sweep ...'
    )

    coarse_factors = float_range(
        COARSE_FACTOR_MIN,
        COARSE_FACTOR_MAX,
        COARSE_FACTOR_STEP
    )

    coarse_results, base_epsilons = (
        evaluate_factor_range(
            coarse_factors
        )
    )

    coarse_results = combine_scores(
        coarse_results
    )


    ### Identify the peak of the coarse sweep

    best_coarse_position = int(
        np.argmax(
            coarse_results[
                'Combined'
            ].to_numpy()
        )
    )

    best_coarse_factor = float(
        coarse_results.iloc[
            best_coarse_position
        ]['eps_factor']
    )

    print()
    print(
        'Best coarse-grained epsilon factor:',
        best_coarse_factor
    )


    ### Automatically determine the fine-grained limits

    lower_position = max(
        0,
        best_coarse_position - 1
    )

    upper_position = min(
        len(coarse_results) - 1,
        best_coarse_position + 1
    )

    fine_minimum = float(
        coarse_results.iloc[
            lower_position
        ]['eps_factor']
    )

    fine_maximum = float(
        coarse_results.iloc[
            upper_position
        ]['eps_factor']
    )

    print(
        'Fine-grained epsilon-factor range:',
        fine_minimum,
        'to',
        fine_maximum
    )

    if (
        best_coarse_position == 0
        or best_coarse_position
        == len(coarse_results) - 1
    ):
        print(
            'Warning: the best coarse result is at '
            'the edge of the configured search range.'
        )


    ### Fine-grained parameter sweep

    print()
    print(
        'Performing fine-grained parameter sweep ...'
    )

    fine_factors = float_range(
        fine_minimum,
        fine_maximum,
        FINE_FACTOR_STEP
    )

    fine_results, base_epsilons = (
        evaluate_factor_range(
            fine_factors
        )
    )

    fine_results = combine_scores(
        fine_results
    )


    ### Tabulate the results

    print()
    print(
        'Results for DBSCAN:'
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


    ### Select the final best epsilon factor

    best_fine_position = int(
        np.argmax(
            fine_results[
                'Combined'
            ].to_numpy()
        )
    )

    best_factor = float(
        fine_results.iloc[
            best_fine_position
        ]['eps_factor']
    )

    best_score = float(
        fine_results.iloc[
            best_fine_position
        ]['Combined']
    )

    final_epsilons = (
        base_epsilons
        * best_factor
    )


    print()
    print(
        'Best epsilon factor:',
        best_factor
    )

    print(
        'Combined evaluation score:',
        best_score
    )

    print()
    print(
        'Final epsilon value for each vector:'
    )

    for vector_name, epsilon in zip(
        vector_names,
        final_epsilons
    ):
        print(
            vector_name + ':',
            epsilon
        )


    # DBSCAN
    save_result('dbscan_epsilon_factor', float(best_factor))


if __name__ == '__main__':
    main()