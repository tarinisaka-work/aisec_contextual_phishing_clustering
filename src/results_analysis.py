"""
Analyze clustering results.

This file was split out from `04_cluster_analysis.ipynb`.

It expects the clustering labels produced by `clustering.py`:
    clustering_results/analysis_df.pkl

It also expects the original project data files:
    datasets/full_dataframe.pkl
    datasets/tagged_phish_df.pkl
    datasets/keys_dict.pkl
    datasets/emails-phishing.mbox
"""

import collections
import mailbox
import pickle
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
    v_measure_score,
)


def internal_eval(emb_vect, labels):
    """Internal clustering evaluation metrics."""
    sls = silhouette_score(emb_vect, labels)
    dbi = 1 / davies_bouldin_score(emb_vect, labels)
    return [sls, dbi]


def external_eval(labels_pred, tagged_phish_df):
    """External clustering evaluation metrics against manually tagged phishing emails."""
    labels_true = tagged_phish_df.drop(
        tagged_phish_df[tagged_phish_df.content_tag == 0].index
    ).content_tag
    labels_pred = list(labels_pred[labels_true.index])

    e1 = adjusted_rand_score(labels_true, labels_pred)
    e2 = adjusted_mutual_info_score(labels_true, labels_pred)
    e3 = v_measure_score(labels_true, labels_pred)

    return [e1, e2, e3]


def homogeneity_eval(labels, full_df):
    """Phish/benign homogeneity: probability of the major class in each cluster."""
    df = pd.DataFrame()
    df["cluster_label"] = labels
    df["tag"] = list(full_df.Tag)

    major_class = []
    for i in set(df.cluster_label):
        temp = df[df.cluster_label == i]
        n_phish = temp[temp["tag"] == "Phish"].shape[0]
        n_benign = temp[temp["tag"] == "Benign"].shape[0]

        if n_phish >= n_benign:
            p2 = n_phish / (n_phish + n_benign)
        else:
            p2 = n_benign / (n_phish + n_benign)

        major_class.append(p2)

    score = sum(major_class) / len(set(df.cluster_label))
    return [score]


def cluster_analysis(df, labels, full_df):
    """Cluster-wise analysis of phish/benign homogeneity."""
    df["clusters"] = labels

    for i in list(set(labels)):
        t = list(df[df["clusters"] == i].index)
        temp = full_df.iloc[t]

        n_phish = temp[temp["Tag"] == "Phish"].shape[0]
        n_benign = temp[temp["Tag"] == "Benign"].shape[0]

        print("For cluster", i, temp.shape[0], ": ")
        print("The percentage of phishing emails are: ", 100 * (n_phish / (n_phish + n_benign)))
        print("The percentage of benign emails are: ", 100 * (n_benign / (n_phish + n_benign)))


def tag_analysis(tag, labels, analysis_df, tagged_phish_df):
    """Distribution of campaign tags among the clusters."""
    df = analysis_df.copy()
    df["clusters"] = labels

    temp_index = list(tagged_phish_df[tagged_phish_df.content_tag == tag].index)
    temp_df = df.iloc[temp_index]
    cluster_dist = collections.Counter(list(temp_df.clusters))

    for u in cluster_dist.items():
        print(u[1], "emails are in cluster", u[0])
    print()


def analyze_clusters(df, cluster_label):
    """
    Analyze clusters to identify dominant phishing and benign clusters and
    extract phishing emails wrongly classified into dominant benign clusters.

    Parameters:
    df (pd.DataFrame): DataFrame with columns 'message_id', 'cluster_label', and 'tag'.

    Returns:
    wrongly_classified_phish (pd.DataFrame): DataFrame of phishing emails wrongly classified into dominant benign clusters.
    """

    # Group by cluster_label and tag to count occurrences
    cluster_analysis = df.groupby([cluster_label, "tag"]).size().unstack(fill_value=0)

    # Identify clusters with dominant phishing and benign emails
    dominant_phishing_clusters = cluster_analysis[
        cluster_analysis["Phish"] > cluster_analysis["Benign"]
    ].index.tolist()
    dominant_benign_clusters = cluster_analysis[
        cluster_analysis["Benign"] > cluster_analysis["Phish"]
    ].index.tolist()

    # Extract phishing emails that were wrongly classified into dominant benign clusters
    wrongly_classified_phish = df[
        (df[cluster_label].isin(dominant_benign_clusters)) & (df["tag"] == "Phish")
    ]

    return wrongly_classified_phish  # dominant_phishing_clusters, dominant_benign_clusters


def find_phish_wrong_across_all_models(analysis_df, full_df):
    """Find phishing emails placed into dominant benign clusters by all three algorithms."""
    ag_phish_wrong = analyze_clusters(analysis_df, "agg_labels")
    db_phish_wrong = analyze_clusters(analysis_df, "db_labels")
    km_phish_wrong = analyze_clusters(analysis_df, "km_labels")

    wrong_phish = (
        ag_phish_wrong.index.intersection(db_phish_wrong.index).intersection(km_phish_wrong.index)
    )
    wrong_phish_df = full_df.iloc[wrong_phish, :]

    return wrong_phish_df


def print_wrong_phish_eml_paths(wrong_phish_df):
    """Print the output paths for wrongly clustered phishing emails from the phishing mbox."""
    ph_mbox = mailbox.mbox("datasets/emails-phishing.mbox")

    for i, msg in enumerate(ph_mbox):
        if str(msg["Message-ID"]) in list(wrong_phish_df["msg_id"]):
            path = "temp/" + msg["Message-ID"].replace("<", "").replace(">", "") + ".eml"
            print(path)
            # with open(path, "wb") as f:
            #     f.write(msg.as_bytes())


def build_scam_tagged_dataframe(tagged_phish_df, analysis_df):
    """Build the scam-tagged dataframe used for scam analysis."""
    scam_df = tagged_phish_df[tagged_phish_df["content_tag"] != 0]
    scam_tagged_df = pd.merge(
        scam_df[["msg_id", "content_tag", "Subject", "Content"]],
        analysis_df[analysis_df["msg_id"].isin(scam_df["msg_id"])],
        on="msg_id",
        how="right",
    ).drop_duplicates()

    return scam_tagged_df


def analyze_cluster_distribution(subset_df, cluster_col="cluster_label"):
    """
    Analyze the distribution of cluster labels within a subset of the DataFrame.

    Parameters:
    subset_df (pd.DataFrame): The subset of the DataFrame to analyze.
    cluster_col (str): The name of the column containing the cluster labels.

    Returns:
    pd.Series: A Series containing the counts of each cluster label in the subset.
    """
    cluster_counts = subset_df[cluster_col].value_counts()
    cluster_fractions = cluster_counts / len(subset_df)
    return cluster_fractions


def print_scam_cluster_distribution(scam_tagged_df, cluster_col):
    """Analyze each scam-tagged subset for one clustering algorithm."""
    grouped = scam_tagged_df.groupby("content_tag")
    distributions = []

    for label, subset in grouped:
        print(f"Analysis for true label '{label}':")
        distribution = analyze_cluster_distribution(subset, cluster_col)
        distributions.append(distribution)
        print(distribution)
        print()

    return distributions


def load_stopwords():
    """Load NLTK English stopwords."""
    nltk.download("stopwords")
    return stopwords.words("english")


def extract_words(content, stopwords_english):
    """Feed the extracted content from the email body."""
    content = content.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    content = content.replace("\t", " ").replace("\n", " ")
    content = content.lower()

    words = content.split()
    words = [i for i in words if len(i) > 1]

    temp = []
    for word in words:
        if word not in stopwords_english and word not in string.punctuation:
            temp.append(word)

    words = " ".join(temp)
    return words


def scam_3_analysis(scam_tagged_df):
    """Original scam 3 analysis helper."""
    stopwords_english = load_stopwords()

    temp_df = scam_tagged_df[scam_tagged_df["content_tag"] == 3]
    sub = [extract_words(sub, stopwords_english) for sub in temp_df["Subject"]]
    org_names = {
        "paypal": 0,
        "ebay": 0,
        "Bank of America": 0,
        "Chase": 0,
        "LaSalle Bank": 0,
    }

    for key in org_names.keys():
        org_names[key] = len([c for c in temp_df["Content"] if key in c])

    return sub, org_names


def print_scam_14_content(scam_tagged_df):
    """Print original content for content_tag 14."""
    temp_df = scam_tagged_df[scam_tagged_df["content_tag"] == 14]

    for text in temp_df["Content"]:
        print("###########################################################################################")
        print(text)


def main():
    full_df = pd.read_pickle("datasets/full_dataframe.pkl")
    tagged_phish_df = pd.read_pickle("datasets/tagged_phish_df.pkl")
    analysis_df = pd.read_pickle("clustering_results/analysis_df.pkl")

    # Key-based tags to analyse the distribution
    key_dict = pickle.load(open("datasets/keys_dict.pkl", "rb"))
    tags = [i + 1 for i in range(len(key_dict))]
    keys = list(key_dict.keys())

    print("loaded clustering results ...")

    wrong_phish_df = find_phish_wrong_across_all_models(analysis_df, full_df)
    print("phishing emails in dominant benign clusters across all models:", wrong_phish_df.shape[0])

    # Original notebook printed possible paths and left writing the .eml files commented out.
    print_wrong_phish_eml_paths(wrong_phish_df)

    scam_tagged_df = build_scam_tagged_dataframe(tagged_phish_df, analysis_df)

    # scam_tagged_df.to_csv("clustering_results/cluster_labels.csv")

    print("Agglomerative scam-tag distribution")
    ag_dist = print_scam_cluster_distribution(scam_tagged_df, "agg_labels")

    print("DBSCAN scam-tag distribution")
    db_dist = print_scam_cluster_distribution(scam_tagged_df, "db_labels")

    print("KMeans scam-tag distribution")
    km_dist = print_scam_cluster_distribution(scam_tagged_df, "km_labels")

    print("Number of clusters in annotated set by algorithm")
    print("Agglomerative:", len(set(scam_tagged_df["agg_labels"])))
    print("KMeans:", len(set(scam_tagged_df["km_labels"])))
    print("DBSCAN:", len(set(scam_tagged_df["db_labels"])))

    sub, org_names = scam_3_analysis(scam_tagged_df)
    print("Scam 3 organization counts:")
    print(org_names)

    # Kept from the notebook, but commented out by default because it prints full email contents.
    # print_scam_14_content(scam_tagged_df)

    return {
        "tags": tags,
        "keys": keys,
        "wrong_phish_df": wrong_phish_df,
        "scam_tagged_df": scam_tagged_df,
        "ag_dist": ag_dist,
        "db_dist": db_dist,
        "km_dist": km_dist,
        "scam_3_subject_words": sub,
        "scam_3_org_names": org_names,
    }


if __name__ == "__main__":
    main()
