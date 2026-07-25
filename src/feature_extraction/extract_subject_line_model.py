"""
Create a BERT-based topic model for phishing email subject lines.

This file creates the subject-line topic model used by the feature
representation step.

Inputs expected from previous preprocessing steps:
    datasets/phishing_dataframe.pkl

Outputs:
    outputs/sub_topic_model.pkl
"""

# Import relevant packages
import os
import json
import pickle
import pandas as pd
import numpy as np
import collections
from nltk.util import ngrams
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


# Function to flatten a list
def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def main():
    os.makedirs('../outputs', exist_ok=True)

    # Read in the dataframe
    phish_df = pd.read_pickle('../datasets/phishing_dataframe.pkl')

    # Generate tri-grams vocab for the subject-lines
    tr_grams_list = [list(ngrams(item.split(), 3)) for item in phish_df.clean_subj]
    all_tr_grams = flatten(tr_grams_list)
    tr_gram_vocab = [' '.join(gram) for gram in all_tr_grams]

    # Encode the trigram vocabulary to create BERT vectors
    bert_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bert_vect_tr = bert_model.encode(tr_gram_vocab)

    # Get average embeddings for all the tri-grams
    df_t = pd.DataFrame()
    df_t['words'] = tr_gram_vocab
    df_t['embeddings'] = list(bert_vect_tr)

    tri_dict = {}
    # FIX: use dict.fromkeys(...) instead of set(...) to keep a deterministic first-seen order.
    #      This keeps the same unique trigram vocabulary but avoids random ordering across runs.
    for tri in dict.fromkeys(tr_gram_vocab):
        temp = df_t[df_t.words == tri]
        # FIX: make the intended element-wise average explicit.
        #      This should match the original np.mean(temp.embeddings) behavior, but is clearer.
        mean_tri_emb = np.mean(np.vstack(temp.embeddings.to_numpy()), axis=0)
        tri_dict[tri] = mean_tri_emb

    # Cluster the BERT vectors to get topic clusters
    n = 30  # number of topics to generate

    # FIX: provide a clear error if there are fewer unique trigrams than requested clusters.
    if len(tri_dict) < n:
        raise ValueError(
            f'Cannot create {n} subject-topic clusters from only {len(tri_dict)} unique trigrams.'
        )

    kmeans_model_tr = KMeans(
        n_clusters=n,
        init='k-means++',
        n_init=20,
        max_iter=100,
        random_state=0,
    ).fit(list(tri_dict.values()))

    labels_tri = kmeans_model_tr.labels_

    df = pd.DataFrame()
    df['words'] = list(tri_dict.keys())
    df['embeddings'] = list(tri_dict.values())
    df['labels'] = labels_tri

    # Extract top 20 keywords from each cluster based on frequency
    topic_keywords = []
    for i in set(labels_tri):
        temp = df[df.labels == i]
        topic = list(temp.words)
        topic_words = ' '.join(topic).split()
        topic_dict = collections.Counter(topic_words)
        keys = [key[0] for key in topic_dict.most_common(20)]
        topic_keywords.append(keys)

    # Save the topic model
    pickle.dump(topic_keywords, open('../outputs/sub_topic_model.pkl', 'wb'))

    with open("../outputs/sub_topic_model.json", "w", encoding="utf-8") as f:
        json.dump(topic_keywords, f, ensure_ascii=False, indent=2)

# with open("../outputs/topic_keywords.json", "r", encoding="utf-8") as f:
    #topic_keywords = json.load(f)


if __name__ == '__main__':
    main()