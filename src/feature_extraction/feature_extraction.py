"""
Create feature representations for the email clustering pipeline.

This file prepares the feature inputs used by the final clustering step.
It was cleaned up from the original `02_feature_representation.py`
without changing the feature-extraction logic.

Inputs expected from previous preprocessing steps:
    datasets/full_dataframe.pkl
    datasets/tranco_list.txt
    outputs/sub_topic_model.pkl
    outputs/phish_urls_df.pkl
    outputs/enron_urls_df.pkl

Outputs:
    outputs/feat_rep_e.pkl
    outputs/feat_rep_s.pkl
    outputs/feat_rep_u.pkl
    outputs/feat_rep_h.pkl
"""

# Import relevant packages
from __future__ import annotations
import os
import pickle
from pathlib import Path
from urllib.parse import urlparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import Levenshtein
import pandas as pd
from sentence_transformers import SentenceTransformer



def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def process_url(url):
    url_value = ""
    url_text = url.text

    for key, value in url.attrs.items():
        if key == "href":
            url_value = value

    url_dom = urlparse(url_value).netloc

    return url_text, url_value, url_dom


def get_count_dots(url):
    return url.count(".")


def concept_check(sub, topic):
    k = 0
    for item in topic:
        if item in sub:
            k = k + 1
    return k


def from_url_domain_match(from_dom, url_list):
    k = 0
    for url in url_list:
        if url.endswith(from_dom):
            k = k + 1
    return k


def dom_tranco_check(dom, tranco_list):
    if dom in tranco_list:
        return 1
    return 0
    


def main():
    ############# FEATURE 1: EMAIL BODY TEXT
    print("EMAIL BODY")

    full_df = pd.read_pickle("../datasets/full_dataframe.pkl").reset_index(drop=True)

    bert_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    email_texts = (
        full_df["clean_content"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    bert_vectors = bert_model.encode(email_texts)
    save_pickle(bert_vectors, "../outputs/feat_rep_e.pkl")

    ############# FEATURE 2: SUBJECT LINE
    print("SUBJECT LINE")

    sub_topics = load_pickle("../outputs/sub_topic_model.pkl")

    subject_vectors = []
    for sub in full_df.clean_subj:
        temp = [concept_check(sub, topic) for topic in sub_topics]
        subject_vectors.append(temp)

    save_pickle(subject_vectors, "../outputs/feat_rep_s.pkl")

    ############# FEATURE 3: URL FEATURES
    print("URL FEATURES")

    temp_e = pd.read_pickle("../outputs/phish_urls_df.pkl")
    temp_p = pd.read_pickle("../outputs/enron_urls_df.pkl")

    url_links = list(temp_p.url_links) + list(temp_e.url_links)

    url_df = pd.DataFrame()
    url_temp_df = pd.DataFrame()

    url_text = []
    url_link = []
    url_doms = []

    for u in url_links:
        if len(u) == 0:
            url_text.append([])
            url_link.append([])
            url_doms.append([])
        else:
            processed_urls = [process_url(t) for t in u]

            url_text.append([item[0] for item in processed_urls])
            url_link.append([item[1] for item in processed_urls])
            url_doms.append([item[2] for item in processed_urls])

    url_temp_df["text"] = url_text
    url_temp_df["link"] = url_link
    url_temp_df["domain"] = url_doms

    # Number of URLs
    url_df["url_count"] = [len(links) for links in url_links]

    # Average URL length
    url_len = []
    for u in url_link:
        if len(u) == 0:
            url_len.append(0)
        elif len(u) == 1:
            url_len.append(len(u[0]))
        else:
            temp = 0
            for t in u:
                temp = temp + len(t)
            url_len.append(temp / len(u))

    url_df["url_len_avg"] = url_len

    # Maximum number of dots in any URL
    url_dot = []
    for u in url_link:
        if len(u) == 0:
            url_dot.append(0)
        elif len(u) == 1:
            url_dot.append(get_count_dots(u[0]))
        else:
            temp = []
            for t in u:
                temp.append(get_count_dots(t))
            url_dot.append(max(temp))

    url_df["max_url_dot"] = url_dot

    # Number of unique domains
    n_uni_dom = []
    n_empty_dom = []
    uni_dom = []

    for u in url_doms:
        domains = [d for d in u if d != ""]
        unique_domains = list(set(domains))

        n_uni_dom.append(len(set(domains)))
        n_empty_dom.append(len(u) - len(set(domains)))
        uni_dom.append(unique_domains)

    url_df["number_uni_domains"] = n_uni_dom
    url_df["non_standard_domains"] = n_empty_dom
    url_temp_df["unique_domain"] = uni_dom

    # Tranco check for URL domains
    tranco_full_list = load_pickle("../datasets/tranco_list.txt")
    tranco_list = tranco_full_list[:10000]

    url_dom_tranco_check = []
    for item in uni_dom:
        k = 0
        for dom in item:
            dom_check = [dom.endswith(item) for item in tranco_list]
            k = k + sum(dom_check)
        url_dom_tranco_check.append(k)

    url_df["url_dom_tranco_check"] = url_dom_tranco_check

    save_pickle(url_df, "../outputs/feat_rep_u.pkl")

    ############# FEATURE 4: HEADER FEATURES
    print("HEADER FEATURES")

    header_df = pd.DataFrame()

    from_list = full_df.from_domain.reset_index(drop=True)
    reply_list = full_df.reply_to.reset_index(drop=True)
    return_list = full_df.return_path.reset_index(drop=True)
    url_dom_list = url_temp_df.unique_domain.reset_index(drop=True)

    # From address and URL domain match
    from_url_dom = []
    for i in range(full_df.shape[0]):
        from_dom = from_list.iloc[i]
        url_list = url_dom_list.iloc[i]
        temp = from_url_domain_match(from_dom, url_list)
        from_url_dom.append(temp)

    header_df["from_url_dom"] = from_url_dom

    # Presence of reply-to
    temp_reply = [1 for _ in reply_list]
    for i, dom in enumerate(reply_list):
        if dom == "None":
            temp_reply[i] = 0

    # Presence of return-path
    temp_return = [1 for _ in return_list]
    for i, dom in enumerate(return_list):
        if dom == "None":
            temp_return[i] = 0

    header_df["reply_to"] = temp_reply
    header_df["return_path"] = temp_return

    # From-address domain and reply-to/return-path match
    from_return_check = []
    for i in range(len(reply_list)):
        dist = Levenshtein.distance(str(from_list.iloc[i]), str(return_list.iloc[i]))
        from_return_check.append(dist)

    from_reply_check = []
    for i in range(len(reply_list)):
        dist = Levenshtein.distance(str(from_list.iloc[i]), str(reply_list.iloc[i]))
        from_reply_check.append(dist)

    header_df["from_return"] = from_return_check
    header_df["from_reply"] = from_reply_check

    # Tranco check with from address domain
    from_tranco = []
    for em in from_list:
        k = dom_tranco_check(em, tranco_list)
        from_tranco.append(k)

    header_df["from_tranco"] = from_tranco

    # Save the dataframe for future use
    header_df["attachment_count"] = full_df["attachment"]

    header_df.to_pickle("../outputs/feat_rep_h.pkl")


    pd.DataFrame(bert_vectors).to_parquet("../outputs/feat_rep_e.parquet", index=False)
    pd.DataFrame(subject_vectors).to_parquet("../outputs/feat_rep_s.parquet", index=False)

    url_df.to_parquet("../outputs/feat_rep_u.parquet", index=False)
    header_df.to_parquet("../outputs/feat_rep_h.parquet", index=False)

    print('body_vectors shape:', pd.DataFrame(bert_vectors).shape)
    print('subject_vectors shape:', pd.DataFrame(subject_vectors).shape)
    print('url_df shape:', url_df.shape)
    print('header_df shape:', header_df.shape)


if __name__ == "__main__":
    main()
