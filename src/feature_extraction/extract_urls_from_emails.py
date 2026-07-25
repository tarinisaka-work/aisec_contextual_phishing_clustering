"""
Extract URLs from the Phishing and Enron email datasets.

This file creates the URL dataframes used by the feature-representation step.
It was cleaned up from the original `00_extract_urls.py` while keeping the
same URL extraction method: BeautifulSoup is used to extract HTML <a> tags.

Inputs expected from previous preprocessing steps:
    datasets/emails-phishing.mbox
    datasets/emails-enron.mbox
    datasets/full_dataframe.pkl
    datasets/phishing_dataframe.pkl
    datasets/enron_dataframe.pkl

Outputs:
    outputs/phish_urls_df.jsonl
    outputs/enron_urls_df.jsonl
    outputs/phish_urls_df.pkl
    outputs/enron_urls_df.pkl

Why JSONL too:
    Pickle stores Python-specific objects and is less portable/reliable.
    This script converts BeautifulSoup <a> tags into plain dictionaries before saving.
"""

# Import relevant packages
import mailbox
import os
import sys
import pandas as pd
from bs4 import BeautifulSoup

sys.setrecursionlimit(10000)


# Function to extract URLs from email HTML
def flatten(nested_list):
    """Flatten a list of lists."""
    return [item for sublist in nested_list for item in sublist]


def extract_url(msg):
    """Extract HTML anchor tags from a decoded email message payload."""
    html = msg.get_payload(decode=True)
    soup = BeautifulSoup(html, features="lxml")
    links = soup.find_all("a")
    return links


def extract_url2(msg):
    """Fallback URL extraction for message parts that are not decoded cleanly."""
    html = msg.get_payload()
    soup = BeautifulSoup(str(html), features="lxml")
    links = soup.find_all("a")
    return links


def extract_urls_from_message(msg):
    """Extract URL links from one email message, using the original fallback logic."""
    try:
        links = extract_url(msg)
        return links, None
    except Exception as error:
        try:
            temp = []
            html = msg.get_payload()

            if isinstance(html, list):
                for part in html:
                    temp.append(extract_url2(part))
                return flatten(temp), error

            return extract_url2(msg), error

        except Exception as fallback_error:
            return [], f"{error}; fallback failed with: {fallback_error}"   # If both extraction methods fail, keep the script running and store an empty URL list for that email instead of crashing.


def extract_url_dataframe(mbox_obj, dataset_name):
    """Extract URL links from every email in an mbox file."""
    temp_df = pd.DataFrame()
    msg_id = []
    url_links = [[] for _ in range(len(mbox_obj))]
    err_emails = []  # indices of emails that raise errors

    for i, msg in enumerate(mbox_obj):
        msg_id.append(str(msg["Message-Id"]))
        links, error = extract_urls_from_message(msg)
        url_links[i] = links

        if error is not None:
            err_emails.append((i, str(error)))

    temp_df["id"] = msg_id
    temp_df["url_links"] = url_links

    if err_emails:
        print(f"{dataset_name}: used fallback/empty URL extraction for {len(err_emails)} emails")

    return temp_df, err_emails


def align_urls_to_full_dataframe(url_df, expected_ids, dataset_name):
    """Align extracted URLs to the exact message order used in full_dataframe.pkl."""

    duplicate_count = url_df["id"].duplicated().sum()
    
    if duplicate_count > 0:
        #  Duplicate Message-Ids would make alignment ambiguous. Keeping the
        #  first copy is closest to the original dataframe-style behavior.
        print(f"{dataset_name}: found {duplicate_count} duplicate Message-Ids; keeping first occurrence")
        url_df = url_df.drop_duplicates(subset="id", keep="first")

    url_lookup = dict(zip(url_df["id"], url_df["url_links"]))

    expected_id_set = set(expected_ids)
    extracted_id_set = set(url_df["id"])

    missing_ids = [msg_id for msg_id in expected_ids if msg_id not in extracted_id_set]
    extra_ids = list(extracted_id_set.difference(expected_id_set))

    if missing_ids:
        # FIX: The original code only removed extras. This also handles missing
        #      expected emails by filling an empty URL list so row counts stay aligned.
        print(f"{dataset_name}: {len(missing_ids)} expected emails were missing from the mbox; filled with []")

    if extra_ids:
        # FIX: Equivalent to the original drop-extra-ID behavior, but without
        #      changing the intended full_df row order.
        print(f"{dataset_name}: dropped {len(extra_ids)} emails not present in full_dataframe.pkl")

    aligned_df = pd.DataFrame()
    aligned_df["id"] = expected_ids
    aligned_df["url_links"] = [url_lookup.get(msg_id, []) for msg_id in expected_ids]

    return aligned_df


def serialize_link(link):
    """ Convert a BeautifulSoup <a> tag into plain JSON-safe data. This avoids saving bs4.element.Tag objects directly."""
    return {
        "href": link.get("href"),
        "text": link.get_text(" ", strip=True),
        "html": str(link),
    }


def serialize_url_dataframe(df):
    """ Convert the url_links column from BeautifulSoup tags to JSON-safe dictionaries.
    Before:
        url_links = [<a href="...">text</a>, ...]

    After:
        url_links = [ {"href": "...", "text": "...", "html": "<a href='...'>text</a>"}, ... ]
    """
    df = df.copy()

    df["url_links"] = df["url_links"].apply(
        lambda links: [serialize_link(link) for link in links]
    )
    return df


def save_jsonl(df, output_path):
    """Save a dataframe as JSON Lines."""
    df.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def main():
    # Read in the dataframes
    full_df = pd.read_pickle("../datasets/full_dataframe.pkl")
    phish_df = pd.read_pickle("../datasets/phishing_dataframe.pkl")
    enron_df = pd.read_pickle("../datasets/enron_dataframe.pkl")

    # Read in the mbox files
    enronbox = mailbox.mbox("../datasets/emails-enron.mbox")  # path to enron mbox
    phishbox = mailbox.mbox("../datasets/emails-phishing.mbox")  # path to phishing mbox

    os.makedirs("../outputs", exist_ok=True)

    n_phish = len(phish_df)
    n_enron = len(enron_df)

    if n_phish + n_enron != len(full_df):
        raise ValueError(
            "The lengths of phishing_dataframe.pkl and enron_dataframe.pkl "
            "do not add up to the length of full_dataframe.pkl."
        )

    phish_expected_ids = [str(i) for i in full_df.iloc[:n_phish].msg_id]
    enron_expected_ids = [str(i) for i in full_df.iloc[n_phish:].msg_id]
    # FIX: Replaced the hardcoded split `full_df[:2193]` / `full_df[2193:]`
    #      with a split based on the actual phishing dataframe length.

    # PHISHING DATASET
    print("PHISHING DATASET")
    temp_p, phish_errors = extract_url_dataframe(phishbox, "Phishing")
    temp_p = align_urls_to_full_dataframe(temp_p, phish_expected_ids, "Phishing")

    # ENRON DATASET
    print("ENRON DATASET")
    temp_e, enron_errors = extract_url_dataframe(enronbox, "Enron")
    temp_e = align_urls_to_full_dataframe(temp_e, enron_expected_ids, "Enron")

    # Convert BeautifulSoup tags to JSON-safe dictionaries
    temp_p_json = serialize_url_dataframe(temp_p)
    temp_e_json = serialize_url_dataframe(temp_e)

    # Save the URL dataframes as JSONL instead of PKL
    phish_output_path = "../outputs/phish_urls_df.jsonl"   #phish_urls_df = pd.read_json("../outputs/phish_urls_df.jsonl", lines=True)
    enron_output_path = "../outputs/enron_urls_df.jsonl"   #enron_urls_df = pd.read_json("../outputs/enron_urls_df.jsonl", lines=True)
    save_jsonl(temp_p_json, phish_output_path)
    save_jsonl(temp_e_json, enron_output_path)

    # Save the URL dataframes
    temp_p.to_pickle("../outputs/phish_urls_df.pkl")
    temp_e.to_pickle("../outputs/enron_urls_df.pkl")


    print(f"Phishing fallback/error count: {len(phish_errors)}")
    print(f"Enron fallback/error count: {len(enron_errors)}")


if __name__ == "__main__":
    main()
