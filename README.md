# Context-Based Clustering to Mitigate Phishing Attacks

This repository contains the code for the paper:

**Context-Based Clustering to Mitigate Phishing Attacks**  
Tarini Saka, Kami E. Vaniea, and Nadin Kökciyan  
AISec 2022

[Read the paper](https://dl.acm.org/doi/10.1145/3560830.3563728)

## Overview

The project uses unsupervised clustering to group similar phishing emails. It combines semantic and contextual features extracted from:

- Email body text
- Subject lines
- URLs
- Email headers

The following clustering algorithms are evaluated:

- Agglomerative Clustering
- DBSCAN
- K-Means

Parameter sweep scripts automatically select suitable clustering parameters before the final clustering step.

## Repository Structure

```text
datasets/                  Input email datasets and processed data
outputs/                   Extracted features and selected parameters
src/feature_extraction/    Feature-extraction scripts
src/parameter_sweep/       Clustering parameter sweeps
src/clustering.py          Final clustering
src/results_analysis.py    Analysis of clustering results
clustering_results/        Generated cluster labels and results
environment.yml            Conda environment
```

## Installation

Clone the repository:

```bash
git clone https://github.com/tarinisaka-work/aisec_contextual_phishing_clustering.git
cd aisec_contextual_phishing_clustering
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate aisec_phishing_clustering
```

## Running the Pipeline

### 1. Extract features

Run these commands from the `src` directory:

```bash
cd src

python feature_extraction/extract_subject_line_model.py
python feature_extraction/extract_urls_from_emails.py
python feature_extraction/feature_extraction.py
```

### 2. Run the parameter sweeps

```bash
python parameter_sweep/agg_clustering_par_sweep.py
python parameter_sweep/db_clustering_par_sweep.py
python parameter_sweep/km_clustering_par_sweep.py
```

The selected parameters are saved in:

```text
outputs/best_clustering_parameters.json
```

### 3. Run final clustering

Return to the repository root:

```bash
cd ..
python src/clustering.py
```

The clustering labels are saved to:

```text
clustering_results/analysis_df.pkl
clustering_results/analysis_df.csv
```

### 4. Analyse the results

```bash
python src/results_analysis.py
```

## Citation

Please cite the following paper when using this code:

> T. Saka, K. E. Vaniea, and N. Kökciyan. “Context-Based Clustering to Mitigate Phishing Attacks.” Proceedings of the 15th ACM Workshop on Artificial Intelligence and Security, 2022, pp. 115–126.

DOI: [10.1145/3560830.3563728](https://doi.org/10.1145/3560830.3563728)

## Disclaimer

This repository contains phishing-email data for research purposes. Do not visit extracted URLs or open attachments from untrusted emails.
