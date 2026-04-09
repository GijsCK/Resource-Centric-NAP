# Resource-Centric-NAP

## Overview

This repository implements a research pipeline for resource-centric next activity prediction using process event logs. The core question is how different sequence encoding techniques affect predictive performance on downstream classifiers.

The project evaluates 7 embedding and encoding approaches using two classifiers:
- Random Forest (`run_experiments_rf.py`)
- LightGBM (`run_experiments_lgbm.py`)

It also includes an experiment script to measure encoding complexity and a results analysis utility.

## Key Concepts

- **Resource-centric prediction**: The dataset is split by resource (`org:resource`), so resources in the test set do not appear in the training set.
- **Next activity prediction**: For each partial sequence of resource activity, predict the next activity.
- **Prefix strategies**:
  - `prefix`: fixed prefix of the first `n` activities
  - `last_k`: last `k` activities before the target point
  - `sliding_window`: windowed prefixes across the trace

## Encoding / Embedding Methods

The repository supports seven methods:

1. `Baseline` - position-wise label encoding of the prefix
2. `OHE` - one-hot encoding of positional activity features
3. `Bigram` - bigram transition count features from activity sequences
4. `W2V` - Word2Vec embeddings trained on activity prefixes
5. `D2V` - Doc2Vec embeddings trained on full resource traces
6. `BERT` - custom BERT-style encoder pre-trained on activity sequences
7. `ACF` - Activity Context Frequency embeddings with PMI post-processing

## Datasets

The repository uses BPI Challenge event logs stored in `datasets/`:
- `BPI_Challenge_2013_Incidents.xes`
- `BPI_Challenge_2017.xes`
- `BPI_Challenge_2018.xes`
- `BPI_Challenge_2019.xes`

These logs are loaded and converted to a resource-centric format containing:
- `case:concept:name`
- `concept:name`
- `org:resource`
- `time:timestamp`

## Main Scripts

### `run_experiments_rf.py`
Run Random Forest experiments across configured datasets, strategies, prefix lengths, and embedding methods.

### `run_experiments_lgbm.py`
Run LightGBM experiments across configured datasets, strategies, prefix lengths, and embedding methods.

### `run_complexity_experiments.py`
Measure encoding complexity for each method, including feature count, matrix memory size, and encoding time.

### `win_counts.py`
Analyze existing experiment CSV results and compute ranking statistics such as:
- total wins per method
- wins per dataset/model configuration
- total points using a top-7 scoring system

## Project Structure

- `modules/`
  - `data_loader.py` - dataset import, cleaning, resource split, and prefix generation
  - `rf_trainer.py` - Random Forest training and grid search utilities
  - `lgbm_trainer.py` - LightGBM training and grid search utilities
  - `encoders/` - encoding and embedding implementations
    - `baseline.py`
    - `one_hot_encoding.py`
    - `bigram.py`
    - `word2vec.py`
    - `doc2vec.py`
    - `bert.py`
    - `acf.py`
- `ACF_code/` - custom implementation of activity-context frequency / PMI embedding generation
- `datasets/` - input XES event logs
- `results/` - experiment output CSV files
- `plots/` - generated visualizations and plots
- `notebooks/` - exploratory analysis notebooks

## Requirements

The code uses the following Python packages (at least):

- pandas
- numpy
- pm4py
- scikit-learn
- gensim
- torch
- transformers
- lightgbm
- matplotlib
- seaborn

If you use the LightGBM GPU settings in `modules/lgbm_trainer.py`, ensure a compatible GPU environment is available or update the trainer configuration to use CPU.

## Usage

1. Install dependencies in your Python environment.
2. Place the target XES datasets in `datasets/`.
3. Run the desired script:

```bash
python run_experiments_rf.py
python run_experiments_lgbm.py
python run_complexity_experiments.py
python win_counts.py
```

4. Inspect generated CSVs in `results/` and plots/ for visual summaries.

## Notes

- Most experiment scripts are configurable via the constants defined at the top of each file (`DATASETS`, `PREFIX_LENGTHS`, `STRATEGIES`, `METHODS`, `USE_GRID_SEARCH`).
- The repository is designed for research experimentation.
- The `data_analysis.ipynb` and `notebooks/RandomForest.ipynb` notebooks contain additional exploratory work and visualization examples, the latter might not be completely up-to-date with the actual encoders, as this was mainly used for first experimentation.
