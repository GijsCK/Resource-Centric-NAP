import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os

# Load all CSVs 
def load(path, sep=','):
    df = pd.read_csv(path, sep=sep)
    df = df[df['status'] == 'Success']
    df['accuracy'] = df['accuracy'].apply(lambda x: x / 1000 if x > 1 else x)
    return df

results_dir = 'results'
df1rf   = load(f'{results_dir}/experiment_results_2013_rf.csv',   sep=';')
df2rf   = load(f'{results_dir}/experiment_results_2017_rf.csv')
df3rf   = load(f'{results_dir}/experiment_results_2018_rf.csv')
df4rf   = load(f'{results_dir}/experiment_results_2019_rf.csv')
df1lgbm = load(f'{results_dir}/experiment_results_2013_lgbm.csv')
df2lgbm = load(f'{results_dir}/experiment_results_2017_lgbm.csv')
df3lgbm = load(f'{results_dir}/experiment_results_2018_lgbm.csv')
df4lgbm = load(f'{results_dir}/experiment_results_2019_lgbm.csv')

# Configs 
configs = [
    (df1rf,   'BPIC 2013', 'RF'),
    (df1lgbm, 'BPIC 2013', 'LightGBM'),
    (df2rf,   'BPIC 2017', 'RF'),
    (df2lgbm, 'BPIC 2017', 'LightGBM'),
    (df3rf,   'BPIC 2018', 'RF'),
    (df3lgbm, 'BPIC 2018', 'LightGBM'),
    (df4rf,   'BPIC 2019', 'RF'),
    (df4lgbm, 'BPIC 2019', 'LightGBM'),
]

custom_order = ['Baseline', 'OHE', 'Bigram', 'W2V', 'D2V', 'BERT', 'ACF']

# Count wins (rank 1 = win) and top-3 per prefix length per config 
all_records = []   # one row per (config × prefix_length × method × rank)

for df, dataset, model in configs:
    filtered = df[df['strategy'] == 'prefix'].copy()
    plot_data = (
        filtered
        .groupby(['length_or_k', 'method'])['accuracy']
        .mean()
        .reset_index()
    )

    for length, grp in plot_data.groupby('length_or_k'):
        # rank descending (rank 1 = highest accuracy)
        grp = grp.copy()
        grp['rank'] = grp['accuracy'].rank(ascending=False, method='min').astype(int)
        for _, row in grp.iterrows():
            all_records.append({
                'dataset': dataset,
                'model':   model,
                'config':  f'{dataset} ({model})',
                'length':  length,
                'method':  row['method'],
                'accuracy': row['accuracy'],
                'rank':    row['rank'],
            })

records = pd.DataFrame(all_records)

wins_total = (
    records[records['rank'] == 1]
    .groupby('method')
    .size()
    .reindex(custom_order, fill_value=0)
    .reset_index(name='total_wins')
)

wins_by_config = (
    records[records['rank'] == 1]
    .groupby(['config', 'method'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=custom_order, fill_value=0)
)

# Top-7 scores: rank 1 → 7 pts, rank 2 → 6 pts, … rank 7 → 1 pt
records['points'] = (8 - records['rank']).clip(lower=0)

points_total = (
    records
    .groupby('method')['points']
    .sum()
    .reindex(custom_order, fill_value=0)
    .reset_index(name='total_points')
)

# Print tables
print("=" * 60)
print("TOTAL WINS (rank #1 at a prefix length) per method")
print("=" * 60)
print(wins_total.sort_values('total_wins', ascending=False).to_string(index=False))

print("\n" + "=" * 60)
print("WINS per config per method")
print("=" * 60)
print(wins_by_config.to_string())

print("\n" + "=" * 60)
print("TOTAL POINTS (top-7 scoring) per method")
print("=" * 60)
print(points_total.sort_values('total_points', ascending=False).to_string(index=False))
