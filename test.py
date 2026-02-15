#S2gR

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import pm4py
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

def import_xes(file_path):
    log = pm4py.read_xes(file_path)
    event_log = pm4py.convert_to_dataframe(log)
    return event_log

event_log = import_xes("datasets/BPI_Challenge_2013_incidents.xes")
df = event_log[['case:concept:name', 'concept:name', 'org:resource', 'time:timestamp']]
df = df.sort_values(by=['org:resource', 'time:timestamp'])

def create_activity_sequences(df, prefix_length):
    sequences, next_activities, resources = [], [], []
    for resource, resource_df in df.groupby('org:resource'):
        activities = resource_df['concept:name'].values
        if len(activities) >= prefix_length + 1:
            sequences.append(activities[:prefix_length])
            next_activities.append(activities[prefix_length])
            resources.append(resource)
    sequences_df = pd.DataFrame(sequences, columns=[f"activity_{i+1}" for i in range(prefix_length)])
    sequences_df['next_activity'] = next_activities
    sequences_df['org:resource'] = resources
    return sequences_df

def create_transition_and_repeat_features(sequences_df):
    unique_activities = sorted(
        set(sequences_df.drop(columns=["next_activity", "org:resource"]).values.flatten()) - {None}
    )
    all_possible_transitions = [(a, b) for a in unique_activities for b in unique_activities]

    transition_counts = []
    repeat_pattern_features = []

    for _, row in sequences_df.iterrows():
        transitions = defaultdict(int)
        activities = row.drop(labels=["next_activity", "org:resource"]).dropna().tolist()

        # Transition counts
        for i in range(len(activities) - 1):
            transitions[(activities[i], activities[i + 1])] += 1
        row_counts = {f"{a}->{b}": transitions.get((a, b), 0) for (a, b) in all_possible_transitions}
        transition_counts.append(row_counts)

        # Repeat pattern features
        current_run = 1
        run_lengths = []
        for i in range(1, len(activities)):
            if activities[i] == activities[i - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        repeat_pattern_features.append({
            "avg_run_length": np.mean(run_lengths),
            "num_runs": len(run_lengths)
        })

    transitions_df = pd.DataFrame(transition_counts)
    repeat_df = pd.DataFrame(repeat_pattern_features)
    return pd.concat([sequences_df.reset_index(drop=True), transitions_df, repeat_df], axis=1)

def oversample_proportional(X, y):
    counts = y.value_counts()
    max_count = counts.max()
    X_resampled, y_resampled = [], []
    for cls in counts.index:
        cls_mask = (y == cls)
        X_cls, y_cls = X[cls_mask], y[cls_mask]
        n_repeat = int(np.ceil(max_count / len(y_cls)))
        X_resampled.append(pd.concat([X_cls]*n_repeat, axis=0))
        y_resampled.append(pd.concat([y_cls]*n_repeat, axis=0))
    X_bal = pd.concat(X_resampled, axis=0).reset_index(drop=True)
    y_bal = pd.concat(y_resampled, axis=0).reset_index(drop=True)
    return X_bal, y_bal

sequence_lengths = [100, 150, 200, 300, 400, 500, 600, 700, 800]

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

for prefix_length in sequence_lengths:
    print(f"\n🚀 Running Random Forest experiment with transition + repeat features: sequence length = {prefix_length}")

    # Create activity sequences
    sequences_df = create_activity_sequences(df, prefix_length)

    # Encode activities numerically
    label_encoder = LabelEncoder()
    activity_cols = [f"activity_{i+1}" for i in range(prefix_length)]
    all_activities = sequences_df[activity_cols + ['next_activity']].values.flatten()
    label_encoder.fit(all_activities)
    for col in activity_cols + ['next_activity']:
        sequences_df[col] = label_encoder.transform(sequences_df[col])

    # Add transition + repeat features
    sequences_df = create_transition_and_repeat_features(sequences_df)
    X = sequences_df.drop(columns=['next_activity', 'org:resource'])
    y = sequences_df['next_activity']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train, y_train = oversample_proportional(X_train, y_train)

    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_

    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate on test set
    y_pred = best_rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Save results
    os.makedirs("results/BPIC2019/RandomForest/S2gR", exist_ok=True)
    out_path = f"results/BPIC2019/RandomForest/S2gR/rf_seq_{prefix_length}.json"
    with open(out_path, "w") as f:
        json.dump({
            "sequence_length": prefix_length,
            "best_params": grid_search.best_params_,
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        }, f, indent=4)
    print(f"💾 Saved results to {out_path}")

print("All Random Forest experiments with transition + repeat features completed")