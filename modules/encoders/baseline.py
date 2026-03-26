import pandas as pd
import numpy as np
from collections import defaultdict


def label_encode_data(train_df, test_df, feature_cols):
    """
    Label encode categorical features from train and test dataframes.
    Each unique value gets mapped to an integer.
    
    Parameters:
    - train_df: Training dataframe
    - test_df: Test dataframe  
    - feature_cols: List of column names to encode (e.g., ['pos_0', 'pos_1', ...])
    
    Returns:
    - X_train, X_test: Encoded feature matrices
    - label_maps: Dictionary mapping each column to its label mapping
    """
    label_maps = {}
    
    # Create label mappings from training data
    for col in feature_cols:
        unique_vals = train_df[col].unique()
        # Create mapping: value -> integer label
        label_maps[col] = {val: idx for idx, val in enumerate(sorted(unique_vals))}
    
    # Transform training data
    X_train = train_df[feature_cols].copy()
    for col in feature_cols:
        X_train[col] = X_train[col].map(label_maps[col])
    
    # Transform test data (unseen values will become NaN, then -1)
    X_test = test_df[feature_cols].copy()
    for col in feature_cols:
        X_test[col] = X_test[col].map(label_maps[col])
        # Handle unknown values with -1
        X_test[col] = X_test[col].fillna(-1).astype(int)
    
    # Convert to numpy arrays
    X_train = X_train.astype(np.int16).values
    X_test = X_test.astype(np.int16).values
    
    return X_train, X_test, label_maps


def transform_subtrace_to_columns(df):
    """
    Convert subtrace lists into separate positional columns.
    Each position in the sequence becomes a column (pos_0, pos_1, etc.)
    """
    prefix_cols = pd.DataFrame(df['subtrace'].tolist(), index=df.index)
    actual_width = prefix_cols.shape[1]
    
    prefix_cols.columns = [f'pos_{i}' for i in range(actual_width)]
    
    # Fill missing values with padding token
    prefix_cols = prefix_cols.fillna('PAD')
    
    return prefix_cols


def prepare_data_for_prediction(train_df, test_df):
    """
    Complete pipeline for preparing next activity prediction data with label encoding.
    """
    # Transform subtraces to columns
    train_features = transform_subtrace_to_columns(train_df)
    test_features = transform_subtrace_to_columns(test_df)
    
    # Get list of all position columns
    feature_cols = [col for col in train_features.columns if col.startswith('pos_')]
    
    # Label encode all position columns
    X_train, X_test, _ = label_encode_data(train_features, test_features, feature_cols)
    
    return X_train, X_test