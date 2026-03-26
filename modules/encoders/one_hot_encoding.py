import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV



def ohe_data(train_df, test_df, feature_cols):
    """
    One-hot encode categorical features from train and test dataframes.
    
    Parameters:
    - train_df: Training dataframe
    - test_df: Test dataframe  
    - feature_cols: List of column names to encode
    
    Returns:
    - X_train, X_test: Encoded feature matrices
    - encoder: Fitted OneHotEncoder object
    """
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    
    # Fit on training data
    encoder.fit(train_df[feature_cols])
    
    # Transform both datasets
    X_train = encoder.transform(train_df[feature_cols])
    X_test = encoder.transform(test_df[feature_cols])
    
    return X_train, X_test, encoder

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

# Complete workflow example:
def prepare_data_for_prediction(train_df, test_df):
    """
    Complete pipeline for preparing next activity prediction data.
    """
    # Step 1: Transform subtraces to columns
    train_features = transform_subtrace_to_columns(train_df)
    test_features = transform_subtrace_to_columns(test_df)
    
    # Step 2: Get list of all position columns
    feature_cols = [col for col in train_features.columns if col.startswith('pos_')]
    
    # Step 3: One-hot encode all position columns
    X_train, X_test, _ = ohe_data(train_features, test_features, feature_cols)
    
    return X_train, X_test
