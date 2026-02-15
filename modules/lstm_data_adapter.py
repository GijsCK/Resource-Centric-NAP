"""
LSTM Data Adapter
=================

This module adapts different encoding methods to work with LSTM models.
Tax et al. originally used one-hot encoding with LSTM, which requires
reshaping the data from (n_samples, n_features) to (n_samples, max_len, feature_dim).

Key insight: 
- Random Forest expects: (n_samples, n_features) - flat features
- LSTM expects: (n_samples, max_len, feature_dim) - sequences of feature vectors

For positional encodings (OHE, Baseline), we need to reshape the data to preserve
the sequential structure.
"""

import numpy as np


def reshape_positional_encoding_for_lstm(X_encoded, n_positions, feature_dim_per_position):
    """
    Reshape positional encodings from flat to sequential format for LSTM.
    
    This handles encodings where each position in the sequence has been encoded
    separately (like positional one-hot encoding or label encoding).
    
    Parameters:
    - X_encoded: Array of shape (n_samples, n_positions * feature_dim_per_position)
                 OR (n_samples, n_positions) for label encoding
    - n_positions: Number of positions in the sequence (max_len)
    - feature_dim_per_position: Features per position (vocab_size for OHE, 1 for label encoding)
    
    Returns:
    - X_reshaped: Array of shape (n_samples, n_positions, feature_dim_per_position)
    
    Example:
        # One-hot encoding with 5 positions and vocab of 10 activities
        # Input shape: (1000, 50) -> 1000 samples, 5 positions * 10 features each
        # Output shape: (1000, 5, 10) -> 1000 samples, 5 timesteps, 10 features per timestep
    """
    n_samples = X_encoded.shape[0]
    
    if len(X_encoded.shape) == 1:
        # Handle 1D array
        X_encoded = X_encoded.reshape(-1, 1)
    
    # Check if this is label encoding (already has correct second dimension)
    if X_encoded.shape[1] == n_positions:
        # Label encoding: (n_samples, n_positions)
        # Reshape to (n_samples, n_positions, 1) for LSTM
        X_reshaped = X_encoded.reshape(n_samples, n_positions, feature_dim_per_position)
    else:
        # One-hot or other positional encoding: (n_samples, n_positions * feature_dim)
        # Reshape to (n_samples, n_positions, feature_dim)
        X_reshaped = X_encoded.reshape(n_samples, n_positions, feature_dim_per_position)
    
    return X_reshaped


def prepare_ohe_for_lstm(X_train, X_test, n_positions):
    """
    Prepare one-hot encoded data for LSTM.
    
    Tax et al. approach: Each position is one-hot encoded separately, then
    the LSTM processes the sequence of one-hot vectors.
    
    Parameters:
    - X_train, X_test: One-hot encoded arrays from prepare_data_for_prediction
    - n_positions: Number of positions (sequence length)
    
    Returns:
    - X_train_reshaped: (n_samples, n_positions, vocab_size)
    - X_test_reshaped: (n_samples, n_positions, vocab_size)
    - vocab_size: Size of vocabulary (features per position)
    
    Example:
        X_train shape: (1000, 50)  # 1000 samples, 5 positions * 10 vocab
        After reshape: (1000, 5, 10)  # 1000 samples, 5 steps, 10 features/step
    """
    total_features = X_train.shape[1]
    vocab_size = total_features // n_positions
    
    X_train_reshaped = reshape_positional_encoding_for_lstm(X_train, n_positions, vocab_size)
    X_test_reshaped = reshape_positional_encoding_for_lstm(X_test, n_positions, vocab_size)
    
    return X_train_reshaped, X_test_reshaped, vocab_size


def prepare_baseline_for_lstm(X_train, X_test):
    """
    Prepare baseline label encoding for LSTM.
    
    Baseline already gives us sequences of integers (n_samples, n_positions).
    We just need to add a feature dimension for LSTM: (n_samples, n_positions, 1)
    
    Actually, for LSTM with embedding layer, we keep it as (n_samples, n_positions)
    and the embedding layer handles the rest.
    
    Parameters:
    - X_train, X_test: Label encoded arrays of shape (n_samples, n_positions)
    
    Returns:
    - X_train: Same shape (n_samples, n_positions) - ready for embedding
    - X_test: Same shape (n_samples, n_positions)
    - vocab_size: Maximum integer value + 1
    """
    vocab_size = int(max(X_train.max(), X_test.max())) + 1
    
    # For LSTM with embedding layer, keep as (n_samples, n_positions)
    # The embedding layer will convert each integer to a vector
    return X_train, X_test, vocab_size


def prepare_aggregated_features_for_lstm(X_train, X_test, original_train_df):
    """
    Prepare aggregated features (Bigram, W2V average, D2V, ACF) for LSTM.
    
    These methods create fixed-size feature vectors that lose sequential structure.
    To use with LSTM, we have two options:
    
    Option 1: Repeat features across timesteps (simple but loses temporal info)
    Option 2: Extract per-position features (if possible from original data)
    
    Here we implement Option 1 as a baseline.
    
    Parameters:
    - X_train, X_test: Aggregated feature arrays (n_samples, n_features)
    - original_train_df: Original dataframe with 'subtrace' column (for getting sequence length)
    
    Returns:
    - X_train_seq: (n_samples, max_len, n_features) - features repeated at each timestep
    - X_test_seq: (n_samples, max_len, n_features)
    - feature_dim: Number of features
    """
    n_features = X_train.shape[1]
    
    # Get max sequence length from original data
    max_len = max(len(trace) for trace in original_train_df['subtrace'])
    
    # Expand features to all timesteps
    # Shape: (n_samples, 1, n_features) -> (n_samples, max_len, n_features)
    X_train_expanded = X_train[:, np.newaxis, :].repeat(max_len, axis=1)
    X_test_expanded = X_test[:, np.newaxis, :].repeat(max_len, axis=1)
    
    return X_train_expanded, X_test_expanded, n_features


def prepare_w2v_sequence_for_lstm(w2v_model, train_df, test_df):
    """
    Prepare Word2Vec embeddings as sequences for LSTM (better approach).
    
    Instead of averaging embeddings, we keep per-position embeddings
    to preserve sequential structure.
    
    Parameters:
    - w2v_model: Trained Word2Vec model
    - train_df, test_df: DataFrames with 'subtrace' column
    
    Returns:
    - X_train_seq: (n_samples, max_len, embedding_dim)
    - X_test_seq: (n_samples, max_len, embedding_dim)
    - y_train, y_test: Target labels
    - embedding_dim: Dimension of W2V embeddings
    """
    embedding_dim = w2v_model.wv.vector_size
    
    # Get max sequence length
    max_len = max(
        max(len(trace) for trace in train_df['subtrace']),
        max(len(trace) for trace in test_df['subtrace'])
    )
    
    def embed_sequences(df):
        sequences = []
        labels = []
        
        for idx, row in df.iterrows():
            trace = row['subtrace']
            label = row['next_activity']
            
            # Create sequence of embeddings
            seq_embeddings = []
            for activity in trace:
                if activity in w2v_model.wv:
                    seq_embeddings.append(w2v_model.wv[activity])
                else:
                    # Use zero vector for unknown activities
                    seq_embeddings.append(np.zeros(embedding_dim))
            
            # Pad sequence to max_len
            while len(seq_embeddings) < max_len:
                seq_embeddings.append(np.zeros(embedding_dim))
            
            sequences.append(seq_embeddings[:max_len])
            labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    X_train, y_train = embed_sequences(train_df)
    X_test, y_test = embed_sequences(test_df)
    
    return X_train, X_test, y_train, y_test, embedding_dim


# ============================================================================
# UNIFIED ADAPTER FUNCTION
# ============================================================================

def adapt_encoding_for_lstm(encoding_method, X_train, X_test, y_train, y_test, 
                            train_df=None, test_df=None, **kwargs):
    """
    Universal adapter that prepares any encoding method for LSTM.
    
    Parameters:
    - encoding_method: String - 'Baseline', 'OHE', 'Bigram', 'W2V', 'D2V', 'BERT', 'ACF'
    - X_train, X_test: Encoded feature arrays
    - y_train, y_test: Target labels
    - train_df, test_df: Original dataframes (needed for some methods)
    - **kwargs: Additional parameters (e.g., w2v_model, n_positions)
    
    Returns:
    - X_train_lstm: Array ready for LSTM input
    - X_test_lstm: Array ready for LSTM input
    - y_train, y_test: Target labels (unchanged)
    - input_dim: Feature dimension for LSTM (vocab_size or embedding_dim)
    - use_embedding: Boolean - whether to use embedding layer in LSTM
    
    The return format depends on use_embedding:
    - If True: X shape is (n_samples, max_len), input_dim is vocab_size
               LSTM should use Embedding layer
    - If False: X shape is (n_samples, max_len, feature_dim), input_dim is feature_dim
                LSTM should NOT use Embedding layer (features are already vectors)
    """
    
    if encoding_method == 'Baseline':
        # Label encoding - use embedding layer
        X_train_lstm, X_test_lstm, vocab_size = prepare_baseline_for_lstm(X_train, X_test)
        return X_train_lstm, X_test_lstm, y_train, y_test, vocab_size, True
    
    elif encoding_method == 'OHE':
        # One-hot encoding - already encoded, no embedding needed
        n_positions = kwargs.get('n_positions')
        if n_positions is None:
            # Try to infer from original data
            if train_df is not None:
                n_positions = max(len(trace) for trace in train_df['subtrace'])
            else:
                raise ValueError("n_positions required for OHE reshaping")
        
        X_train_lstm, X_test_lstm, vocab_size = prepare_ohe_for_lstm(
            X_train, X_test, n_positions
        )
        # Return vocab_size as input_dim, but use_embedding=False since already one-hot
        return X_train_lstm, X_test_lstm, y_train, y_test, vocab_size, False
    
    elif encoding_method == 'Bigram':
        # Bigram features - aggregate features, repeat across timesteps
        if train_df is None:
            raise ValueError("train_df required for Bigram adaptation")
        
        X_train_lstm, X_test_lstm, feature_dim = prepare_aggregated_features_for_lstm(
            X_train, X_test, train_df
        )
        return X_train_lstm, X_test_lstm, y_train, y_test, feature_dim, False
    
    elif encoding_method == 'W2V':
        # Word2Vec - can use sequential embeddings
        w2v_model = kwargs.get('w2v_model')
        if w2v_model is not None and train_df is not None:
            # Use sequential W2V (better)
            X_train_lstm, X_test_lstm, y_train, y_test, embedding_dim = \
                prepare_w2v_sequence_for_lstm(w2v_model, train_df, test_df)
            return X_train_lstm, X_test_lstm, y_train, y_test, embedding_dim, False
        else:
            # Fall back to averaged W2V (repeat across timesteps)
            X_train_lstm, X_test_lstm, feature_dim = prepare_aggregated_features_for_lstm(
                X_train, X_test, train_df
            )
            return X_train_lstm, X_test_lstm, y_train, y_test, feature_dim, False
    
    elif encoding_method in ['D2V', 'ACF']:
        # Doc2Vec, ACF - aggregate features
        if train_df is None:
            raise ValueError(f"train_df required for {encoding_method} adaptation")
        
        X_train_lstm, X_test_lstm, feature_dim = prepare_aggregated_features_for_lstm(
            X_train, X_test, train_df
        )
        return X_train_lstm, X_test_lstm, y_train, y_test, feature_dim, False
    
    elif encoding_method == 'BERT':
        # BERT - depends on implementation
        # If BERT returns sequences, use as-is
        # If BERT returns aggregated, repeat across timesteps
        if len(X_train.shape) == 3:
            # Already sequential: (n_samples, max_len, bert_dim)
            return X_train, X_test, y_train, y_test, X_train.shape[2], False
        else:
            # Aggregated: (n_samples, bert_dim)
            X_train_lstm, X_test_lstm, feature_dim = prepare_aggregated_features_for_lstm(
                X_train, X_test, train_df
            )
            return X_train_lstm, X_test_lstm, y_train, y_test, feature_dim, False
    
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("="*70)
    print("LSTM DATA ADAPTER - EXAMPLES")
    print("="*70)
    
    # Example data
    train_df = pd.DataFrame({
        'subtrace': [['A', 'B', 'C'], ['A', 'B'], ['B', 'C', 'D']],
        'next_activity': ['D', 'C', 'E']
    })
    
    # Example 1: Baseline (Label Encoding)
    print("\nExample 1: Baseline Label Encoding")
    X_train_baseline = np.array([[1, 2, 3], [1, 2, 0], [2, 3, 4]])
    X_test_baseline = np.array([[1, 2], [2, 3]])
    y_train = np.array(['D', 'C', 'E'])
    y_test = np.array(['C', 'E'])
    
    X_train_lstm, X_test_lstm, y_train_out, y_test_out, vocab_size, use_emb = \
        adapt_encoding_for_lstm('Baseline', X_train_baseline, X_test_baseline, 
                               y_train, y_test, train_df=train_df)
    
    print(f"Input shape: {X_train_baseline.shape}")
    print(f"Output shape: {X_train_lstm.shape}")
    print(f"Vocab size: {vocab_size}")
    print(f"Use embedding layer: {use_emb}")
    
    # Example 2: OHE
    print("\n" + "-"*70)
    print("Example 2: One-Hot Encoding")
    # Simulated OHE: 3 positions, vocab size 5 -> 15 features
    X_train_ohe = np.random.randint(0, 2, size=(3, 15))
    X_test_ohe = np.random.randint(0, 2, size=(2, 15))
    
    X_train_lstm, X_test_lstm, y_train_out, y_test_out, vocab_size, use_emb = \
        adapt_encoding_for_lstm('OHE', X_train_ohe, X_test_ohe, 
                               y_train, y_test, train_df=train_df, n_positions=3)
    
    print(f"Input shape: {X_train_ohe.shape}")
    print(f"Output shape: {X_train_lstm.shape}")
    print(f"Vocab size: {vocab_size}")
    print(f"Use embedding layer: {use_emb}")
    
    print("\n" + "="*70)