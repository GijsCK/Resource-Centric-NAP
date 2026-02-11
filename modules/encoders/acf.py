import pandas as pd
import numpy as np
from ACF_code import activity_context_frequency, pmi


def build_alphabet_from_log(train_log):
    """
    Extract unique activities from training log.
    
    Parameters:
    - train_log: List of lists (activity sequences)
    
    Returns:
    - alphabet: Set of unique activities
    """
    alphabet = set(act for trace in train_log for act in trace)
    return alphabet


def train_acf_embeddings(train_log, alphabet, ngram_size=5, bag_of_words=1, ppmi=1):
    """
    Generate ACF embeddings with optional PMI post-processing.
    
    Parameters:
    - train_log: List of lists (activity sequences)
    - alphabet: Set of unique activities
    - ngram_size: Size of n-grams for context window
    - bag_of_words: Whether to use bag-of-words (1) or positional (0)
    - ppmi: Whether to use Positive PMI (1) or regular PMI (0)
    
    Returns:
    - embeddings: Dictionary mapping activity -> numpy vector
    - dist_mat: Distance matrix between activities (optional, for analysis)
    - metadata: Dictionary with frequency info (for debugging/analysis)
    """
    print(f"Generating ACF matrices for {len(alphabet)} activities...")
    
    # Step 1: Generate raw ACF matrix
    dist_mat, acf_embeddings, act_freq, ctx_freq, ctx_idx = \
        activity_context_frequency.get_activity_context_frequency_matrix(
            log=train_log,
            alphabet=alphabet,
            ngram_size=ngram_size,
            bag_of_words=bag_of_words
        )
    
    # Step 2: Apply PMI post-processing
    pmi_dist_mat, pmi_embeddings = pmi.get_activity_context_frequency_matrix_pmi(
        embeddings=acf_embeddings,
        activity_freq_dict=act_freq,
        context_freq_dict=ctx_freq,
        context_index=ctx_idx,
        ppmi=ppmi
    )
    
    print("ACF embeddings generated successfully")
    
    # Package metadata for potential debugging
    metadata = {
        'activity_freq': act_freq,
        'context_freq': ctx_freq,
        'context_index': ctx_idx
    }
    
    return pmi_embeddings, pmi_dist_mat, metadata


def vectorize_sequences(sequences, embeddings, method='average'):
    """
    Convert activity sequences into feature vectors using pre-trained embeddings.
    
    Parameters:
    - sequences: List of lists (activity sequences / subtraces)
    - embeddings: Dictionary mapping activity -> numpy vector
    - method: 'last' (use last activity's embedding) or 
              'average' (mean of all activities in sequence)
    
    Returns:
    - feature_matrix: numpy array (n_sequences, embedding_dim)
    """
    feature_matrix = []
    
    # Get embedding dimensionality
    embedding_size = len(next(iter(embeddings.values())))
    
    for seq in sequences:
        # Filter out activities not in training vocabulary
        valid_acts = [act for act in seq if act in embeddings]
        
        if not valid_acts:
            # No valid activities -> return zero vector
            feature_matrix.append(np.zeros(embedding_size))
            continue
        
        if method == 'last':
            # Use embedding of the last valid activity
            vec = embeddings[valid_acts[-1]]
            
        elif method == 'average':
            # Average embeddings of all valid activities
            vectors = [embeddings[act] for act in valid_acts]
            vec = np.mean(vectors, axis=0)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'last' or 'average'.")
        
        feature_matrix.append(vec)
    
    return np.array(feature_matrix)

def acf_embed_data(acf_embeddings, df, method='average'):
    """
    Embed activity sequences using ACF embeddings.
    
    Parameters:
    - acf_embeddings: Dictionary mapping activity -> numpy vector
    - df: DataFrame with 'subtrace' and 'next_activity' columns
    - method: 'last' or 'average' for sequence aggregation
    
    Returns:
    - X: Feature matrix (n_samples, embedding_dim)
    - y: Target labels (next_activity values)
    """
    X = vectorize_sequences(df['subtrace'].tolist(), acf_embeddings, method=method)
    y = df['next_activity'].values
    
    return X, y

def prepare_acf_features(acf_embeddings, train_df, test_df, method='average'):
    """
    Complete pipeline: embed train and test data using ACF embeddings.
    
    Parameters:
    - acf_embeddings: Pre-trained ACF embedding dictionary
    - train_df: Training DataFrame with 'subtrace' and 'next_activity'
    - test_df: Test DataFrame with 'subtrace' and 'next_activity'
    - method: 'last' or 'average' for sequence aggregation
    
    Returns:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Target labels
    """
    X_train, y_train = acf_embed_data(acf_embeddings, train_df, method=method)
    X_test, y_test = acf_embed_data(acf_embeddings, test_df, method=method)
    
    return X_train, X_test, y_train, y_test