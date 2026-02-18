import pandas as pd
from collections import defaultdict


def create_bigram_features(df_train, df_test):

    # all activities in the train set
    train_activities = set([act for prefix in df_train['subtrace'] for act in prefix])
    unique_activities = sorted(list(train_activities))

    # all possible bigram transitions
    all_transitions = [f"{a}->{b}" for a in unique_activities for b in unique_activities]


    def _extract_counts(df):
        bigram_rows = []
        for prefix in df['subtrace']:
            counts = defaultdict(int)
            
            for i in range(len(prefix) - 1):
                key = f"{prefix[i]}->{prefix[i+1]}"
                counts[key] += 1
            
            row = [counts[t] for t in all_transitions]
            bigram_rows.append(row)
            
        return pd.DataFrame(bigram_rows, columns=all_transitions, index=df.index, dtype='uint16')

    X_train_bigram = _extract_counts(df_train)
    X_test_bigram = _extract_counts(df_test)
    
    return X_train_bigram, X_test_bigram


def create_bigram_features_sparse(df_train, df_test, include_start=False):
    """
    More efficient version - only creates features for bigrams seen in training.
    """
    
    # Find all bigrams that actually appear in training data
    observed_bigrams = set()
    
    for prefix in df_train['subtrace']:
        if include_start and len(prefix) > 0:
            observed_bigrams.add(f"START->{prefix[0]}")
        
        for i in range(len(prefix) - 1):
            observed_bigrams.add(f"{prefix[i]}->{prefix[i+1]}")
    
    all_transitions = sorted(list(observed_bigrams))
    
    def _extract_counts(df, include_start_token=False):
        bigram_rows = []
        
        for prefix in df['subtrace']:
            counts = defaultdict(int)
            
            if include_start_token and len(prefix) > 0:
                counts[f"START->{prefix[0]}"] += 1
            
            for i in range(len(prefix) - 1):
                key = f"{prefix[i]}->{prefix[i+1]}"
                counts[key] += 1
            
            row = [counts[t] for t in all_transitions]
            bigram_rows.append(row)
        
        return pd.DataFrame(bigram_rows, columns=all_transitions, index=df.index)
    
    X_train_bigram = _extract_counts(df_train, include_start)
    X_test_bigram = _extract_counts(df_test, include_start)
    
    return X_train_bigram, X_test_bigram