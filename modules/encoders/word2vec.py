import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Train word2vec
def train_word2vec_model(train_df, vector_size=16, window=5, min_count=1, sg=1):
    model = Word2Vec(
        sentences=train_df['subtrace'],
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=4,  
        epochs=10   
    )
    return model

def word2vec_embed_data(w2v_model, df, context_length=3, pad_direction='left'):
    X = []
    y = df['next_activity'].values
    vector_size = w2v_model.vector_size
    
    for trace in df['subtrace']:
        # Take the last 'context_length' activities
        current_context = trace[-context_length:] if len(trace) >= context_length else trace
        
        vectors = []
        for act in current_context:
            if act in w2v_model.wv:
                vectors.append(w2v_model.wv[act])
            else:
                # Unknown activity (not in training vocab) -> zero vector
                vectors.append(np.zeros(vector_size))
        
        # Pad if sequence is shorter than context_length
        while len(vectors) < context_length:
            if pad_direction == 'left':
                vectors.insert(0, np.zeros(vector_size))
            else:  # 'right'
                vectors.append(np.zeros(vector_size))
        
        X.append(vectors)
    
    X = np.array(X)  # Shape: (n_samples, context_length, vector_size)
    X_flat = X.reshape(X.shape[0], -1)  # Shape: (n_samples, context_length * vector_size)
    
    return X_flat, y

def prepare_word2vec_features(train_df, test_df, vector_size=16, context_length=3):
    # Train Word2Vec on training data
    w2v_model = train_word2vec_model(train_df, vector_size=vector_size)
    
    # Transform both datasets
    X_train, y_train = word2vec_embed_data(w2v_model, train_df, context_length=context_length)
    X_test, y_test = word2vec_embed_data(w2v_model, test_df, context_length=context_length)
    
    return X_train, X_test, y_train, y_test, w2v_model