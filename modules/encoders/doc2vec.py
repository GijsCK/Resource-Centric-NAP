import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import LabelEncoder





def prepare_tagged_data_from_series(series):
    """
    Convert a pandas Series of trace lists into TaggedDocument format for Doc2Vec.
    
    Parameters:
    - series: pandas Series where index is resource_id and values are trace lists
    
    Returns:
    - List of TaggedDocument objects
    """
    tagged_docs = []
    for resource_id, trace_list in series.items():
        tagged_docs.append(TaggedDocument(words=trace_list, tags=[str(resource_id)]))
    return tagged_docs

def train_doc2vec_model(tagged_documents, vector_size=64, window=5, min_count=1, 
                        workers=4, epochs=40):
    """
    Train a Doc2Vec model on tagged documents.
    
    Parameters:
    - tagged_documents: List of TaggedDocument objects
    - vector_size: Dimensionality of document vectors
    - window: Context window size
    - min_count: Minimum frequency for activities
    - workers: Number of parallel workers
    - epochs: Number of training epochs
    
    Returns:
    - Trained Doc2Vec model
    """
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )
    
    model.build_vocab(tagged_documents)
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model

def doc2vec_embed_data(doc2vec_model, df, infer_epochs=50):
    """
    Convert activity sequences to Doc2Vec embeddings.
    
    Parameters:
    - doc2vec_model: Trained Doc2Vec model
    - df: DataFrame with 'subtrace' and 'next_activity' columns
    - infer_epochs: Number of epochs for inference (vector inference)
    
    Returns:
    - X: Document embedding matrix (n_samples, vector_size)
    - y: Target labels (raw, not encoded)
    """
    X = np.array([
        doc2vec_model.infer_vector(subtrace, epochs=infer_epochs) 
        for subtrace in df['subtrace']
    ])
    
    y = df['next_activity'].values
    
    return X, y

def fit_label_encoder(train_df, test_df):
    """
    Fit a label encoder on all unique activities from train and test sets.
    
    Parameters:
    - train_df: Training DataFrame with 'next_activity' column
    - test_df: Test DataFrame with 'next_activity' column
    
    Returns:
    - Fitted LabelEncoder
    """
    le = LabelEncoder()
    all_activities = pd.concat([train_df['next_activity'], test_df['next_activity']])
    le.fit(all_activities)
    return le


def prepare_doc2vec_features(doc2vec_model, train_df, test_df, label_encoder, infer_epochs=50):
    """
    Complete pipeline: embed data and encode labels.
    
    Parameters:
    - doc2vec_model: Trained Doc2Vec model
    - train_df: Training DataFrame
    - test_df: Test DataFrame
    - label_encoder: Fitted LabelEncoder
    - infer_epochs: Number of epochs for vector inference
    
    Returns:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Encoded target labels
    """
    # Embed the sequences
    X_train, y_train_raw = doc2vec_embed_data(doc2vec_model, train_df, infer_epochs)
    X_test, y_test_raw = doc2vec_embed_data(doc2vec_model, test_df, infer_epochs)
    
    # Encode labels
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    
    return X_train, X_test, y_train, y_test