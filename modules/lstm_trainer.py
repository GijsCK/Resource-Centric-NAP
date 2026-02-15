import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import time


# ============================================================================
# CONFIGURATION: LSTM PARAMETERS
# ============================================================================

LSTM_PARAM_GRID = {
    'lstm_units': [50, 100],
    'lstm_layers': [1, 2],
    'dropout': [0.2, 0.3],
    'learning_rate': [0.001, 0.002],
    'batch_size': [32, 64]
}

# Base parameters
LSTM_BASE_PARAMS = {
    'epochs': 100,
    'patience': 10,
    'verbose': 0
}


# ============================================================================
# DATA PREPARATION FOR LSTM
# ============================================================================

def prepare_lstm_input_from_label_encoding(X_encoded, max_len=None):
    """
    Convert label-encoded positional features to LSTM-ready sequences.
    
    Parameters:
    - X_encoded: numpy array of shape (n_samples, n_positions) with integer labels
    - max_len: Maximum sequence length (if None, uses the width of X_encoded)
    
    Returns:
    - X_sequences: numpy array of shape (n_samples, max_len) ready for LSTM
    - max_len: The sequence length used
    """
    if max_len is None:
        max_len = X_encoded.shape[1]
    
    # X_encoded is already in the right format for LSTM
    # Shape: (n_samples, sequence_length)
    X_sequences = X_encoded[:, :max_len]
    
    return X_sequences, max_len


def prepare_lstm_input_from_subtraces(df, activity_to_idx, max_len=None, pad_value=0):
    """
    Alternative: Directly convert subtraces to sequences using a vocabulary.
    
    Parameters:
    - df: DataFrame with 'subtrace' column containing lists of activities
    - activity_to_idx: Dictionary mapping activities to integer indices
    - max_len: Maximum sequence length (if None, computed from data)
    - pad_value: Value to use for padding (typically 0)
    
    Returns:
    - X_sequences: Padded sequences of shape (n_samples, max_len)
    - max_len: The sequence length used
    """
    sequences = []
    
    for subtrace in df['subtrace']:
        # Convert activities to indices
        seq = [activity_to_idx.get(act, pad_value) for act in subtrace]
        sequences.append(seq)
    
    # Find max length if not provided
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    # Pad sequences
    X_sequences = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        X_sequences[i, :length] = seq[:length]
    
    return X_sequences, max_len


# ============================================================================
# LSTM MODEL BUILDING
# ============================================================================

def build_lstm_model(vocab_size, max_len, n_classes, 
                     lstm_units=100, lstm_layers=2, dropout=0.2,
                     embedding_dim=None, use_embedding=True, input_shape=None):
    """
    Build LSTM model for next activity prediction (Tax et al. style).
    
    Parameters:
    - vocab_size: Size of activity vocabulary (max activity index + 1) OR feature dimension
    - max_len: Maximum sequence length
    - n_classes: Number of classes to predict
    - lstm_units: Number of units in LSTM layers
    - lstm_layers: Number of stacked LSTM layers (1 or 2)
    - dropout: Dropout rate
    - embedding_dim: Dimension of embedding layer (if None, uses lstm_units)
    - use_embedding: If True, use Embedding layer (for integer sequences)
                     If False, expect pre-embedded inputs (for OHE, W2V, etc.)
    - input_shape: For non-embedded inputs, shape is (max_len, feature_dim)
    
    Returns:
    - model: Compiled Keras model
    """
    if embedding_dim is None:
        embedding_dim = lstm_units
    
    model = Sequential()
    
    if use_embedding:
        # For integer-encoded sequences (Baseline, BERT with tokens)
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            mask_zero=True  # Mask padding values (0s)
        ))
    else:
        # For pre-embedded sequences (OHE, W2V, D2V, etc.)
        # Input shape: (max_len, feature_dim)
        if input_shape is None:
            input_shape = (max_len, vocab_size)
        
        # Add masking layer for padded sequences
        model.add(Masking(mask_value=0.0, input_shape=input_shape))
    
    # LSTM layers
    if lstm_layers == 1:
        model.add(LSTM(lstm_units, dropout=dropout))
    else:
        # First LSTM layer(s) return sequences
        for i in range(lstm_layers - 1):
            model.add(LSTM(lstm_units, dropout=dropout, return_sequences=True))
        # Last LSTM layer
        model.add(LSTM(lstm_units, dropout=dropout))
    
    # Output layer
    model.add(Dense(n_classes, activation='softmax'))
    
    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_evaluate_lstm(X_train, X_test, y_train, y_test,
                       vocab_size=None,
                       lstm_units=100,
                       lstm_layers=2, 
                       dropout=0.2,
                       learning_rate=0.002,
                       batch_size=64,
                       epochs=100,
                       patience=10,
                       verbose=0,
                       use_embedding=True):
    """
    Train LSTM model and evaluate on test set.
    Handles label encoding and unseen labels.
    
    Parameters:
    - X_train, X_test: Sequence data 
                       If use_embedding=True: (n_samples, max_len) with integer indices
                       If use_embedding=False: (n_samples, max_len, feature_dim) with vectors
    - y_train, y_test: Target labels
    - vocab_size: Size of vocabulary (if None, computed from X_train)
                  For use_embedding=True: max index + 1
                  For use_embedding=False: feature dimension
    - lstm_units: Number of LSTM units
    - lstm_layers: Number of stacked LSTM layers
    - dropout: Dropout rate
    - learning_rate: Learning rate for optimizer
    - batch_size: Batch size for training
    - epochs: Maximum number of epochs
    - patience: Early stopping patience
    - verbose: Verbosity level (0, 1, or 2)
    - use_embedding: If True, use Embedding layer; if False, expect pre-embedded inputs
    
    Returns:
    - accuracy: Test accuracy
    - f1score: Test F1 score
    - model: Trained Keras model
    - history: Training history
    - train_time: Training time in seconds
    """
    # Encode labels if needed
    if y_train.dtype == 'object' or isinstance(y_train[0] if len(y_train) > 0 else "", str):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        # Handle unseen labels in test set
        seen_mask = np.isin(y_test, le.classes_)
        n_unseen = (~seen_mask).sum()
        
        if n_unseen > 0:
            print(f"      ⚠ Filtering {n_unseen}/{len(y_test)} test samples with unseen labels")
            X_test = X_test[seen_mask]
            y_test = y_test[seen_mask]
            
            if len(y_test) == 0:
                raise ValueError("No test samples remain after filtering unseen labels")
        
        y_test_encoded = le.transform(y_test)
        n_classes = len(le.classes_)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
        n_classes = len(np.unique(y_train))
    
    # Determine input dimensions
    if use_embedding:
        # For integer sequences
        if vocab_size is None:
            vocab_size = int(max(X_train.max(), X_test.max())) + 1
        max_len = X_train.shape[1]
        input_shape = None
    else:
        # For pre-embedded sequences
        max_len = X_train.shape[1]
        feature_dim = X_train.shape[2]
        if vocab_size is None:
            vocab_size = feature_dim
        input_shape = (max_len, feature_dim)
    
    # Build model
    model = build_lstm_model(
        vocab_size=vocab_size,
        max_len=max_len,
        n_classes=n_classes,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        dropout=dropout,
        use_embedding=use_embedding,
        input_shape=input_shape
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    if verbose > 0:
        print(f"      Model architecture:")
        if use_embedding:
            print(f"        - Vocab size: {vocab_size}")
            print(f"        - Sequence length: {max_len}")
        else:
            print(f"        - Input shape: {input_shape}")
        print(f"        - Classes: {n_classes}")
        print(f"        - LSTM layers: {lstm_layers} x {lstm_units} units")
        print(f"        - Dropout: {dropout}")
        print(f"        - Learning rate: {learning_rate}")
        print(f"        - Use embedding: {use_embedding}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=verbose
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=verbose
        )
    ]
    
    # Train model
    print(f"      Training LSTM model...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train_encoded,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    train_time = time.time() - start_time
    
    # Evaluate on test set
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_test_encoded, y_pred_classes)
    f1score = f1_score(y_test_encoded, y_pred_classes, average='weighted', zero_division=0)
    
    if verbose > 0:
        print(f"      Training completed in {train_time:.2f}s")
        print(f"      Final epoch: {len(history.history['loss'])}/{epochs}")
        print(f"      Test Accuracy: {accuracy:.4f}")
        print(f"      Test F1: {f1score:.4f}")
    
    return accuracy, f1score, model, history, train_time


def train_evaluate_lstm_grid_search(X_train, X_test, y_train, y_test,
                                    param_grid=None,
                                    verbose=0):
    """
    Perform manual grid search for LSTM hyperparameters.
    
    Note: This is a manual grid search since Keras models don't work well
    with sklearn's GridSearchCV in this context.
    
    Parameters:
    - X_train, X_test: Sequence data
    - y_train, y_test: Target labels
    - param_grid: Dictionary of parameters to search (if None, uses LSTM_PARAM_GRID)
    - verbose: Verbosity level
    
    Returns:
    - accuracy: Best test accuracy
    - f1score: Best test F1 score
    - best_model: Best trained model
    - best_params: Dictionary of best parameters
    - grid_search_time: Total grid search time
    """
    if param_grid is None:
        param_grid = LSTM_PARAM_GRID
    
    # Generate all parameter combinations
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    all_combinations = list(product(*param_values))
    n_combinations = len(all_combinations)
    
    print(f"      Running grid search with {n_combinations} combinations...")
    
    best_accuracy = 0
    best_f1 = 0
    best_model = None
    best_params = None
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for i, combo in enumerate(all_combinations):
        params = dict(zip(param_names, combo))
        
        if verbose > 0:
            print(f"      [{i+1}/{n_combinations}] Testing: {params}")
        
        try:
            # Train with current parameters
            acc, f1, model, history, _ = train_evaluate_lstm(
                X_train, X_test, y_train, y_test,
                **params,
                **LSTM_BASE_PARAMS,
                verbose=0
            )
            
            # Get final validation loss
            val_loss = min(history.history['val_loss'])
            
            # Select best model based on validation loss (primary) and F1 (secondary)
            if val_loss < best_val_loss or (val_loss == best_val_loss and f1 > best_f1):
                best_val_loss = val_loss
                best_accuracy = acc
                best_f1 = f1
                best_model = model
                best_params = params
                
                if verbose > 0:
                    print(f"        ✓ New best! Val Loss: {val_loss:.4f}, F1: {f1:.4f}")
        
        except Exception as e:
            print(f"        ✗ Failed with error: {e}")
            continue
    
    grid_search_time = time.time() - start_time
    
    print(f"      Grid search complete in {grid_search_time:.2f}s")
    print(f"      Best params: {best_params}")
    print(f"      Best validation loss: {best_val_loss:.4f}")
    print(f"      Best test F1: {best_f1:.4f}")
    
    return best_accuracy, best_f1, best_model, best_params, grid_search_time


# ============================================================================
# CONVENIENCE WRAPPER
# ============================================================================

def train_evaluate_lstm_wrapper(X_train, X_test, y_train, y_test,
                                use_grid_search=False,
                                **kwargs):
    """
    Wrapper function that chooses between grid search and simple training.
    Matches the API of the Random Forest wrapper.
    
    Parameters:
    - use_grid_search: If True, use grid search; if False, use simple training
    - **kwargs: Additional arguments passed to the chosen function
    
    Returns:
    - accuracy, f1score, model, best_params (or empty dict if no grid search)
    """
    if use_grid_search:
        accuracy, f1score, model, best_params, grid_time = train_evaluate_lstm_grid_search(
            X_train, X_test, y_train, y_test, **kwargs
        )
        return accuracy, f1score, model, best_params
    else:
        accuracy, f1score, model, history, train_time = train_evaluate_lstm(
            X_train, X_test, y_train, y_test, **kwargs
        )
        return accuracy, f1score, model, {}


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Generate dummy data
    np.random.seed(42)
    
    n_train = 1000
    n_test = 200
    max_len = 10
    vocab_size = 20
    n_classes = 10
    
    # Random sequences (simulating label-encoded activities)
    X_train = np.random.randint(1, vocab_size, size=(n_train, max_len))
    X_test = np.random.randint(1, vocab_size, size=(n_test, max_len))
    
    # Random labels
    y_train = np.random.randint(0, n_classes, size=n_train)
    y_test = np.random.randint(0, n_classes, size=n_test)
    
    print("Example 1: Simple training")
    print("-" * 50)
    acc, f1, model, _ = train_evaluate_lstm_wrapper(
        X_train, X_test, y_train, y_test,
        use_grid_search=False,
        lstm_units=50,
        lstm_layers=1,
        dropout=0.2,
        learning_rate=0.002,
        batch_size=32,
        epochs=5,  # Small for demo
        verbose=1
    )
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    print("\n" + "="*50)
    print("Example 2: Grid search (small grid for demo)")
    print("-" * 50)
    
    small_grid = {
        'lstm_units': [32, 50],
        'dropout': [0.2, 0.3],
        'learning_rate': [0.001, 0.002],
        'batch_size': [32],
        'lstm_layers': [1]
    }
    
    acc, f1, model, best_params = train_evaluate_lstm_wrapper(
        X_train, X_test, y_train, y_test,
        use_grid_search=True,
        param_grid=small_grid,
        verbose=0
    )
    print(f"Best Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"Best params: {best_params}")