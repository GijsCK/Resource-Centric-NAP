import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
import time
import itertools


# ============================================================================
# CONFIGURATION: RF PARAMETER GRID
# ============================================================================

RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# Base parameters (not part of grid search)
RF_BASE_PARAMS = {
    'random_state': 42,
    'n_jobs': -1
}


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_evaluate_rf_grid_search(X_train, X_test, y_train, y_test, 
                                   param_grid=None, cv=3, scoring='f1_weighted'):
    """
    Train Random Forest with grid search and return best model + metrics.
    Handles unseen labels in test set by filtering them out.
    
    Parameters:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Target labels (already encoded if needed)
    - param_grid: Dictionary of parameters to search (if None, uses RF_PARAM_GRID)
    - cv: Number of cross-validation folds
    - scoring: Scoring metric for grid search ('accuracy', 'f1_weighted', etc.)
    
    Returns:
    - accuracy: Test accuracy with best model
    - f1score: Test F1 score with best model
    - best_rf: Best trained Random Forest model
    - best_params: Dictionary of best parameters found
    - grid_search_time: Time spent on grid search
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID
    
    # Encode labels if they're strings
    if y_train.dtype == 'object' or isinstance(y_train[0], str):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        # Handle unseen labels in test set
        seen_mask = np.isin(y_test, le.classes_)
        n_unseen = (~seen_mask).sum()
        
        if n_unseen > 0:
            print(f"      ⚠ Filtering {n_unseen}/{len(y_test)} test samples with unseen labels")
            
            # Filter test set
            X_test = X_test[seen_mask]
            y_test = y_test[seen_mask]
            
            # Check if we have any test samples left
            if len(y_test) == 0:
                raise ValueError("No test samples remain after filtering unseen labels")
        
        y_test_encoded = le.transform(y_test)
        
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    # Create base estimator
    rf_base = RandomForestClassifier(**RF_BASE_PARAMS)
    
    # Grid search
    print(f"      Running grid search with {len(list(itertools.product(*param_grid.values())))} combinations...")
    
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    
    grid_search.fit(X_train, y_train_encoded)
    grid_search_time = time.time() - start_time
    
    # Get best model
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"      Grid search complete in {grid_search_time:.2f}s")
    print(f"      Best params: {best_params}")
    
    # Evaluate on test set
    y_pred = best_rf.predict(X_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1score = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    
    return accuracy, f1score, best_rf, best_params, grid_search_time


def train_evaluate_rf_simple(X_train, X_test, y_train, y_test, rf_params=None):
    """
    Train Random Forest with fixed parameters (no grid search).
    Handles unseen labels in test set by filtering them out.
    
    Parameters:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Target labels
    - rf_params: Dictionary of RF parameters (if None, uses defaults)
    
    Returns:
    - accuracy, f1score, rf model
    """
    if rf_params is None:
        rf_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'n_jobs': -1
        }
    
    # Encode labels if needed
    if y_train.dtype == 'object' or isinstance(y_train[0], str):
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
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    # Train model
    rf = RandomForestClassifier(**rf_params)
    
    start_time = time.time()
    rf.fit(X_train, y_train_encoded)
    train_time = time.time() - start_time
    
    # Predict
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1score = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    
    return accuracy, f1score, rf


# ============================================================================
# UTILITY: Generate parameter combinations for manual grid search
# ============================================================================

def generate_param_combinations(param_grid):
    """
    Generate all combinations of parameters from a grid.
    Useful if you want to loop manually instead of using GridSearchCV.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)
    
    return combinations


# For backward compatibility with your existing code
def train_evaluate_rf(X_train, X_test, y_train, y_test, 
                      use_grid_search=True, **kwargs):
    """
    Wrapper function that chooses between grid search and simple training.
    
    Parameters:
    - use_grid_search: If True, use GridSearchCV; if False, use simple training
    - **kwargs: Additional arguments passed to the chosen function
    """
    if use_grid_search:
        accuracy, f1score, rf, best_params, grid_time = train_evaluate_rf_grid_search(
            X_train, X_test, y_train, y_test, **kwargs
        )
        return accuracy, f1score, rf, best_params
    else:
        accuracy, f1score, rf = train_evaluate_rf_simple(
            X_train, X_test, y_train, y_test, **kwargs
        )
        return accuracy, f1score, rf, {}