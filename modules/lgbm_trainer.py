import lightgbm as lgb
import numpy as np
import time
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import re

import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ============================================================================
# CONFIGURATION: LIGHTGBM PARAMETER GRID
# ============================================================================

# LightGBM typically requires tuning 'num_leaves' and 'learning_rate'
LGBM_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth' : [-1],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8, 1.0]   # Similar to max_features
}

# Base parameters
LGBM_BASE_PARAMS = {
    'random_state': 1,
    'n_jobs': 1,
    'verbose': -1,             # Suppress warnings
    'objective': 'multiclass',  # Force multiclass for activity prediction
    'device' : 'gpu',
    'gpu_platform_id' : 0,
    'gpu_device_id' : 0
}

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_evaluate_lgbm_grid_search(X_train, X_test, y_train, y_test, 
                                   param_grid=None, cv=3, scoring='f1_weighted'):
    if param_grid is None:
        param_grid = LGBM_PARAM_GRID
    
    if hasattr(X_train, 'columns'):
        X_train = X_train.rename(columns=lambda x: re.sub(r'[^\w]', '_', str(x)))
        X_test = X_test.rename(columns=lambda x: re.sub(r'[^\w]', '_', str(x)))
    
    # --- 1. Label Encoding for Target (y) ---
    if y_train.dtype == 'object' or isinstance(y_train[0], str):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        n_classes = len(le.classes_)
        
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
        # Ensure n_classes is defined even if data is already numeric
        n_classes = len(np.unique(y_train)) 

    # --- 2. Grid Search ---
    # FIX: Initialize lgbm_base ONCE, ensuring num_class is passed
    lgbm_base = lgb.LGBMClassifier(**LGBM_BASE_PARAMS, num_class=n_classes)
    
    print(f"      Running LightGBM grid search with {len(list(itertools.product(*param_grid.values())))} combinations...")
    
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=lgbm_base,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        verbose=0,
        refit=True
        # Pro-Tip: If you ever want to see the REAL error, add: error_score='raise'
    )
    
    grid_search.fit(X_train, y_train_encoded)
    grid_search_time = time.time() - start_time
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"      Grid search complete in {grid_search_time:.2f}s")
    print(f"      Best params: {best_params}")
    
    # --- 3. Evaluate ---
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1score = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    
    return accuracy, f1score, best_model, best_params, grid_search_time


def train_evaluate_lgbm_simple(X_train, X_test, y_train, y_test, lgbm_params=None):
    if lgbm_params is None:
        lgbm_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'multiclass'
        }
    
    # --- 1. Label Encoding for Target (y) ---
    if y_train.dtype == 'object' or isinstance(y_train[0], str):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        n_classes = len(le.classes_)
        
        seen_mask = np.isin(y_test, le.classes_)
        if (~seen_mask).sum() > 0:
            X_test = X_test[seen_mask]
            y_test = y_test[seen_mask]
        
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
        n_classes = len(np.unique(y_train))
    
    # --- 2. Train ---
    # FIX: Pass num_class explicitly here as well
    model = lgb.LGBMClassifier(**lgbm_params, num_class=n_classes)
    
    start_time = time.time()
    model.fit(X_train, y_train_encoded)
    train_time = time.time() - start_time
    
    # --- 3. Evaluate ---
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1score = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    
    return accuracy, f1score, model

# ============================================================================
# MAIN WRAPPER
# ============================================================================

def train_evaluate_lgbm(X_train, X_test, y_train, y_test, 
                        use_grid_search=True, **kwargs):
    """
    Wrapper function to choose between grid search and simple training.
    """
    if use_grid_search:
        return train_evaluate_lgbm_grid_search(X_train, X_test, y_train, y_test, **kwargs)
    else:
        accuracy, f1score, model = train_evaluate_lgbm_simple(X_train, X_test, y_train, y_test, **kwargs)
        return accuracy, f1score, model, {}