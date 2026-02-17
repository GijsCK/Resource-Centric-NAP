import pandas as pd
import traceback
import os
from modules.data_loader import process_dataset, import_xes
from modules.encoders import baseline, one_hot_encoding, bigram, word2vec, doc2vec, bert, acf
from modules.rf_trainer import (
    train_evaluate_rf_grid_search, 
    train_evaluate_rf_simple,
    RF_PARAM_GRID
)

# --- CONFIGURATION ---
DATASETS = ["datasets/BPI_Challenge_2017.xes"]
#PREFIX_LENGTHS = [10, 20, 30, 40, 50, 75, 100, 125, 150]
PREFIX_LENGTHS = [100, 150, 200, 400, 600, 800, 1000, 1200, 1400, 1500, 2000]
#PREFIX_LENGHTS = [100, 150, 200, 300, 400, 500, 600, 700, 800]
K_VALUES = [3, 5, 10, 20]
METHODS = ['Baseline', 'OHE', 'Bigram', 'W2V', 'D2V', 'BERT', 'ACF'] 
#METHODS = ['Baseline'] 


#STRATEGIES = ['prefix', 'sliding_window', 'last_k']
STRATEGIES = ['last_k']


# Grid search configuration
USE_GRID_SEARCH = True 
GRID_SEARCH_CV = 3   
GRID_SEARCH_SCORING = 'accuracy' 

RESULTS_FILE = "results/experiment_results_2017.csv"
os.makedirs("results", exist_ok=True)

print("Configuration loaded")
print(f"Grid Search: {'ENABLED' if USE_GRID_SEARCH else 'DISABLED'}")
if USE_GRID_SEARCH:
    import itertools
    n_combinations = len(list(itertools.product(*RF_PARAM_GRID.values())))
    print(f"Testing {n_combinations} parameter combinations per experiment")


def log_result(result_dict):
    """Appends a single result row to CSV immediately."""
    df = pd.DataFrame([result_dict])
    header = not os.path.exists(RESULTS_FILE)
    df.to_csv(RESULTS_FILE, mode='a', header=header, index=False)
    print(f"✓ Saved: {result_dict['method']} - Acc: {result_dict.get('accuracy', 'N/A'):.4f}")


if __name__ == "__main__":
    print("Starting experiment loop...")
    
    for dataset_path in DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_path}")
        print(f"{'='*80}")
        
        try:
            full_log = import_xes(dataset_path)
            print(f"Loaded {len(full_log)} traces")
        except Exception as e:
            print(f"CRITICAL: Failed to load dataset: {e}")
            traceback.print_exc()
            continue
        
        for strategy in STRATEGIES:
            print(f"\n--- Strategy: {strategy.upper()} ---")
            
            tasks = []
            
            if strategy == 'last_k':
                # For last_k, we need to test every K for every Prefix
                # This creates pairs: (10, 3), (10, 5), ..., (20, 3), etc.
                # 
                tasks = list(itertools.product(PREFIX_LENGTHS, K_VALUES))
            else:
                # For other strategies, K is not applicable (None)
                # This creates pairs: (10, None), (20, None), etc.
                tasks = [(p, None) for p in PREFIX_LENGTHS]
            
            for length, k_val in tasks:
                print(f"\n  {length}: {length}")
                
                try:
                    # 1. DATA PREPARATION

                    train_df, test_df, full_train_df, full_test_df = process_dataset(
                        full_log, length, strategy=strategy, k=k_val
                    )
                    
                    if train_df.empty or test_df.empty:
                        print(f"    ⚠ Skipping - Empty DataFrame")
                        continue
                    
                    print(f"    Data: train={len(train_df)}, test={len(test_df)}")
                    
                    # PRE-TRAIN MODELS (same as before)
                    d2v_model = None
                    d2v_le = None
                    if 'D2V' in METHODS:
                        try:
                            print("    Pre-training Doc2Vec...")
                            tagged_data = doc2vec.prepare_tagged_data_from_series(full_train_df)
                            d2v_model = doc2vec.train_doc2vec_model(tagged_data)
                            d2v_le = doc2vec.fit_label_encoder(train_df, test_df)
                        except Exception as e:
                            print(f"    ⚠ Doc2Vec pre-training failed: {e}")
                    
                    bert_encoder = None
                    bert_vocab = None
                    if 'BERT' in METHODS:
                        try:
                            print("    Pre-training BERT...")
                            bert_vocab, inv_bert_vocab = bert.build_vocab_from_traces(full_train_df)
                            bert_encoder = bert.pretrain_bert(
                                full_train_df, bert_vocab, epochs=5, 
                                batch_size=32, max_len=150, hidden_size=128
                            )
                        except Exception as e:
                            print(f"    ⚠ BERT pre-training failed: {e}")
                    
                    acf_embeddings = None
                    if 'ACF' in METHODS:
                        try:
                            print("    Pre-training ACF...")
                            acf_alphabet = acf.build_alphabet_from_log(full_train_df)
                            acf_embeddings, dist_mat, metadata = acf.train_acf_embeddings(
                                full_train_df, acf_alphabet
                            )
                        except Exception as e:
                            print(f"    ⚠ ACF pre-training failed: {e}")
                    
                    w2v_model = None
                    if 'W2V' in METHODS:
                        try:
                            print("    Pre-training Word2Vec...")
                            w2v_model = word2vec.train_word2vec_model(train_df)
                        except Exception as e:
                            print(f"    ⚠ Word2Vec pre-training failed: {e}")
                    
                    # LOOP THROUGH METHODS
                    for method in METHODS:
                        print(f"    → Method: {method}")
                        
                        try:
                            # 2. ENCODING
                            if method == 'Baseline':
                                X_train, X_test = baseline.prepare_data_for_prediction(train_df, test_df)
                                y_train = train_df['next_activity'].values
                                y_test = test_df['next_activity'].values
                            elif method == 'OHE':
                                X_train, X_test = one_hot_encoding.prepare_data_for_prediction(train_df, test_df)
                                y_train = train_df['next_activity'].values
                                y_test = test_df['next_activity'].values
                                
                            elif method == 'Bigram':
                                X_train, X_test = bigram.create_bigram_features_sparse(train_df, test_df, True)
                                y_train = train_df['next_activity'].values
                                y_test = test_df['next_activity'].values
                                
                            elif method == 'W2V':
                                if w2v_model is None:
                                    raise ValueError("Word2Vec model pre-training failed")
                                X_train, y_train = word2vec.word2vec_embed_data(w2v_model, train_df)
                                X_test, y_test = word2vec.word2vec_embed_data(w2v_model, test_df)
                                
                            elif method == 'D2V':
                                if d2v_model is None or d2v_le is None:
                                    raise ValueError("Doc2Vec model pre-training failed")
                                X_train, X_test, y_train, y_test = doc2vec.prepare_doc2vec_features(
                                    d2v_model, train_df, test_df, d2v_le
                                )
                                
                            elif method == 'BERT':
                                if bert_encoder is None or bert_vocab is None:
                                    raise ValueError("BERT model pre-training failed")
                                X_train, X_test, y_train, y_test = bert.prepare_bert_features(
                                    bert_encoder, train_df, test_df, bert_vocab
                                )
                                
                            elif method == 'ACF':
                                if acf_embeddings is None:
                                    raise ValueError("ACF embeddings pre-training failed")
                                X_train, X_test, y_train, y_test = acf.prepare_acf_features(
                                    acf_embeddings, train_df, test_df
                                )

                            # 3. TRAINING WITH GRID SEARCH
                            if USE_GRID_SEARCH:
                                accuracy, f1, rf, best_params, grid_time = train_evaluate_rf_grid_search(
                                    X_train, X_test, y_train, y_test,
                                    param_grid=RF_PARAM_GRID,
                                    cv=GRID_SEARCH_CV,
                                    scoring=GRID_SEARCH_SCORING
                                )
                            else:
                                accuracy, f1, rf = train_evaluate_rf_simple(
                                    X_train, X_test, y_train, y_test
                                )
                                best_params = {}
                                grid_time = 0
                            
                            if strategy == 'last_k':
                                strat = 'last' + str(k_val)
                            else:
                                strat = strategy


                            # 4. LOGGING
                            result = {
                                'dataset': dataset_path,
                                'strategy': strat,
                                'length_or_k': length,
                                'method': method,
                                'accuracy': accuracy,
                                'f1_score': f1,
                                'train_size': len(train_df),
                                'test_size': len(test_df),
                                'status': 'Success'
                            }
                            
                            # Add best parameters to result if grid search was used
                            if USE_GRID_SEARCH:
                                result['grid_search_time'] = grid_time
                                for param_name, param_value in best_params.items():
                                    result[f'rf_{param_name}'] = param_value
                            
                            log_result(result)

                        except Exception as e:
                            print(f"      ✗ ERROR in {method}: {str(e)}")
                            result = {
                                'dataset': dataset_path,
                                'strategy': strategy,
                                'length_or_k': length,
                                'method': method,
                                'accuracy': None,
                                'f1_score': None,
                                'train_size': len(train_df),
                                'test_size': len(test_df),
                                'status': f"Error: {str(e)[:100]}"
                            }
                            log_result(result)

                except Exception as e:
                    print(f"    ✗ CRITICAL ERROR processing length {length}: {e}")
                    traceback.print_exc()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {RESULTS_FILE}")
    print("="*80)
'''
## Key features:

1. ✅ **Grid search integration**: Uses `GridSearchCV` for hyperparameter tuning
2. ✅ **Configurable**: Toggle grid search on/off with `USE_GRID_SEARCH`
3. ✅ **Logs best params**: Saves best RF parameters to CSV
4. ✅ **Cross-validation**: Uses CV to find best parameters
5. ✅ **Backward compatible**: Can still run simple training if needed

## Your CSV will now include columns like:
'''
# dataset, strategy, length_or_k, method, accuracy, f1_score, grid_search_time, rf_n_estimators, rf_max_depth, rf_min_samples_split, ...