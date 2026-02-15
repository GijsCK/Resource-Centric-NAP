import pandas as pd
import traceback
import os
import itertools
from modules.data_loader import process_dataset, import_xes
from modules.encoders import baseline, one_hot_encoding, bigram, word2vec, doc2vec, bert, acf

# Import LSTM functions
from modules.lstm_trainer import (
    train_evaluate_lstm_grid_search,
    train_evaluate_lstm,
    LSTM_PARAM_GRID
)

# Import data adapter
from modules.lstm_data_adapter import adapt_encoding_for_lstm

# --- CONFIGURATION ---
DATASETS = ["datasets/BPI_Challenge_2013_incidents.xes"]
#PREFIX_LENGTHS = [100, 150, 200, 400, 600, 800, 1000, 1200, 1400, 1500, 2000]
PREFIX_LENGTHS = [10, 20, 30, 40, 50, 75, 100, 125, 150]
K_VALUES = [3, 5, 10, 20]
# Test all methods with LSTM!
METHODS = ['Baseline', 'OHE', 'Bigram', 'W2V', 'D2V', 'BERT', 'ACF'] 

STRATEGIES = ['prefix']

# Grid search configuration
USE_GRID_SEARCH = False  # Set to True for hyperparameter tuning (much slower)

RESULTS_FILE = "results/experiment_results_lstm_all_encodings_2017.csv"
os.makedirs("results", exist_ok=True)

print("Configuration loaded")
print(f"Grid Search: {'ENABLED' if USE_GRID_SEARCH else 'DISABLED'}")
if USE_GRID_SEARCH:
    n_combinations = len(list(itertools.product(*LSTM_PARAM_GRID.values())))
    print(f"Testing {n_combinations} parameter combinations per experiment")


def log_result(result_dict):
    """Appends a single result row to CSV immediately."""
    df = pd.DataFrame([result_dict])
    header = not os.path.exists(RESULTS_FILE)
    df.to_csv(RESULTS_FILE, mode='a', header=header, index=False)
    acc_str = f"{result_dict.get('accuracy', 0):.4f}" if result_dict.get('accuracy') is not None else 'N/A'
    print(f"✓ Saved: {result_dict['method']} - Acc: {acc_str}")


if __name__ == "__main__":
    print("Starting LSTM experiment loop (ALL ENCODINGS)...")
    
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
                tasks = list(itertools.product(PREFIX_LENGTHS, K_VALUES))
            else:
                tasks = [(p, None) for p in PREFIX_LENGTHS]
            
            for length, k_val in tasks:
                print(f"\n  Length: {length}, K: {k_val if k_val else 'N/A'}")
                
                try:
                    # 1. DATA PREPARATION
                    train_df, test_df, full_train_df, full_test_df = process_dataset(
                        full_log, length, strategy=strategy, k=k_val
                    )
                    
                    if train_df.empty or test_df.empty:
                        print(f"    ⚠ Skipping - Empty DataFrame")
                        continue
                    
                    print(f"    Data: train={len(train_df)}, test={len(test_df)}")
                    
                    # Get max sequence length for reshaping
                    max_seq_len = max(len(trace) for trace in train_df['subtrace'])
                    
                    # PRE-TRAIN MODELS
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
                            # 2. ENCODING (using original encoders)
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
                            
                            print(f"      Original encoding shape: X_train={X_train.shape}, X_test={X_test.shape}")
                            
                            # 3. ADAPT FOR LSTM (this is the key new step!)
                            X_train_lstm, X_test_lstm, y_train, y_test, input_dim, use_embedding = \
                                adapt_encoding_for_lstm(
                                    method, X_train, X_test, y_train, y_test,
                                    train_df=train_df,
                                    test_df=test_df,
                                    n_positions=max_seq_len,
                                    w2v_model=w2v_model if method == 'W2V' else None
                                )
                            
                            print(f"      LSTM-ready shape: X_train={X_train_lstm.shape}, X_test={X_test_lstm.shape}")
                            print(f"      Input dimension: {input_dim}, Use embedding: {use_embedding}")
                            
                            # 4. TRAINING WITH LSTM
                            if USE_GRID_SEARCH:
                                # Grid search (slower but finds best params)
                                # Note: Grid search function would need to be updated to handle use_embedding
                                print("      ⚠ Grid search with adapted encodings not fully implemented")
                                print("      Using simple training instead...")
                                USE_GRID_SEARCH = False
                            
                            if not USE_GRID_SEARCH:
                                # Simple training with default parameters
                                accuracy, f1, lstm_model, history, train_time = train_evaluate_lstm(
                                    X_train_lstm, X_test_lstm, y_train, y_test,
                                    vocab_size=input_dim,
                                    lstm_units=100,
                                    lstm_layers=2,
                                    dropout=0.2,
                                    learning_rate=0.002,
                                    batch_size=64,
                                    epochs=50,
                                    patience=5,
                                    verbose=0,
                                    use_embedding=use_embedding
                                )
                                best_params = {}
                                grid_time = train_time
                            
                            # Format strategy name
                            if strategy == 'last_k':
                                strat = 'last' + str(k_val)
                            else:
                                strat = strategy

                            # 5. LOGGING
                            result = {
                                'dataset': dataset_path,
                                'strategy': strat,
                                'length_or_k': length,
                                'method': method,
                                'accuracy': accuracy,
                                'f1_score': f1,
                                'train_size': len(train_df),
                                'test_size': len(test_df),
                                'input_dim': input_dim,
                                'use_embedding': use_embedding,
                                'lstm_input_shape': str(X_train_lstm.shape),
                                'status': 'Success'
                            }
                            
                            # Add training time
                            result['train_time'] = grid_time
                            
                            # Add best parameters if grid search was used
                            if USE_GRID_SEARCH and best_params:
                                for param_name, param_value in best_params.items():
                                    result[f'lstm_{param_name}'] = param_value
                            
                            log_result(result)

                        except Exception as e:
                            print(f"      ✗ ERROR in {method}: {str(e)}")
                            traceback.print_exc()
                            
                            # Format strategy name for error logging
                            if strategy == 'last_k':
                                strat = 'last' + str(k_val)
                            else:
                                strat = strategy
                            
                            result = {
                                'dataset': dataset_path,
                                'strategy': strat,
                                'length_or_k': length,
                                'method': method,
                                'accuracy': None,
                                'f1_score': None,
                                'train_size': len(train_df) if 'train_df' in locals() else 0,
                                'test_size': len(test_df) if 'test_df' in locals() else 0,
                                'input_dim': None,
                                'use_embedding': None,
                                'lstm_input_shape': None,
                                'status': f"Error: {str(e)[:100]}"
                            }
                            log_result(result)

                except Exception as e:
                    print(f"    ✗ CRITICAL ERROR processing length {length}: {e}")
                    traceback.print_exc()

    print("\n" + "="*80)
    print("LSTM EXPERIMENT COMPLETE (ALL ENCODINGS)")
    print(f"Results saved to: {RESULTS_FILE}")
    print("="*80)