import pandas as pd
import numpy as np
import time
import traceback
import os
import itertools
import gc
from modules.data_loader import process_dataset, import_xes
from modules.encoders import baseline, one_hot_encoding, bigram, word2vec, doc2vec, bert, acf

DATASETS = ["datasets/BPI_Challenge_2013_Incidents.xes"]
PREFIX_LENGTHS = [10, 20, 30, 40, 50, 75, 100, 125, 150] 
#PREFIX_LENGTHS = [100, 150, 200, 400, 600, 800, 1000, 1200, 1400, 1500, 2000] 
#PREFIX_LENGTHS = [100, 150, 200, 400, 600, 800, 1000, 1200, 1400, 1500, 2000, 2500]
#PREFIX_LENGTHS = [100, 150, 200, 300, 400, 500, 600, 700, 800]


K_VALUES = [3,5,10, 20]
METHODS = ['Baseline', 'OHE', 'Bigram', 'W2V', 'D2V', 'BERT', 'ACF']
STRATEGIES = ['last_k','prefix']

COMPLEXITY_FILE = "results/encoding_complexity_TEST.csv"
os.makedirs("results", exist_ok=True)

def get_matrix_stats(X):
    """Extract feature space metrics from an encoded matrix."""
    if hasattr(X, 'toarray'):  # sparse matrix (Bigram)
        nbytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
        n_features = X.shape[1]
    elif hasattr(X, 'to_numpy'):  # pandas DataFrame
        nbytes = X.to_numpy().nbytes
        n_features = X.shape[1]
    else:  # plain numpy array
        nbytes = X.nbytes
        n_features = X.shape[1]
    return n_features, nbytes

def log_result(result_dict):
    df = pd.DataFrame([result_dict])
    header = not os.path.exists(COMPLEXITY_FILE)
    df.to_csv(COMPLEXITY_FILE, mode='a', header=header, index=False)
    print(f"✓ Saved: {result_dict['method']} | "
          f"features={result_dict.get('n_features')} | "
          f"encode={result_dict.get('encode_time_per_trace_ms'):.2f}ms/trace")

if __name__ == "__main__":

    for dataset_path in DATASETS:
        print(f"\n{'='*80}\nDataset: {dataset_path}\n{'='*80}")

        try:
            full_log = import_xes(dataset_path)
            gc.collect()
        except Exception as e:
            print(f"Critical: Failed to load dataset: {e}")
            continue

        for strategy in STRATEGIES:
            tasks = list(itertools.product(PREFIX_LENGTHS, K_VALUES)) \
                if strategy == 'last_k' else [(p, None) for p in PREFIX_LENGTHS]

            for length, k_val in tasks:
                print(f"\n  Prefix length: {length}, k: {k_val}")

                try:
                    train_df, test_df, full_train_df, full_test_df = process_dataset(
                        full_log, length, strategy=strategy, k=k_val
                    )
                    if train_df.empty or test_df.empty:
                        print("Skipping - empty DataFrame")
                        continue

                    n_train_traces = len(train_df)
                    n_test_traces = len(test_df)
                    strat = ('last' + str(k_val)) if strategy == 'last_k' else strategy

                    pretrain_times = {m: None for m in METHODS}

                    if 'W2V' in METHODS:
                        t0 = time.time()
                        w2v_model = word2vec.train_word2vec_model(train_df)
                        pretrain_times['W2V'] = time.time() - t0

                    if 'D2V' in METHODS:
                        t0 = time.time()
                        tagged_data = doc2vec.prepare_tagged_data_from_series(full_train_df)
                        d2v_model = doc2vec.train_doc2vec_model(tagged_data)
                        d2v_le = doc2vec.fit_label_encoder(train_df, test_df)
                        pretrain_times['D2V'] = time.time() - t0

                    if 'BERT' in METHODS:
                        t0 = time.time()
                        bert_vocab, inv_bert_vocab = bert.build_vocab_from_traces(full_train_df)
                        bert_encoder = bert.pretrain_bert(
                            full_train_df, bert_vocab, epochs=5,
                            batch_size=32, max_len=150, hidden_size=128
                        )
                        pretrain_times['BERT'] = time.time() - t0

                    if 'ACF' in METHODS:
                        t0 = time.time()
                        acf_alphabet = acf.build_alphabet_from_log(full_train_df)
                        acf_embeddings, dist_mat = acf.train_acf_embeddings(
                            full_train_df, acf_alphabet
                        )
                        pretrain_times['ACF'] = time.time() - t0

                    for method in METHODS:
                        print(f"    → {method}")
                        try:
                            t0 = time.time()

                            if method == 'Baseline':
                                X_train, X_test = baseline.prepare_data_for_prediction(train_df, test_df)
                            elif method == 'OHE':
                                X_train, X_test = one_hot_encoding.prepare_data_for_prediction(train_df, test_df)
                            elif method == 'Bigram':
                                X_train, X_test = bigram.create_bigram_features_sparse(train_df, test_df, True)
                            elif method == 'W2V':
                                X_train, X_test, y_train, y_test, model = word2vec.prepare_word2vec_features(train_df, test_df)
                            elif method == 'D2V':
                                X_train, X_test, _, _ = doc2vec.prepare_doc2vec_features(
                                    d2v_model, train_df, test_df, d2v_le)
                            elif method == 'BERT':
                                X_train, X_test, _, _ = bert.prepare_bert_features(
                                    bert_encoder, train_df, test_df, bert_vocab)
                            elif method == 'ACF':
                                X_train, X_test, _, _ = acf.prepare_acf_features(
                                    acf_embeddings, train_df, test_df)

                            encode_time = time.time() - t0
                            n_features, train_nbytes = get_matrix_stats(X_train)
                            _, test_nbytes  = get_matrix_stats(X_test)

                            log_result({
                                'dataset':                   dataset_path,
                                'strategy':                  strat,
                                'length_or_k':               length,
                                'method':                    method,
                                'n_train_traces':            n_train_traces,
                                'n_test_traces':             n_test_traces,
                                'n_features':                n_features,
                                'train_matrix_bytes':        train_nbytes,
                                'test_matrix_bytes':         test_nbytes,
                                'encode_time_seconds':       round(encode_time, 4),
                                'encode_time_per_trace_ms':  round(encode_time / (n_train_traces + n_test_traces) * 1000, 4),
                                'pretrain_time_seconds':     round(pretrain_times[method], 4) if pretrain_times[method] else None,
                            })

                        except Exception as e:
                            print(f"ERROR in {method}: {e}")
                            traceback.print_exc()

                except Exception as e:
                    print(f"CRITICAL ERROR at length {length}: {e}")
                    traceback.print_exc()

    print(f"Done. Results saved to {COMPLEXITY_FILE}")