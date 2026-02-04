#!/usr/bin/env python



import os, sys, copy, random, time, re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from ACF_code.pmi import \
    get_activity_context_frequency_matrix_pmi, get_activity_activity_frequency_matrix_pmi
from ACF_code.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from TAX_LSTM.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
from TAX_LSTM.de_koninck_2018_act2vec.our_hyperparas import get_act2vec_distance_matrix_our
from ACF_code.activity_activity_co_occurence.activity_activity_co_occurrence import \
    get_activity_activity_co_occurence_matrix
from ACF_code.activity_contex_frequency import \
    get_activity_context_frequency_matrix
from TAX_LSTM.chiorrini_2022_embedding_process_structure.embedding_process_structure import \
    get_embedding_process_structure_distance_matrix
from TAX_LSTM.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import \
    get_context_based_distance_matrix

# -----------------------
# Set cuDNN environment variables (must be set before TensorFlow is imported)
""" 
if os.environ.get("MY_CUDNN_SET") != "true":
    os.environ[
        "LD_LIBRARY_PATH"] = "/vol/fob-vol4/mi17/kirchmah/cudnn-8.9.6/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib:" + os.environ.get(
        "LD_LIBRARY_PATH", "")
    os.environ[
        "LD_PRELOAD"] = "/vol/fob-vol4/mi17/kirchmah/cudnn-8.9.6/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn.so.8.9.6"
    os.environ["MY_CUDNN_SET"] = "true"
    os.execv(sys.executable, [sys.executable] + sys.argv)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
"""
import tensorflow as tf
# Set TensorFlow seeds to reduce randomness.
tf.compat.v1.set_random_seed(42)
tf.random.set_seed(42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# Set GPU memory growth.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limit TensorFlow to use 4096 MB on the first GPU.
        """
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=20096)])
        """
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
# Set other seeds for reproducibility.
random.seed(42)
np.random.seed(42)

# -----------------------
# User Options:
# Choose encoding_mode: "one_hot" or "embedding"
encoding_mode = "embedding"  # Change to "one_hot" for the baseline one–hot representation.
use_one_hot = (encoding_mode == "one_hot")
print("Encoding mode:", encoding_mode)

# For computed embeddings, choose the source split: "train", "validation", or "train_val"
embedding_source = "train_val"  # Options: "train", "validation", "train_val"
print("Embedding source:", embedding_source)

# -----------------------
# Evaluation parameters.
BATCH_SIZE = 32
EPOCHS = 200

# -----------------------
# Directories.
from definitions import ROOT_DIR

NA_DIR = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "next_activity_prediction")
RAW_DATASETS_DIR = os.path.join(NA_DIR, "raw_datasets")
SPLIT_DATASETS_DIR = os.path.join(NA_DIR, "split_datasets")
RESULTS_DIR = os.path.join(NA_DIR, "results_tax")
MODELS_DIR = os.path.join(NA_DIR, "models_tax")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

"""
 
        "one_hot",
    "De Koninck 2018 act2vec skip-gram",
    "De Koninck 2018 act2vec CBOW",
    "Activity-Activitiy Co Occurrence Bag Of Words",
    "Activity-Activitiy Co Occurrence N-Gram",
    "Activity-Activitiy Co Occurrence Bag Of Words PMI",
    "Activity-Activitiy Co Occurrence N-Gram PMI",
    "Activity-Activitiy Co Occurrence Bag Of Words PPMI",
    "Activity-Activitiy Co Occurrence N-Gram PPMI",
    "Activity-Context Bag Of Words",
    "Activity-Context N-Grams",
    "Activity-Context Bag Of Words PMI",
    "Activity-Context N-Grams PMI",
    "Activity-Context Bag Of Words PPMI",
    "Activity-Context N-Grams PPMI",
    "Bose 2009 Substitution Scores",
    "Chiorrini 2022 Embedding Process Structure",
    "Gamallo Fernandez 2023 Context Based"
"""

# -----------------------
# Supported Encoding Methods.
encoding_methods = [
    "one_hot",
    "Uniform Zero Embedding",
    "Random Uniform Embedding",
    "De Koninck 2018 act2vec skip-gram",
    "De Koninck 2018 act2vec CBOW",
    "Activity-Activitiy Co Occurrence Bag Of Words",
    "Activity-Activitiy Co Occurrence N-Gram",
    "Activity-Activitiy Co Occurrence Bag Of Words PMI",
    "Activity-Activitiy Co Occurrence N-Gram PMI",
    "Activity-Activitiy Co Occurrence Bag Of Words PPMI",
    "Activity-Activitiy Co Occurrence N-Gram PPMI",
    "Activity-Context Bag Of Words",
    "Activity-Context N-Grams",
    "Activity-Context Bag Of Words PMI",
    "Activity-Context N-Grams PMI",
    "Activity-Context Bag Of Words PPMI",
    "Activity-Context N-Grams PPMI",
    "Bose 2009 Substitution Scores",
    "Chiorrini 2022 Embedding Process Structure",
]

encoding_methods = [
    "one_hot"]
# Optionally, add window size variations.
from evaluation.data_util.util_activity_distances_intrinsic import add_window_size_evaluation

window_size_list = [3, 5, 9]
# For reproducibility, you can choose a single window size.

encoding_methods = add_window_size_evaluation(encoding_methods, window_size_list)
encoding_methods = ["one_hot"]

#encoding_methods = ["Gamallo Fernandez 2023 Context Based w_3"]
print("Encoding methods to evaluate:")
print(encoding_methods)

# -----------------------
# Data Loading using PM4Py.
from pm4py.objects.log.importer.xes import importer as xes_importer


def load_file_xes(file_path):
    log = xes_importer.apply(file_path)
    lines, timeseqs, timeseqs2, timeseqs3, timeseqs4, caseids = [], [], [], [], [], []
    for trace in log:
        caseid = trace.attributes.get("concept:name", "case_" + str(len(caseids)))
        caseids.append(caseid)
        tokens, times, times2, times3, times4 = [], [], [], [], []
        casestart, lastevent = None, None
        for event in trace:
            activity = event["concept:name"]
            timestamp = event["time:timestamp"]
            tokens.append(activity)
            if casestart is None:
                casestart = timestamp
                lastevent = timestamp
            diff = (timestamp - lastevent).total_seconds()
            diff2 = (timestamp - casestart).total_seconds()
            midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            diff3 = (timestamp - midnight).total_seconds()
            diff4 = timestamp.weekday()
            times.append(diff)
            times2.append(diff2)
            times3.append(diff3)
            times4.append(diff4)
            lastevent = timestamp
        tokens.append('!')
        lines.append(tokens)
        timeseqs.append(times)
        timeseqs2.append(times2)
        timeseqs3.append(times3)
        timeseqs4.append(times4)
    return lines, timeseqs, timeseqs2, timeseqs3, timeseqs4, caseids


def build_vocab(lines_full):
    vocab = sorted(set(token for trace in lines_full for token in trace))
    target_tokens = copy.copy(vocab)
    if '!' in vocab:
        vocab.remove('!')
    target_token_indices = {token: i for i, token in enumerate(target_tokens)}
    indices_token = {i: token for i, token in enumerate(target_tokens)}
    return vocab, target_tokens, target_token_indices, indices_token


def extract_window_size(s):
    match = re.search(r"w_(\d+)", s)
    return int(match.group(1)) if match else 3


def get_embeddings_for_method(method, embedding_input):
    # Remove termination tokens.
    log_input = [trace[:-1] for trace in embedding_input]
    alphabet = sorted(set(token for trace in log_input for token in trace))
    win_size = extract_window_size(method)
    if method == "one_hot":
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    elif method == "Uniform Zero Embedding":
        # Each activity gets a zero vector of dimension equal to number of unique activities.
        emb = {activity: np.zeros(len(alphabet)) for activity in alphabet}
        return emb, len(alphabet)
    elif method == "Random Uniform Embedding":
        # Each activity gets a random vector with values in the range [-10, 10]
        emb = {activity: np.random.uniform(-10, 10, size=(len(alphabet),)) for activity in alphabet}
        return emb, len(alphabet)
    elif method.startswith("Unit Distance"):
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    elif method.startswith("Bose 2009 Substitution Scores"):
        _, emb = get_substitution_and_insertion_scores(log_input, alphabet, win_size)
        return emb, len(alphabet)
    elif method.startswith("De Koninck 2018 act2vec"):
        sg = 0 if "CBOW" in method else 1
        _, emb = get_act2vec_distance_matrix(log_input, alphabet, sg, win_size)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Our act2vec"):
        emb = get_act2vec_distance_matrix_our(log_input, alphabet, win_size)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Activity-Activitiy Co Occurrence"):
        bag = True if "Bag Of Words" in method else False
        _, emb, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(log_input, alphabet,
                                                                                               win_size,
                                                                                               bag_of_words=bag)
        if "PPMI" in method:
            _, emb = get_activity_activity_frequency_matrix_pmi(emb, activity_freq_dict, activity_index, 1)
            return emb, len(next(iter(emb.values())))
        elif "PMI" in method:
            _, emb = get_activity_activity_frequency_matrix_pmi(emb, activity_freq_dict, activity_index, 0)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Activity-Context"):
        if "Bag Of Words" in method and "N-Grams" not in method:
            bag_mode = 2
        elif "N-Grams" in method:
            bag_mode = 0
        else:
            bag_mode = 2
        _, emb, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(log_input,
                                                                                                             alphabet,
                                                                                                             win_size,
                                                                                                             bag_of_words=bag_mode)
        if "PPMI" in method:
            _, emb = get_activity_context_frequency_matrix_pmi(emb, activity_freq_dict, context_freq_dict,
                                                               context_index, 1)
            return emb, len(next(iter(emb.values())))
        elif "PMI" in method:
            _, emb = get_activity_context_frequency_matrix_pmi(emb, activity_freq_dict, context_freq_dict,
                                                               context_index, 0)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Chiorrini 2022 Embedding Process Structure"):
        _, emb = get_embedding_process_structure_distance_matrix(log_input, alphabet, False)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Gamallo Fernandez 2023 Context Based"):
        _, emb = get_context_based_distance_matrix(log_input, win_size)
        return emb, len(next(iter(emb.values())))
    else:
        raise ValueError("Unknown encoding method: " + method)



# -----------------------
# Vectorization Function.
def vectorize_fold(lines, ts, ts2, ts3, ts4, divisor, divisor2, encoding_dim, extra_features=5,
                   use_one_hot=False, activity_embeddings=None, vocab=None, one_hot_indices=None):
    sentences = []
    next_tokens = []
    sent_ts, sent_ts2, sent_ts3, sent_ts4 = [], [], [], []
    for tokens, t_seq, t2_seq, t3_seq, t4_seq in zip(lines, ts, ts2, ts3, ts4):
        for i in range(1, len(tokens)):
            sentences.append(tokens[:i])
            sent_ts.append(t_seq[:i])
            sent_ts2.append(t2_seq[:i])
            sent_ts3.append(t3_seq[:i])
            sent_ts4.append(t4_seq[:i])
            next_tokens.append(tokens[i])
    num_seq = len(sentences)
    X = np.zeros((num_seq, maxlen, encoding_dim + extra_features), dtype=np.float32)
    y_act = np.zeros((num_seq, len(target_tokens)), dtype=np.float32)
    y_time = np.zeros(num_seq, dtype=np.float32)
    for i, sentence in enumerate(sentences):
        leftpad = maxlen - len(sentence)
        for t, token in enumerate(sentence):
            if use_one_hot:
                if token in one_hot_indices:
                    X[i, t + leftpad, one_hot_indices[token]] = 1
            else:
                if token in activity_embeddings:
                    X[i, t + leftpad, :encoding_dim] = activity_embeddings[token]
                else:
                    X[i, t + leftpad, :encoding_dim] = np.zeros(encoding_dim)
            # Append the additional 5 time features.
            X[i, t + leftpad, encoding_dim] = t + 1
            X[i, t + leftpad, encoding_dim + 1] = sent_ts[i][t] / divisor
            X[i, t + leftpad, encoding_dim + 2] = sent_ts2[i][t] / divisor2
            X[i, t + leftpad, encoding_dim + 3] = sent_ts3[i][t] / 86400
            X[i, t + leftpad, encoding_dim + 4] = sent_ts4[i][t] / 7
        target = next_tokens[i]
        for token in target_tokens:
            if token == target:
                y_act[i, target_token_indices[token]] = 1
        y_time[i] = sent_ts[i][-1] / divisor
    return X, y_act, y_time

raw_logs = [f for f in os.listdir(RAW_DATASETS_DIR) if f.endswith(".xes.gz")]
print(raw_logs)
results_summary = []

for raw_log in raw_logs:
    log_name = os.path.splitext(os.path.splitext(raw_log)[0])[0]  # remove ".xes.gz"
    full_path = os.path.join(RAW_DATASETS_DIR, raw_log)
    train_path = os.path.join(SPLIT_DATASETS_DIR, f"train_{log_name}.xes.gz")
    val_path = os.path.join(SPLIT_DATASETS_DIR, f"val_{log_name}.xes.gz")
    test_path = os.path.join(SPLIT_DATASETS_DIR, f"test_{log_name}.xes.gz")
    print("\n========== Processing log:", log_name, "==========")
    print("Full log:", full_path)
    print("Train:", train_path)
    print("Val:", val_path)
    print("Test:", test_path)

    # Load logs.
    lines_full_log, ts_full_log, ts2_full_log, ts3_full_log, ts4_full_log, _ = load_file_xes(full_path)
    lines_train, ts_train, ts2_train, ts3_train, ts4_train, _ = load_file_xes(train_path)
    lines_val, ts_val, ts2_val, ts3_val, ts4_val, _ = load_file_xes(val_path)
    lines_test, ts_test, ts2_test, ts3_test, ts4_test, _ = load_file_xes(test_path)

    # Build vocabulary and compute per–log parameters.
    maxlen = max(len(trace) for trace in lines_full_log)
    print("Maximum trace length for", log_name, ":", maxlen)
    all_times = [t for trace in ts_full_log for t in trace]
    divisor = np.mean(all_times) if all_times else 1
    all_times2 = [t for trace in ts2_full_log for t in trace]
    divisor2 = np.mean(all_times2) if all_times2 else 1
    vocab, target_tokens, target_token_indices, indices_token = build_vocab(lines_full_log)

    # Loop over each encoding method.
    for method in encoding_methods:
        print("\n--- Encoding method:", method, "for log:", log_name, "---")
        if method == "one_hot":
            use_one_hot = True
            encoding_dim = len(vocab)
        else:
            use_one_hot = False
            # For "Unit Distance", simply use an identity mapping.
            if method.startswith("Unit Distance"):
                activity_embeddings = {activity: np.eye(len(vocab))[i] for i, activity in enumerate(vocab)}
                encoding_dim = len(vocab)
            else:
                if embedding_source == "train":
                    embedding_input = lines_train
                elif embedding_source == "validation":
                    embedding_input = lines_val
                elif embedding_source == "train_val":
                    embedding_input = lines_train + lines_val
                else:
                    raise ValueError("Unknown embedding_source")
                try:
                    activity_embeddings, encoding_dim = get_embeddings_for_method(method, embedding_input)
                except Exception as e:
                    print(f"Error computing embeddings for method {method} on log {log_name}: {e}")
                    continue

        print("Final encoding dimension for", log_name, "with method", method, ":", encoding_dim)
        total_features = encoding_dim + 5

        # Vectorize the datasets.
        if use_one_hot:
            one_hot_indices = {token: i for i, token in enumerate(vocab)}
            X_train, y_act_train, y_time_train = vectorize_fold(lines_train, ts_train, ts2_train, ts3_train, ts4_train,
                                                                divisor, divisor2, encoding_dim,
                                                                use_one_hot=True, one_hot_indices=one_hot_indices)
            X_val, y_act_val, y_time_val = vectorize_fold(lines_val, ts_val, ts2_val, ts3_val, ts4_val,
                                                          divisor, divisor2, encoding_dim,
                                                          use_one_hot=True, one_hot_indices=one_hot_indices)
            X_test, y_act_test, y_time_test = vectorize_fold(lines_test, ts_test, ts2_test, ts3_test, ts4_test,
                                                             divisor, divisor2, encoding_dim,
                                                             use_one_hot=True, one_hot_indices=one_hot_indices)
        else:
            X_train, y_act_train, y_time_train = vectorize_fold(lines_train, ts_train, ts2_train, ts3_train, ts4_train,
                                                                divisor, divisor2, encoding_dim,
                                                                activity_embeddings=activity_embeddings)
            X_val, y_act_val, y_time_val = vectorize_fold(lines_val, ts_val, ts2_val, ts3_val, ts4_val,
                                                          divisor, divisor2, encoding_dim,
                                                          activity_embeddings=activity_embeddings)
            X_test, y_act_test, y_time_test = vectorize_fold(lines_test, ts_test, ts2_test, ts3_test, ts4_test,
                                                             divisor, divisor2, encoding_dim,
                                                             activity_embeddings=activity_embeddings)

        # -----------------------
        # Build the LSTM model.
        print("Building model for log", log_name, "with method", method)
        main_input = Input(shape=(maxlen, total_features), name='main_input')
        l1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2, unroll=True)(main_input)
        b1 = BatchNormalization()(l1)
        l2_1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2, unroll=True)(b1)
        b2_1 = BatchNormalization()(l2_1)
        l2_2 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2, unroll=True)(b1)
        b2_2 = BatchNormalization()(l2_2)
        act_output = Dense(len(target_tokens), activation='softmax', kernel_initializer='glorot_uniform',
                           name='act_output')(b2_1)
        time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)
        model = Model(inputs=[main_input], outputs=[act_output, time_output])
        opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
        model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
                      optimizer=opt,
                      metrics={"act_output": "acc", "time_output": "mae"})
        model.summary()

        # -----------------------
        # Setup callbacks and file paths.
        best_model_path = os.path.join(MODELS_DIR, f"{log_name}_{method}.h5")
        early_stopping = EarlyStopping(monitor='val_loss', patience=42)
        model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=1,
                                           save_best_only=True, save_weights_only=False)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=0.0001)

        # Train the model. Note: shuffle is disabled for reproducibility.
        print("Training model for log", log_name, "with method", method)
        model.fit(X_train, {'act_output': y_act_train, 'time_output': y_time_train},
                  validation_data=(X_val, {"act_output": y_act_val, "time_output": y_time_val}),
                  verbose=1, callbacks=[early_stopping, model_checkpoint, lr_reducer],
                  batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False)

        # Evaluate the model.
        print("Evaluating model for log", log_name, "with method", method)
        model.load_weights(best_model_path)
        model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
                      optimizer=opt,
                      metrics={"act_output": "acc", "time_output": "mae"})
        metrics_values = model.evaluate(X_test, {'act_output': y_act_test, 'time_output': y_time_test},
                                        verbose=1, batch_size=BATCH_SIZE)
        preds = model.predict([X_test])
        y_act_pred_probs = preds[0]
        y_act_pred = np.argmax(y_act_pred_probs, axis=1)
        y_act_true = np.argmax(y_act_test, axis=1)
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(y_act_true, y_act_pred)
        f1 = f1_score(y_act_true, y_act_pred, average="weighted")
        print(f"Results for log {log_name} with method {method}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

        # Save individual results.
        result_dict = {"log": log_name, "method": method, "accuracy": acc, "f1": f1}
        results_summary.append(result_dict)
        df_log = pd.DataFrame([result_dict])
        d = os.path.join(RESULTS_DIR, log_name, method)
        os.makedirs(d, exist_ok=True)
        df_log.to_csv(os.path.join(d, f"{log_name}_{method}_results.csv"), index=False)

        # Clear the model from memory before next iteration.
        tf.keras.backend.clear_session()

# -----------------------
# Aggregate overall results.
df_results = pd.DataFrame(results_summary)
df_results.to_csv(os.path.join(RESULTS_DIR, "all_logs_results.csv"), index=False)
print("\nOverall results:")
print(df_results)
avg_results = df_results.groupby("method").agg({"accuracy": "mean", "f1": "mean"}).reset_index()
print("\nAverage results per method:")
print(avg_results)
avg_results.to_csv(os.path.join(RESULTS_DIR, "average_results_per_method.csv"), index=False)

print("Pipeline evaluation finished.")