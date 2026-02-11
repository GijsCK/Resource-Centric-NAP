import pandas as pd
import pm4py
from sklearn.model_selection import train_test_split



# Importing dataset from file path
def import_xes(file_path):
    log = pm4py.read_xes(file_path)
    return pm4py.convert_to_dataframe(log)

# Cleaning dataset: removing unnecessary columns, shifting to resource focus
def clean_dataset(df):
    df_final = df[['case:concept:name', 'concept:name', 'org:resource', 'time:timestamp']]
    df_final = df_final.sort_values(['org:resource', 'time:timestamp'])
    return df_final


# creating 80/20 split based on resources, ensuring a resource is in EITHER the test set OR the train set
def create_split(df_clean, test_size):
    resource_traces = (
        df_clean.sort_values(["org:resource", "time:timestamp"])
               .groupby("org:resource")["concept:name"]
               .apply(list)
    )

    resource_traces = resource_traces[resource_traces.apply(len) > 1]

    resources = resource_traces.index.tolist()

    # create set of train/test resource ids
    train_res, test_res = train_test_split(
        resources,
        test_size=test_size,
        random_state=1
    )

    train_traces = resource_traces.loc[train_res]
    test_traces = resource_traces.loc[test_res]

    return train_traces, test_traces


# prefix extraction on set list of prefix lengths, already implicitly buckets on prefix length
def build_prefix_df(resource_traces, prefix_lengths=[10], sliding_window=False, lastk=False, k = 3):

    all_rows = []
    for length in prefix_lengths:
        for resource, seq in resource_traces.items():

            if(len(seq) < length + 1):
                continue

            if(sliding_window):
                for i in range(length, len(seq)):
                    prefix = seq[i-length:i]
                    next_act = seq[i]

                    all_rows.append({
                    'resource': resource,
                    'subtrace': prefix,
                    'prefix_length': length,
                    'last_activity': prefix[-1],
                    'next_activity': next_act
                    })
            elif (not sliding_window and not lastk):
                prefix = seq[:length]
                next_act = seq[length]

                all_rows.append({
                'resource': resource,
                'subtrace': prefix,
                'prefix_length': length,
                'last_activity': prefix[-1],
                'next_activity': next_act
                })
            else:

                start_idx = max(0, length - k)
                
                prefix = seq[start_idx : length]
                
                next_act = seq[length]

                all_rows.append({
                    'resource': resource,
                    'subtrace': prefix,
                    'prefix_length': length,
                    'last_activity': prefix[-1] if len(prefix) > 0 else None, 
                    'next_activity': next_act
                })
            
            
    return pd.DataFrame(all_rows)




def process_dataset(df, prefix_length,strategy='prefix', k=None):
    df_clean = clean_dataset(df)

    train_split, test_split = create_split(df_clean, 0.2)

    if strategy == 'prefix':
        train_df = build_prefix_df(train_split, [prefix_length], False, False)
        test_df = build_prefix_df(test_split, [prefix_length], False, False)
    elif strategy == 'last_k':
        train_df = build_prefix_df(train_split, [prefix_length], False, True,k=k)
        test_df = build_prefix_df(test_split, [prefix_length], False, True, k=k)
    elif strategy == 'sliding_window':
        train_df = build_prefix_df(train_split, [prefix_length], True, False)
        test_df = build_prefix_df(test_split, [prefix_length], True, False)    

    return train_df, test_df, train_split, test_split