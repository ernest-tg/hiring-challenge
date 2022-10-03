import numpy as np
import pandas as pd


def determine_split_size(data, split_size_param):
    if isinstance(split_size_param, int):
        return split_size_param
    elif isinstance(split_size_param, float):
        return int(len(data) * split_size_param)
    else:
        raise ValueError("Invalid split size parameter")


def naive_split(data, val_size=0.1):
    n_rows = len(data)
    n_val_rows = determine_split_size(
        data=data, split_size_param=val_size)
    val_ids = np.random.choice(a=n_rows, size=n_val_rows)
    train_ids_set = set(range(n_rows)) - set(val_ids)
    train_ids = list(train_ids_set)

    train_data = data.iloc[train_ids]
    val_data = data.iloc[val_ids]
    return {
        'train': train_data,
        'val': val_data,
    }
    
def split_based_on_column(data, col_name, val_size=0.1):
    pass