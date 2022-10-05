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


def split_based_on_column(
            data,
            col_name: str,
            min_val_size: int | float = 0.1,
            max_val_size: int | float = 0.2,
            random_seed: int | None = None,
):
    if random_seed is not None:
        np.random.seed(random_seed)
    min_val_rows = determine_split_size(
        data=data, split_size_param=min_val_size)
    max_val_rows = determine_split_size(
        data=data, split_size_param=max_val_size)

    group_counts = data.groupby([col_name])[col_name].count()
    n_groups = len(group_counts)

    shuffled_group_ids = np.arange(n_groups)
    np.random.shuffle(shuffled_group_ids)

    val_groups = []
    current_n_rows = 0
    for group_id in shuffled_group_ids:
        if current_n_rows >= min_val_rows:
            break
        group_col = group_counts.index[group_id]
        n_rows_in_group = group_counts.iloc[group_id]
        if current_n_rows + n_rows_in_group < max_val_rows:
            val_groups.append(group_col)
            current_n_rows += n_rows_in_group

    val_mask = data[col_name].isin(val_groups)
    train_data = data[~val_mask]
    val_data = data[val_mask]
    if not (min_val_rows <= current_n_rows <= max_val_rows):
        raise ValueError(
            "Could not find a split of the right size")
    return {
        'train': train_data,
        'val': val_data,
    }
