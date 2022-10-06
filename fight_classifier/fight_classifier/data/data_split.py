from typing import Dict

import numpy as np
import pandas as pd


def determine_split_size(data: pd.DataFrame, split_size_param: int | float) -> int:
    """Determine the number of rows of a split based on a param.

    We allow to specify the size of a split either directly (with an integer),
    or as a fraction of the whole dataset (with a float). In the second case,
    this function compute how many rows this represent.

    Args:
        data (pd.DataFrame):
            The data we want to split. Only its length is actually needed.
        split_size_param (int | float):
            The size of the split (validation/test/...) we want.
            - If an int, this is the number of rows, and this function returns
              `split_size_param` directly.
            - If a float, this is the fraction of the dataset which should go
                to the split.

    Returns:
        n_rows (int): The number of rows of the split.
    """
    if isinstance(split_size_param, int):
        return split_size_param
    elif isinstance(split_size_param, float):
        return int(len(data) * split_size_param)
    else:
        raise ValueError("Invalid split size parameter")


def naive_split(
        data: pd.DataFrame,
        val_size: int | float = 0.1,
        random_seed: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """Returns a train/val split without taking into account any column

    Args:
        data (pd.DataFrame):
            The data we want to split.
        val_size (int | float, optional):
            The size of the validation dataset. Either as a number of rows (int)
            or as a fraction of the dataset (float).
        random_seed (int | None, optional):
            If non-None, this will be used as a random seed.

    Returns:
        dataset_split (Dict[str, pd.DataFrame]):
            The dictionary maps 'train' to a training dataset, and 'val' to a
            validation dataset (they are complementary to each other).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    n_rows = len(data)
    n_val_rows = determine_split_size(data=data, split_size_param=val_size)
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
            data: pd.DataFrame,
            col_name: str,
            min_val_size: int | float = 0.1,
            max_val_size: int | float = 0.2,
            random_seed: int | None = None,
):
    """Returns a train/val split separating rows with the same `col_name` value

    Args:
        data (pd.DataFrame):
            The data we want to split.
        min_val_size (int | float, optional):
            The minimum size of the validation dataset. Either as a number of
            rows (int) or as a fraction of the dataset (float).
        max_val_size (int | float, optional):
            The minimum size of the validation dataset. Either as a number of
            rows (int) or as a fraction of the dataset (float).
        random_seed (int | None, optional):
            If non-None, this will be used as a random seed.

    Returns:
        dataset_split (Dict[str, pd.DataFrame]):
            The dictionary maps 'train' to a training dataset, and 'val' to a
            validation dataset (they are complementary to each other).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    min_val_rows = determine_split_size(data=data, split_size_param=min_val_size)
    max_val_rows = determine_split_size(data=data, split_size_param=max_val_size)

    # Series of the number of rows for each `col_name` value, indexed by `col_name`
    group_counts = data.groupby([col_name])[col_name].count()

    n_groups = len(group_counts)

    shuffled_group_ids = np.arange(n_groups)
    np.random.shuffle(shuffled_group_ids)

    # We start with an empty validation dataset. And we add groups (set of rows)
    # with the same `col_name` value until we have at least `min_val_size` rows.
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
    if not (min_val_rows <= current_n_rows <= max_val_rows):
        raise ValueError(
            "Could not find a split of the right size")

    val_mask = data[col_name].isin(val_groups)
    train_data = data[~val_mask]
    val_data = data[val_mask]
    dataset_split = {'train': train_data, 'val': val_data}
    return dataset_split
