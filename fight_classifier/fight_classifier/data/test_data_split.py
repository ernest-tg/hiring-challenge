
import pandas as pd
import pytest

from fight_classifier.data.data_split import (
    determine_split_size, naive_split, multiple_splits_based_on_column,
    split_based_on_column)


@pytest.fixture
def df_fixture():
    """DataFrame to illustrate the dataset splitting functions

    This corresponds to the problem we want to solve. Some pairs of examples
    come from the same source (column 'src'). And, because of that, they are
    almost duplicates of each others (only the random field 'r' changes).

    A classifier might rely on the column 'trap' for classification. This
    represents features which are unique to each source. And, since examples
    from the same source have the same groundtruth, we can easily reach 100%
    train accuracy. But, since there is no relationship between 'trap' and 'gt',
    this would not generalize.
    * If, for every example in the validation dataset, there is an example with
        the same source in the train dataset, we may have 100% accuracy on the
        validation set, and mistake ourselves into thinking our classifier will
        generalize.
    * If, the sources are distinct between the train and validation dataset,
        then a classifier based on the 'trap' column would do no better than
        chance on the validation dataset. We will be aware that our classifier
        does not generalize.

    The 'informative' column is a good (but imperfect) predictor of the
    groundtruth 'gt'.

    If a concrete example is easier, you can imagine that each row corresponds
    to a small clip.
    - "src" is the url of the original video, from which clips are taken.
    - "trap" represents the watermark added to each video (different for each
      video)
    - "r": is a hash of the clip
    - "informative": is the number of people in the video.
    - "gt": is whether there is a fight in the clip or not.
    """
    return pd.DataFrame([
        {'src': 'a', 'trap': 0.0, 'r': 0.51, 'informative': 1, 'gt': False},
        {'src': 'a', 'trap': 0.0, 'r': 0.75, 'informative': 1, 'gt': False},
        {'src': 'b', 'trap': 0.8, 'r': 0.95, 'informative': 1, 'gt': False},
        {'src': 'b', 'trap': 0.8, 'r': 0.80, 'informative': 1, 'gt': False},
        {'src': 'c', 'trap': 0.6, 'r': 0.64, 'informative': 2, 'gt': False},
        {'src': 'c', 'trap': 0.6, 'r': 0.98, 'informative': 2, 'gt': False},
        {'src': 'd', 'trap': 0.4, 'r': 0.53, 'informative': 1, 'gt': False},
        {'src': 'd', 'trap': 0.4, 'r': 0.37, 'informative': 1, 'gt': False},
        {'src': 'e', 'trap': 0.3, 'r': 0.95, 'informative': 1, 'gt': False},
        {'src': 'e', 'trap': 0.3, 'r': 0.11, 'informative': 1, 'gt': False},
        {'src': 'f', 'trap': 0.5, 'r': 0.53, 'informative': 2, 'gt': True},
        {'src': 'f', 'trap': 0.5, 'r': 0.15, 'informative': 2, 'gt': True},
        {'src': 'g', 'trap': 0.2, 'r': 0.65, 'informative': 2, 'gt': True},
        {'src': 'g', 'trap': 0.2, 'r': 0.25, 'informative': 2, 'gt': True},
        {'src': 'h', 'trap': 0.7, 'r': 0.36, 'informative': 2, 'gt': True},
        {'src': 'h', 'trap': 0.7, 'r': 0.63, 'informative': 2, 'gt': True},
        {'src': 'i', 'trap': 0.1, 'r': 0.98, 'informative': 2, 'gt': True},
        {'src': 'i', 'trap': 0.1, 'r': 0.41, 'informative': 2, 'gt': True},
        {'src': 'j', 'trap': 0.9, 'r': 0.39, 'informative': 2, 'gt': True},
        {'src': 'j', 'trap': 0.9, 'r': 0.68, 'informative': 2, 'gt': True},
    ])


def test_determine_split_size(df_fixture):
    # Test case with split_size_param an integer
    assert determine_split_size(data=df_fixture, split_size_param=5) == 5
    assert determine_split_size(data=df_fixture, split_size_param=7) == 7

    # Test case with split_size_param a float
    assert determine_split_size(data=df_fixture, split_size_param=0.1) == 2
    assert determine_split_size(data=df_fixture, split_size_param=0.16) == 3

    # Raises an exception for other types
    with pytest.raises(ValueError):
        determine_split_size(data=df_fixture, split_size_param='not_a_number')


def test_naive_split(df_fixture):
    dataset_split = naive_split(data=df_fixture, val_size=0.1)
    assert set(dataset_split.keys()) == {'train', 'val'}

    train_dataset = dataset_split['train']
    val_dataset = dataset_split['val']
    assert len(train_dataset) == 18
    assert len(val_dataset) == 2

    # Since the 'r' values are all distinct, this means `train_dataset` and
    # `val_dataset` form a partition of `df_fixture`.
    assert set(train_dataset['r']) | set(val_dataset['r']) == set(df_fixture['r'])


def test_naive_split_random_seed(df_fixture):
    # We test that we always get the same result if we fix the random seed
    dataset_split = naive_split(data=df_fixture, val_size=0.1, random_seed=0)
    val_dataset = dataset_split['val']
    assert set(val_dataset['r']) == {0.63, 0.65}


def test_split_based_on_column(df_fixture):
    dataset_split = split_based_on_column(
        data=df_fixture, col_name='src', min_val_size=0.1, max_val_size=0.2)
    assert set(dataset_split.keys()) == {'train', 'val'}

    train_dataset = dataset_split['train']
    val_dataset = dataset_split['val']
    assert len(train_dataset) == 18
    assert len(val_dataset) == 2

    # Since the 'r' values are all distinct, this means `train_dataset` and
    # `val_dataset` form a partition of `df_fixture`.
    assert set(train_dataset['r']) | set(
        val_dataset['r']) == set(df_fixture['r'])

    # We check that there is no source with examples in both the train and
    # validation datasets.
    assert set(train_dataset['src']).isdisjoint(val_dataset['src'])
