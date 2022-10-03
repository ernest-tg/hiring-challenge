from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from fight_classifier.data.data_split import naive_split


class ImageDataset(Dataset):
    def __init__(
            self,
            image_df: pd.DataFrame,
            image_path_col: str = 'frame_path',
            groundtruth_col: str = 'is_fight',
            preprocess: Callable | None = None,
    ):
        self.image_df = image_df
        self.image_path_col = image_path_col
        self.groundtruth_col = groundtruth_col
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx: int):
        image_row = self.image_df.iloc[idx]
        image_path = image_row[self.image_path_col]
        image = Image.open(image_path)
        if self.preprocess is None:
            image_t = np.asarray(image)
        else:
            image_t = self.preprocess(image)

        groundtruth = image_row[self.groundtruth_col]
        return {
            'image': image_t,
            'groundtruth': groundtruth,
        }


class ImageDataModule(pl.LightningDataModule):
    def __init__(
            self,
            image_df: pd.DataFrame,
            batch_size: int,
            image_path_col: str = 'frame_path',
            groundtruth_col: str = 'is_fight',
            preprocess: Callable | None = None,
    ):
        super().__init__(),
        self.image_df = image_df
        self.batch_size = batch_size
        self.image_path_col = image_path_col
        self.groundtruth_col = groundtruth_col
        self.preprocess = preprocess

    def setup(self, stage: str | None = None) -> None:
        split = naive_split(data=self.image_df, val_size=0.1)
        self.train_dataset = ImageDataset(
            split['train'],
            image_path_col=self.image_path_col,
            groundtruth_col=self.groundtruth_col,
            preprocess=self.preprocess)
        self.val_dataset = ImageDataset(
            split['val'],
            image_path_col=self.image_path_col,
            groundtruth_col=self.groundtruth_col,
            preprocess=self.preprocess)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True)

    def val_dataloader(self):
        # TODO: later, differentiate from test by adding transforms here but
        # not on test
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return self.test_dataloader()