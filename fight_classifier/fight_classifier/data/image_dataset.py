from typing import Any, Callable, Dict


import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision.transforms import functional as F

from fight_classifier.data.data_split import split_based_on_column


class ImageDataset(Dataset):
    def __init__(
            self,
            image_df: pd.DataFrame,
            resize_size,
            crop_size,
            mean,
            std,
            image_path_col: str = 'frame_path',
            groundtruth_col: str = 'is_fight',
    ):
        self.image_df = image_df
        self.image_path_col = image_path_col
        self.groundtruth_col = groundtruth_col
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx: int):
        image_row = self.image_df.iloc[idx]
        image_path = image_row[self.image_path_col]
        image = Image.open(image_path)
        image_np = np.asarray(image)

        transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=self.resize_size),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomCrop(height=self.crop_size, width=self.crop_size),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                # A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )
        augmented_image = transform(image=image_np)["image"]

        # TODO: replace with SmallestMaxSize --> random crop
        # resized_image = F.resize(image, size=self.resize_size)
        # cropped_image = F.center_crop(resized_image, self.crop_size)
        # if not isinstance(cropped_image, torch.Tensor):
        #    cropped_image = F.pil_to_tensor(cropped_image)
        # float_image = F.convert_image_dtype(cropped_image, torch.float)
        #normalized_image = F.normalize(float_image, mean=self.mean, std=self.std)

        groundtruth = image_row[self.groundtruth_col]
        return {
            # We may want to visualize the original (before augmentations and
            # normalization) for debuggging purpose. However, the original
            # images may have different shapes, which would be a problem when
            # batching. This resized image is the closest thing to the original
            # image which can be batched.
            # 'image_raw': np.asarray(F.resize(image, size=[self.resize_size])),
            'image': augmented_image,
            'groundtruth': groundtruth,
        }


class ImageDataModule(pl.LightningDataModule):
    def __init__(
            self,
            image_df: pd.DataFrame,
            batch_size: int,
            split_coherence_col: str,
            preprocess_kwargs: Dict[str, Any],
            image_path_col: str = 'frame_path',
            groundtruth_col: str = 'is_fight',
            random_seed: int | None = None,
    ):
        super().__init__(),
        self.image_df = image_df
        self.batch_size = batch_size
        self.split_coherence_col = split_coherence_col
        self.image_path_col = image_path_col
        self.groundtruth_col = groundtruth_col
        self.preprocess_kwargs = preprocess_kwargs
        self.random_seed = random_seed

    def setup(self, stage: str | None = None) -> None:
        split = split_based_on_column(
            data=self.image_df, col_name=self.split_coherence_col,
            min_val_size=0.1, max_val_size=0.15
        )
        self.train_dataset = ImageDataset(
            split['train'],
            image_path_col=self.image_path_col,
            groundtruth_col=self.groundtruth_col,
            **self.preprocess_kwargs)
        self.val_dataset = ImageDataset(
            split['val'],
            image_path_col=self.image_path_col,
            groundtruth_col=self.groundtruth_col,
            **self.preprocess_kwargs)

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