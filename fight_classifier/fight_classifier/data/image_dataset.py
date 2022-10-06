from typing import Any, Dict, Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch

from fight_classifier.data.data_split import split_based_on_column


class ImageDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
            self,
            image_df: pd.DataFrame,
            resize_size: int,
            crop_size: int,
            mean: Sequence[float],
            std: Sequence[float],
            image_path_col: str = 'frame_path',
            groundtruth_col: str = 'is_fight',
            image_augmentation: bool = False,
    ):
        """Classification dataset of images stored locally

        The images are stored in the paths of `image_df[image_path_col]`.

        Attributes:
            image_df (pd.DataFrame):
                A dataframe containing at least the columns `image_path_col`
                (paths to images) and `groundtruth_col` (classification
                groundtruth).
            resize_size (int):
                The length of the smallest side of the image after resizing.
            crop_size (int):
                The final lengths of the images' sides.
            mean (Sequence[float]):
                The values to subtract to each pixel.
            std (Sequence[float]):
                The values by which to divide each pixel.
            image_path_col (str, optional):
                The name of the column, in `image_df`, containing the paths
                of images.
            groundtruth_col (str, optional):
                The name of the column, in `image_df`, of the groundtruth.
            image_augmentation (bool, optional):
                Whether to apply some color augmentations to the images (after
                resizing and cropping to get them to the correct size).
        """
        self.image_df = image_df
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.image_path_col = image_path_col
        self.groundtruth_col = groundtruth_col
        self.image_augmentation = image_augmentation

    def __len__(self):
        return len(self.image_df)

    def preprocess_image(self, image_np):
        resize_transform = A.Compose([
            A.SmallestMaxSize(max_size=self.resize_size),
            A.RandomCrop(height=self.crop_size, width=self.crop_size),
        ])
        resized_image = resize_transform(image=image_np)["image"]

        augmentation_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.2, p=1.0),
            A.ChannelShuffle(p=0.1),
            A.InvertImg(p=0.5),

            A.GaussNoise(p=0.3),
            A.GaussianBlur(p=0.3),
        ])
        if self.image_augmentation:
            augmented_image = augmentation_transform(
                image=resized_image)["image"]
        else:
            augmented_image = resized_image

        normalize_transform = A.Normalize(mean=self.mean, std=self.std)
        normalized_image = normalize_transform(image=augmented_image)["image"]
        return augmented_image, normalized_image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_row = self.image_df.iloc[idx]
        image_path = image_row[self.image_path_col]
        image = Image.open(image_path)
        image_np = np.asarray(image)

        augmented_image, normalized_image = self.preprocess_image(
            image_np=image_np)
        groundtruth = image_row[self.groundtruth_col]
        return {
            'image_augmented': ToTensorV2()(image=augmented_image)["image"],
            'input': ToTensorV2()(image=normalized_image)["image"],
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
            image_augmentation: bool = False,
    ):
        super().__init__(),
        self.image_df = image_df
        self.batch_size = batch_size
        self.split_coherence_col = split_coherence_col
        self.preprocess_kwargs = preprocess_kwargs
        self.image_path_col = image_path_col
        self.groundtruth_col = groundtruth_col
        self.random_seed = random_seed
        self.image_augmentation = image_augmentation

    def setup(self, stage: str | None = None) -> None:
        self.split = split_based_on_column(
            data=self.image_df, col_name=self.split_coherence_col,
            min_val_size=0.1, max_val_size=0.15,
            random_seed=self.random_seed,
        )
        self.train_dataset = ImageDataset(
            self.split['train'],
            image_path_col=self.image_path_col,
            groundtruth_col=self.groundtruth_col,
            image_augmentation=self.image_augmentation,
            **self.preprocess_kwargs)
        self.val_dataset = ImageDataset(
            self.split['val'],
            image_path_col=self.image_path_col,
            groundtruth_col=self.groundtruth_col,
            image_augmentation=self.image_augmentation,
            **self.preprocess_kwargs)
        self.test_dataset = ImageDataset(
            self.split['val'],
            image_path_col=self.image_path_col,
            groundtruth_col=self.groundtruth_col,
            image_augmentation=False,
            **self.preprocess_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,)

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=10,)

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=10,)

    def predict_dataloader(self):
        return self.test_dataloader()