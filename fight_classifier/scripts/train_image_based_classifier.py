import argparse
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from fight_classifier import DATASET_DIR, PROJECT_DIR
from fight_classifier.data.image_dataset import ImageDataModule
from fight_classifier.model.image_based_model import (
    ProjFromFeatures, ImageClassifierModule)


def train(
    batch_size: int, split_coherence_col: str = 'fine_category',
):
    frames_dir = DATASET_DIR / 'raw_frames/'

    frames_df = pd.read_csv(str(frames_dir / 'frames.csv'))

    base_model_weights = MobileNet_V3_Large_Weights.DEFAULT
    base_model = mobilenet_v3_large(weights=base_model_weights)
    base_model.eval()
    preprocess = base_model_weights.transforms()

    image_data_module = ImageDataModule(
        image_df=frames_df,
        batch_size=batch_size,
        preprocess=preprocess,
        split_coherence_col=split_coherence_col)

    classifier = ProjFromFeatures()

    classif_module = ImageClassifierModule(classifier=classifier)

    trainer = pl.Trainer(
        default_root_dir=str(PROJECT_DIR),
        val_check_interval=500,
    )

    trainer.fit(
        model=classif_module,
        datamodule=image_data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--split_coherence_col", type=str, default='fine_category')
    args = parser.parse_args()
    
    train(
        batch_size=args.batch_size,
        split_coherence_col=args.split_coherence_col)
