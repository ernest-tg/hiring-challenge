from abc import ABCMeta, abstractmethod
from typing import Dict, Iterable

import torch
from torch.nn.functional import softmax
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class ImageClassifier(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, n_classes: int):
        """Abstract class of image classifier

        Args:
            n_classes (int): The number of classes.
        """
        super().__init__()
        self.n_classes = n_classes

    @abstractmethod
    def trainable_parameters(self) -> Iterable[torch.nn.parameter.Parameter]:
        """Parameters to be trained"""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns classification probas of images

        Args:
            x (torch.Tensor):
                Tensor of the shape (batch_size, 3, h, w)

        Returns:
            Dict[str, torch.Tensor]:
                The dictionary must contain at least the key 'image_probas',
                mapping it to a tensor of shape (batch_size, self.n_classes)
                with values in [0, 1], summing to 1 along dimension 1.
        """
        ...


class ImageBasedVideoClassifier(torch.nn.Module):
    """Runs a classifier on every other frame and averages the results"""
    def __init__(self, image_classifier: ImageClassifier):
        super().__init__()
        self.image_classifier = image_classifier
        self.n_classes = self.image_classifier.n_classes

    def forward(self, x):
        """Returns classification probas of videos

        Args:
            x (torch.Tensor):
                Batch of videos, of shape (batch_size, n_frames, 3, h, w)
        """
        # We keep every other frame to speed up the computation
        x = x[:, ::2]
        batch_size, n_frames, _, h, w = x.shape
        frame_in_batch_x = torch.reshape(
            input=torch.as_tensor(x), shape=(batch_size*n_frames, 3, h, w)
        ).float()
        # {'image_probas': Tensor(batch_size*n_frames, n_classes)}
        frame_in_batch_results = self.image_classifier(x=frame_in_batch_x)

        # Tensor(batch_size, n_frames, self.n_classes)
        frames_probas = torch.reshape(
            input=frame_in_batch_results['image_probas'],
            shape=(batch_size, n_frames, self.n_classes))

        videos_probas = torch.mean(input=frames_probas, dim=1)
        return videos_probas


class ProjFromFeatures(ImageClassifier):
    """Image classifier based on mobilenet features

    We take a pre-trained mobilenet and freeze its weights (they do not appear
    in the `self.trainable_parameters()` used by the `ImageClassifierModule`).

    We only train a single fully connected layer on top of it.
    """
    def __init__(self, n_classes):
        super().__init__(n_classes=n_classes)
        base_model_weights = MobileNet_V3_Large_Weights.DEFAULT
        self.base_model = mobilenet_v3_large(
            weights=base_model_weights)
        self.proj_layer = torch.nn.Linear(
            in_features=960, out_features=2, bias=True)

        self.feature_extractor = create_feature_extractor(
            self.base_model, return_nodes=['flatten'])

    def forward(self, x):
        """Returns classification probas of images

        Args:
            x (torch.Tensor):
                Tensor of the shape (batch_size, 3, h, w)

        Returns:
            Dict[str, torch.Tensor]:
                The dictionary must contain at least the key 'image_probas',
                mapping it to a tensor of shape (batch_size, self.n_classes)
                with values in [0, 1], summing to 1 along dimension 1.
        """
        base_features = self.feature_extractor(x)['flatten']
        logits = self.proj_layer(base_features)
        probas = softmax(logits, dim=1)
        return {
            'image_probas': probas,
        }

    def trainable_parameters(self) -> Iterable[torch.nn.parameter.Parameter]:
        """Parameters to be trained

        We only return the parameters of the final layer, not of the mobilenet.
        """
        return self.proj_layer.parameters()
