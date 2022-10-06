from typing import Dict, Iterable

import torch
from torch.nn.functional import relu, softmax

from fight_classifier.model.image_based_model import ImageClassifier


class SmallCnnImageClassifier(ImageClassifier):
    """Classifies small patches and average the prediction over the image

    This classifier was designed to investigate spurious correlations:
    - The models are too small and their receptive field is too small (3x3 for
        layer 1, 7x7 for layer 2) to correctly classify an image. But it could
        detect a difference in color distribution, in blurriness, or other small
        detail whose correlation with the groundtruth in the dataset is a
        coincidence.
    - Since we are classifying patches, as an intermediary step to classify the
        whole image, we can visualize those patches' classifications. It can
        help us understand what signal those small classifiers pick up.

    Attributes:
        n_classes (int): The number of classes to output.
        n_layers (int): The number of convolution layers to apply.
    """
    def __init__(self, n_classes: int, n_layers: int = 3):
        super().__init__(n_classes=n_classes)
        self.n_layers = n_layers
        d1 = 2 if n_layers == 1 else 20
        d2 = 2 if n_layers == 2 else 30
        self.lay1 = torch.nn.Conv2d(
            in_channels=3, out_channels=d1, kernel_size=3, stride=2)
        self.lay2 = torch.nn.Conv2d(
            in_channels=d1, out_channels=d2, kernel_size=3)
        self.lay3 = torch.nn.Conv2d(
            in_channels=d2, out_channels=self.n_classes, kernel_size=3)

    def trainable_parameters(self) -> Iterable[torch.nn.parameter.Parameter]:
        """Parameters to be trained"""
        return self.parameters()

    def patches_probas(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification probas of patches

        Args:
            x (torch.Tensor):
                Tensor of the shape (batch_size, 3, h, w)

        Returns:
            patches_probas (torch.Tensor):
                Tensor of the shape (batch_size, n_classes, h_p, w_p) with
                values between 0 and 1. Summing to 1 along dimension 1.
                Each `patches_probas[b, class_id, r, c]` is the probability that
                the `x[b]` image is of class `class_id`. But, given that the
                model has a very small receptive field, this should be seen as
                the probability that the receptive field patch is part of an
                image of class `class_id`.
        """
        out1 = self.lay1(x)
        out2 = self.lay2(relu(out1))
        out3 = self.lay3(relu(out2))
        outs = [out1, out2, out3]
        patches_logits = outs[self.n_layers-1]
        patches_probas = softmax(patches_logits, dim=1)
        return patches_probas

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns classification probas of images as well as patches

        Args:
            x (torch.Tensor):
                Tensor of the shape (batch_size, 3, h, w)

        Returns:
            Dict[str, torch.Tensor]:
                The dictionary contains the following mapping:
                - "patches_probas" is mapped to a tensor of the shape
                    (batch_size, n_classes, h_p, w_p), representing the
                    classification probabilites of images' patches.
                - "image_probas" is mapped to a tensor of the shape
                    (batch_size, self.num_classes) representing the
                    classification probabilities of each image.
        """
        patches_probas = self.patches_probas(x=x)
        images_probas = torch.mean(patches_probas, dim=(2, 3))
        return {
            'patches_probas': patches_probas,
            'image_probas': images_probas,
        }
