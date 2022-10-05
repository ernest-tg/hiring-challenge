import pytorch_lightning as pl
import torch
from torch.nn.functional import nll_loss, softmax
import torchmetrics
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from fight_classifier.visualization.patch_classification import viz_patch_heatmap

# TODO: could I set the dataset's preprocess from the model inside
#    the torch lightning module?


class ImageBasedVideoClassifier(torch.nn.Module):
    def __init__(self, image_classifier):
        self.image_classifier = image_classifier

    def forward(self, x):
        """

        Args:
            x (torch.Tensor):
                Batch of videos, of shape (batch_size, n_frames, 3, h, w)
        """
        batch_size, n_frames, _, h, w = x.shape

class ProjFromFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base_model_weights = MobileNet_V3_Large_Weights.DEFAULT
        self.base_model = mobilenet_v3_large(
            weights=base_model_weights)
        # TODO: can we get this 1000 from the weights?
        self.proj_layer = torch.nn.Linear(
            in_features=960, out_features=2, bias=True)

        self.feature_extractor = create_feature_extractor(
            self.base_model, return_nodes=['flatten'])

    def forward(self, x):
        base_features = self.feature_extractor(x)['flatten']
        logits = self.proj_layer(base_features)
        probas = softmax(logits, dim=1)
        return {
            'image_probas': probas,
        }

    def trainable_parameters(self):
        return self.proj_layer.parameters()


class ImageClassifierModule(pl.LightningModule):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.accuracy = torchmetrics.Accuracy()

    def basic_step(self, batch, batch_idx, split_name: str):
        images = batch['input']
        # self.log('raw_images', batch['image_raw'])
        tb_logger: torch.utils.tensorboard.writer.SummaryWriter = self.logger.experiment
        if batch_idx % 20 == 0:
            tb_logger.add_image(
                'augmented_images', batch['image_augmented'][0], global_step=batch_idx)
            tb_logger.add_image(
                'input_images', images[0], global_step=batch_idx)
        groundtruth = batch['groundtruth'].long()

        classifier_output = self.classifier(images)
        probas = classifier_output['image_probas']
        if 'patches_probas' in classifier_output and batch_idx % 20 == 0:
            patches_probas_viz = viz_patch_heatmap(
                batch['image_augmented'][0],
                classifier_output['patches_probas'][0, 1].detach(),
            )
            tb_logger.add_image(
                'patches_probas', patches_probas_viz, global_step=batch_idx)

        # loss = torch.nn.functional.cross_entropy(input=logits, target=groundtruth)
        loss = nll_loss(
            input=torch.log(probas),
            target=groundtruth,
            reduction='sum')
        self.accuracy(probas, groundtruth)
        self.log(f'{split_name}_accuracy', self.accuracy, prog_bar=True)
        self.log(f'{split_name}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.basic_step(
            batch=batch, batch_idx=batch_idx, split_name='train')

    def validation_step(self, batch, batch_idx):
        return self.basic_step(
            batch=batch, batch_idx=batch_idx, split_name='val')

    def test_step(self, batch, batch_idx):
        return self.basic_step(
            batch=batch, batch_idx=batch_idx, split_name='test')

    def configure_optimizers(self):
        # TODO: the model should define the trainable parameters?
        optimizer = torch.optim.Adam(
            self.classifier.trainable_parameters(),
            lr=1e-3)
        return optimizer
