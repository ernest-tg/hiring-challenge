import pytorch_lightning as pl
import torch
import torchmetrics
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# TODO: could I set the dataset's preprocess from the model inside
#    the torch lightning module?


class ProjFromLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base_model_weights = MobileNet_V3_Large_Weights.DEFAULT
        self.base_model = mobilenet_v3_large(
            weights=base_model_weights)
        # TODO: can we get this 1000 from the weights?
        self.proj_layer = torch.nn.Linear(
            in_features=1000, out_features=2, bias=True)

    def forward(self, x):
        base_model_logits = self.base_model(x)
        logits = self.proj_layer(base_model_logits)
        return logits


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
        return logits


class ImageClassifierModule(pl.LightningModule):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.accuracy = torchmetrics.Accuracy()

    def basic_step(self, batch, batch_idx, split_name: str):
        images = batch['image']
        groundtruth = batch['groundtruth'].long()

        logits = self.classifier(images)
        loss = torch.nn.functional.cross_entropy(
            input=logits, target=groundtruth)
        self.accuracy(logits, groundtruth)
        self.log(f'{split_name}_accuracy', self.accuracy)
        self.log(f'{split_name}_loss', loss)
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
            self.classifier.proj_layer.parameters(),
            lr=1e-3)
        return optimizer
