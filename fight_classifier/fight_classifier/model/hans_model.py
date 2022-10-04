import torch
from torch.nn.functional import relu, softmax

# TODO: could I set the dataset's preprocess from the model inside
#    the torch lightning module?


class SmallCnnImageClassifier(torch.nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        d1 = 2 if n_layers == 1 else 20
        d2 = 2 if n_layers == 2 else 30
        self.lay1 = torch.nn.Conv2d(
            in_channels=3, out_channels=d1, kernel_size=3, stride=2)
        self.lay2 = torch.nn.Conv2d(
            in_channels=d1, out_channels=d2, kernel_size=3)
        self.lay3 = torch.nn.Conv2d(
            in_channels=d2, out_channels=2, kernel_size=3)

    def trainable_parameters(self):
        return self.parameters()

    def patches_probas(self, x):
        out1 = self.lay1(x)
        out2 = self.lay2(relu(out1))
        out3 = self.lay3(relu(out2))
        outs = [out1, out2, out3]
        patches_logits = outs[self.n_layers-1]
        patches_probas = softmax(patches_logits, dim=1)
        return patches_probas

    def forward(self, x):
        patches_probas = self.patches_probas(x=x)
        images_probas = torch.mean(patches_probas, dim=(2, 3))
        return images_probas
