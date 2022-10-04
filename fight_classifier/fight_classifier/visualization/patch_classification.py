import torch

def visualization_probas(img, probas):
    if tuple(img.shape[1:]) != tuple(probas.shape[1:]):
        probas = torch.nn.functional.interpolate(
            input=probas, size=img.shape[1:])
    img_gray = torch.mean(img, dim=0, keepdim=True)
    print(img_gray.shape)
    probas_viz = 