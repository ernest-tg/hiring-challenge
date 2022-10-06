import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from fight_classifier.visualization import chw_array_from_fig


def viz_patch_heatmap(img, probas):
    img_gray = torch.mean(img, dim=0, dtype=torch.float)

    fig, axes = plt.subplots(ncols=2)
    axes[0].set_axis_off()
    axes[1].set_axis_off()
    axes[0].imshow(torch.permute(img, dims=(1, 2, 0)))

    hmax = sns.heatmap(
        data=probas,
        cmap=sns.color_palette("vlag", as_cmap=True),
        alpha=0.5,
        zorder=2,
        ax=axes[1],
    )
    hmax.imshow(
        img_gray, aspect=hmax.get_aspect(),
        extent=hmax.get_xlim()+hmax.get_ylim(),
        zorder=1,
    )
    fig_as_array = chw_array_from_fig(fig=fig)
    return fig_as_array
