import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def imshow_chw(img_chw, ax=None, **kwargs):
    img_hwc = np.transpose(a=img_chw, axes=(1, 2, 0))
    if ax is None:
        plt.imshow(img_hwc, **kwargs)
    else:
        ax.imshow(img_hwc, **kwargs)


def chw_array_from_fig(fig: matplotlib.figure.Figure):
    fig.canvas.draw()
    fig_as_hwc_array = np.array(fig.canvas.renderer.buffer_rgba())
    fig_as_chw_array = np.transpose(fig_as_hwc_array, (2, 0, 1))
    # plt.close(fig=fig)
    fig.clf()
    plt.close()
    return fig_as_chw_array
