import matplotlib.pyplot as plt
import numpy as np


def imshow_chw(img_chw):
    img_hwc = np.transpose(a=img_chw, axes=(1, 2, 0))
    plt.imshow(img_hwc)
