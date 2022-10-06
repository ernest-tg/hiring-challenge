import itertools
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from fight_classifier.visualization import chw_array_from_fig


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Returns (as a numpy array) a visualization of a confusion matrix

    Adapted from: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    to save time. In a commercial project, I'd avoid that for copyright purpose.

    Args:
       cm (np.ndarray, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    fig, ax = plt.subplots()
    cm_mappable = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")
    plt.colorbar(cm_mappable, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks, class_names, rotation=45, )
    ax.set_yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)
                   [:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        ax.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()

    fig_as_array = chw_array_from_fig(fig=fig)
    return fig_as_array


def plot_precision_recall_curve(
        precisions: Sequence[float],
        recalls: Sequence[float],
        average_precision: float,
):
    # Create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, color='purple')

    # Add axis labels to plot
    ax.set_title(f'Precision-Recall Curve (avg {100*average_precision:.1f}%)')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    fig_as_array = chw_array_from_fig(fig=fig)
    return fig_as_array
