import numpy as np
import sklearn.metrics

from aurainer.metrics.abstract_metric import BaseAscendingMetric

class MLPrecision(BaseAscendingMetric):
    def __init__(self):
        """F1 macro metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-precision",
            fn=sklearn.metrics.precision_score,
            average="macro",
        )

    def unitary(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unitary evaluation of metric.

        Metric computed with `average='binary'`
        i.e. only accounting for the positive label.

        Args:
            y_true: ground truth values.
            y_pred: prediction values.

        Returns:
            The unitary score.
        """
        return float(
            sklearn.metrics.precision_score(
                y_true=y_true, y_pred=y_pred, average="binary"
            )
        )


class MLRecall(BaseAscendingMetric):
    def __init__(self):
        """F1 macro metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-recall",
            fn=sklearn.metrics.recall_score,
            average="macro",
        )

    def unitary(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unitary evaluation of metric.

        Metric computed with `average='binary'`
        i.e. only accounting for the positive label.

        Args:
            y_true: ground truth values.
            y_pred: prediction values.

        Returns:
            The unitary score.
        """
        return float(
            sklearn.metrics.recall_score(
                y_true=y_true, y_pred=y_pred, average="binary"
            )
        )