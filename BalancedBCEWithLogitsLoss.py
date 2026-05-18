import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autrainer.datasets import AbstractDataset 

class BalancedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """
    Calculates a weighted BCEWithLogitsLoss, based on the frequency of the targets in the dataset.
    The higher the occurence, the lower the loss, and vice versa.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.CrossEntropyLoss.forward`.

        Converts the targets to `long` if it is a 1D tensor.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        if x.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: x has shape {x.shape}, but y has shape {y.shape}."
            )
        return super().forward(x, y.float())


    def setup(self, data: "AbstractDataset") -> None:
        """Calculate balanced weights for the dataset based on the target
        frequency in the training set.

        Args:
            data: Instance of the dataset.
        """
        target_matrix = data.df_train[data.target_column].apply(data.target_transform).values
        label_counts = target_matrix.sum(axis=0)  
        frequency = label_counts / len(target_matrix)

        # Compute weights as the inverse of frequency
        weight = torch.tensor(1.0 / frequency, dtype=torch.float32)
        weight /= weight.sum()
        self.weight = weight


if __name__=='__main__':
    # loss = BalancedBCEWithLogitsLoss()
    print("To be implemented...")