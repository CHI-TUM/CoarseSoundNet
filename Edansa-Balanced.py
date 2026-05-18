import os
import torch
import pandas as pd
from autrainer.transforms import RandomCrop, SmartCompose
from typing import Any, Dict, List, Tuple, Union, Optional
from autrainer.datasets import BaseMLClassificationDataset, EDANSA2019
from autrainer.transforms import SmartCompose
from torch.utils.data import DataLoader, WeightedRandomSampler
from omegaconf import OmegaConf
from omegaconf import DictConfig
from functools import cached_property


class EdansaBalanced(EDANSA2019):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: List[str],
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        features_path: Optional[str] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            features_path=features_path,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
            threshold=threshold,
        )

        # Get label frequencies and weights
        self.multilabels = self.df_train[target_column]
        self.num_samples = len(self.df_train)

        self.label_counts = self.multilabels.sum(axis=0)
        self.label_frequencies = self.label_counts / self.num_samples
        self.label_weights = 1.0 / self.label_frequencies

        # Determine the sample weights
        # For this, take the maximum weight of all active labels for a sample
        self.sample_weights = torch.tensor((self.multilabels * self.label_weights).max(axis=1), dtype=torch.float)
        self.sampler = WeightedRandomSampler(weights=self.sample_weights, num_samples=self.num_samples, replacement=True, generator=self._generator)