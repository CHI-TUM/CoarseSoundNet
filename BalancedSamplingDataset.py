import os
import torch
import pandas as pd
from autrainer.transforms import RandomCrop, SmartCompose
from typing import Any, Dict, List, Tuple, Union, Optional
from autrainer.datasets import BaseMLClassificationDataset
from autrainer.transforms import SmartCompose
from torch.utils.data import DataLoader, WeightedRandomSampler
from omegaconf import OmegaConf
from omegaconf import DictConfig
from functools import cached_property


class BalancedSamplingDataset(BaseMLClassificationDataset):
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
        # batch_size: int, # obsolete in new version
        # inference_batch_size: Optional[int] = None, # obsolete in new version
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
            # batch_size=batch_size, # obsolete in new version
            # inference_batch_size=inference_batch_size, # obsolete in new version
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
            threshold=threshold,
        )
        # Quick fix to extract PowerSpectrograms
        # self.df_train = pd.read_csv(os.path.join(self.path, "train_fullPaths.csv"))
        # self.df_dev = pd.read_csv(os.path.join(self.path, "dev_fullPaths.csv"))
        # self.df_test = pd.read_csv(os.path.join(self.path, "test_fullPaths.csv"))

        # Get label frequencies and weights
        self.multilabels = self.df_train[target_column]
        self.num_samples = len(self.df_train)

        self.label_counts = self.multilabels.sum(axis=0)
        self.label_frequencies = self.label_counts / self.num_samples
        self.label_weights = 1.0 / self.label_frequencies

        # Determine the sample weights
        # For this, take the maximum weight of all active labels for a sample
        # Question: Should we rather build the sum out of all weights for the active labels?
        self.sample_weights = torch.tensor((self.multilabels * self.label_weights).max(axis=1), dtype=torch.float)
        self.sampler = WeightedRandomSampler(weights=self.sample_weights, num_samples=self.num_samples, replacement=True, generator=self._generator)


    # # Just Quickly overwrite it for the PowerSpec extraction
    # @overwrite
    # def load_dataframes(
    #     self,
    # ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     """Load the dataframes.

    #     Returns:
    #         Dataframes for training, development, and testing.
    #     """
    #     return (
    #         pd.read_csv(os.path.join(self.path, "train_fullPaths.csv")),
    #         pd.read_csv(os.path.join(self.path, "dev_fullPaths.csv")),
    #         pd.read_csv(os.path.join(self.path, "test_fullPaths.csv")),
    #     )

    @cached_property
    def train_loader(self, batch_size) -> DataLoader:
        """Get the training loader.

        Returns:
            Training loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False, # Mutually exclusive with sampler option
            generator=self._generator,
            collate_fn=self.train_transform.get_collate_fn(self),
            sampler=self.sampler
        )

    # @cached_property
    # def train_loader(self) -> DataLoader:
    #     """Get the training loader.

    #     Returns:
    #         Training loader.
    #     """
    #     return DataLoader(
    #         self.train_dataset,
    #         shuffle=False, # Mutually exclusive with sampler option
    #         generator=self._generator,
    #         collate_fn=self.train_transform.get_collate_fn(self),
    #         sampler=self.sampler
    #     )




if __name__=='__main__':
    dataset_config_path = "/path/to/dataset.yaml"
    trainer_config_path = "/path/to/config.yaml"
    dataset_config = OmegaConf.load(dataset_config_path)
    trainer_config = OmegaConf.load(trainer_config_path)

    print(trainer_config)
    print(trainer_config.hydra.sweeper.params['+seed'])
    print("\n")
    print("Transform:\n", dataset_config.transform.train)
    sc = SmartCompose([RandomCrop(301, -2)])
    print("SC:\n", sc)
    # exit()
    # Instantiate the dataset
    try:
        dataset = BalancedSamplingDataset(
            path=dataset_config.path,
            features_subdir=dataset_config.features_subdir,
            seed=trainer_config.hydra.sweeper.params['+seed'],
            metrics=dataset_config.metrics,
            tracking_metric=dataset_config.tracking_metric,
            index_column=dataset_config.index_column,
            target_column=dataset_config.target_column,
            file_type=dataset_config.file_type,
            file_handler=dataset_config.file_handler,
            # batch_size=trainer_config.hydra.sweeper.params['+batch_size'],
            # batch_size=8,
            # inference_batch_size=trainer_config.inference_batch_size,
            train_transform=sc,
            dev_transform=None,  # Adjust as needed
            test_transform=None,  # Adjust as needed
            stratify=None,  # Adjust as needed
            threshold=0.5,  # Adjust as needed
        )
        print("Dataset initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        exit()
    
    dataloader = dataset.train_loader
    # Test the train loader
    try:
        train_loader = dataset.train_loader
        for idx, batch in enumerate(train_loader):
            print(f"Batch received with shape {batch}")
            if idx > 3:
                break 
        print("Train loader works as expected")
    except Exception as e:
        print(f"Error while testing train loader: {e}")
