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


class Edansav2(BaseMLClassificationDataset):
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
        additional_train_domains: List[Dict] = None,
        test_paths: List[Dict] = None
    ) -> None:
        self.additional_train_domains = additional_train_domains
        self.test_paths = test_paths
        self.selected_columns = [index_column] + target_column
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
        

    def _load_additional_df(self, domain, path, split) -> pd.DataFrame:
        """
        path: Path to the dataset directory containing the train, dev, and test csvs.
        split: The name of the csv, e.g. "train.csv"
        """
        df = pd.read_csv(os.path.join(path, split))
        df = df[self.selected_columns]
        df["domain"] = domain
        # Make paths absolute in order to not use filepaths relative to path for the additional data.
        df["filename"] = df["filename"].apply(lambda x: os.path.join(path, self.features_subdir, x))
        return df


    @cached_property
    def df_train(self) -> pd.DataFrame:
        # Load the edansa data
        df_train = pd.read_csv(os.path.join(self.path, "train.csv"))
        df_train = df_train.rename(columns={"Clip Path": "filename"})
        df_train = df_train[self.selected_columns]
        df_train["domain"] = "edansa"
        print("Edansa: ", df_train.shape)
        print(df_train.head())

        # Load the data from the additional domains
        if self.additional_train_domains:
            additional_dfs = []
            print("Use the following domains for train: ", self.additional_train_domains)
            for domain, path in self.additional_train_domains.items():
                df = self._load_additional_df(domain, path, "train.csv")
                print(f"\n{domain}: ", df.shape)
                print(df.head())
                additional_dfs.append(df)
            combined_df = pd.concat(additional_dfs).reset_index(drop=True)
            df_train = pd.concat([df_train, combined_df]).reset_index(drop=True)
        print("df_train.shape: ", df_train.shape)
        return df_train


    @cached_property
    def df_dev(self) -> pd.DataFrame:
        df_dev = pd.read_csv(os.path.join(self.path, "dev.csv"))
        df_dev = df_dev.rename(columns={"Clip Path": "filename"})
        df_dev = df_dev[self.selected_columns]
        df_dev["domain"] = "edansa"
        print("Edansa: ", df_dev.shape)

        # Load the data from the additional domains
        if self.additional_train_domains:
            additional_dfs = []
            print("Use the following domains for dev: ", self.additional_train_domains)
            for domain, path in self.additional_train_domains.items():
                df = self._load_additional_df(domain, path, "dev.csv")
                print(f"{domain}: ", df.shape)
                additional_dfs.append(df)
            combined_df = pd.concat(additional_dfs).reset_index(drop=True)
            df_dev = pd.concat([df_dev, combined_df]).reset_index(drop=True)
        print("df_dev.shape: ", df_dev.shape)
        return df_dev


    @cached_property
    def df_test(self) -> pd.DataFrame:
        print("\nTest:")
        df_test = pd.read_csv(os.path.join(self.path, "test.csv"))
        df_test = df_test.rename(columns={"Clip Path": "filename"})
        df_test = df_test[self.selected_columns]
        df_test["domain"] = "edansa"
        print("Edansa: ", df_test.shape)

        if self.test_paths:
            additional_dfs = []
            print("Use the following additional domains for test: ", self.test_paths)
            for domain, path in self.test_paths.items():
                df = self._load_additional_df(domain, path, "test.csv")
                print(f"{domain}: ", df.shape)
                additional_dfs.append(df)
            combined_df = pd.concat(additional_dfs).reset_index(drop=True)
        
            df_test = pd.concat([df_test, combined_df]).reset_index(drop=True)
        print("df_test.shape: ", df_test.shape)
        return df_test



class Edansav2Balanced(Edansav2):
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
        additional_train_domains: List[Dict] = None,
        test_paths: List[Dict] = None
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
            additional_train_domains=additional_train_domains,
            test_paths=test_paths
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
        # dataset = CoarseBalancedSamplingDataset(
        dataset = Edansav2Balanced(
            path=dataset_config.path,
            test_paths=dataset_config.test_paths,
            features_subdir=dataset_config.features_subdir,
            additional_train_domains=dataset_config.additional_train_domains,
            seed=trainer_config.hydra.sweeper.params['+seed'],
            metrics=dataset_config.metrics,
            tracking_metric=dataset_config.tracking_metric,
            index_column=dataset_config.index_column,
            target_column=dataset_config.target_column,
            file_type=dataset_config.file_type,
            file_handler=dataset_config.file_handler,
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
    
    # dataloader = dataset.train_loader(batch_size=4)

    train_loader = DataLoader(
        dataset.train_dataset,
        batch_size=4,
        shuffle=False, # Mutually exclusive with sampler option
        generator=dataset._generator,
        collate_fn=dataset.train_transform.get_collate_fn(self),
        sampler=dataset.sampler
    )

    # Test the train loader
    try:
        train_loader = dataset.train_loader(batch_size=4)
        for idx, batch in enumerate(train_loader):
            print(f"Batch received with shape {batch}")
            if idx > 3:
                break 
        print("Train loader works as expected")
    except Exception as e:
        print(f"Error while testing train loader: {e}")