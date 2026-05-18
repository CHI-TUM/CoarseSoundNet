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


selected_columns = ["filename", "Anth", "Bio", "Geo", "Sil"]
supercategories = ["Anth", "Bio", "Geo", "Sil"]

class CoarseBalancedSamplingDataset(BaseMLClassificationDataset):
    def __init__(
        self,
        path: str,
        test_paths: List[str],
        additional_train_domains: Dict[str, str],
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: List[str],
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        self.test_paths=test_paths
        self.additional_train_domains=additional_train_domains
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

        # Get label frequencies and weights
        self.multilabels = self.df_train[target_column]
        self.num_samples = len(self.df_train)

        self.label_counts = self.multilabels.sum(axis=0)
        self.label_frequencies = self.label_counts / self.num_samples
        self.label_weights = 1.0 / self.label_frequencies

        # TODO: Do weighted sampling based on domain instead of classes (maybe both as well)
        # Determine the sample weights
        # For this, take the maximum weight of all active labels for a sample
        print("Initializing the weighted sampler...")
        self.sample_weights = torch.tensor((self.multilabels * self.label_weights).max(axis=1), dtype=torch.float)
        self.sampler = WeightedRandomSampler(weights=self.sample_weights, num_samples=self.num_samples, replacement=True, generator=self._generator)
        print("Initialized.")

    
    # Keep the train/dev with the simple structure, where we have already defined train and dev csvs.
    @cached_property
    def df_train(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.path, "train.csv"))

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.path, "dev.csv"))

    @cached_property
    def df_test(self) -> pd.DataFrame:
        df_test = pd.read_csv(os.path.join(self.path, "test.csv"))

        # If there are other test_paths specified, combine the test set from path with the other test sets.
        if self.test_paths is not None:
            print("Use the following test csvs: ", self.test_paths)
            df_test["domain"] = "mixed"
            df_test[supercategories] = df_test[supercategories].astype(int)
            df_all_tests = self._all_df_test
            df_test = pd.concat([df_test, df_all_tests]).reset_index(drop=True)

        return df_test
    

    # Combine all test set csvs into one big dataframe for testing.
    # TODO: Check how the filepaths are handled...
    @cached_property
    def _all_df_test(self) -> pd.DataFrame:
        print(">>> Instantiate all test dataframes")
        # Gather the various test csvs into one combined test dataframe.
        df_list = []
        for csv_path in self.test_paths:
            print("CSV: ", csv_path)

            if "sandra_dominik" in csv_path: # Akwamo
                prependix = "/path/to/data/sandra_dominik/log_mel_32k/"
                domain = "akwamo"
            elif "edansa" in csv_path: # Edansa
                prependix = "/path/to/data/edansa_only/log_mel_32k/"
                domain = "edansa"
            elif "BE_data" in csv_path: # BE-test data
                prependix = "/path/to/data/BE_data/log_mel_32k/segments/"
                domain = "BESound"
            elif "esc50" in csv_path: # esc50 data
                prependix = "/path/to/data/test_sets/esc50_test/log_mel_32k/"
                domain = "esc50"
            elif "synthetic" in csv_path:
                prependix = "/path/to/data/synthetic_data/log_mel_32k/"
                domain = "mixed"
            else: # maybe raise an Exception
                prependix = ""
                domain = "unknown"
            
            df = pd.read_csv(csv_path)
            df = df[selected_columns]
            df["filename"] = df["filename"].apply(lambda x: os.path.join(prependix, x))
            df["domain"] = domain
            df[supercategories] = df[supercategories].astype(int)
            # print(df.head())
            df_list.append(df)
        
        df_combined = pd.concat(df_list, axis=0).reset_index(drop=True)
        # print(df_combined)
        return df_combined


    @cached_property
    def train_loader(self, batch_size) -> DataLoader:
        """Get the training loader.

        Returns:
            Training loader.
        """
        print(">>> Using the weighted random sampler for the dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False, # Mutually exclusive with sampler option
            generator=self._generator,
            collate_fn=self.train_transform.get_collate_fn(self),
            sampler=self.sampler
        )


class BESoundv1(CoarseBalancedSamplingDataset):
    
    def _load_additional_df(self, domain, path, split) -> pd.DataFrame:
        """
        path: Path to the dataset directory containing the train, dev, and test csvs.
        split: The name of the csv, e.g. "train.csv"
        """
        df = pd.read_csv(os.path.join(path, split))
        df = df[selected_columns]
        df["domain"] = domain
        # Make paths absolute in order to not use filepaths relative to path for the additional data.
        df["filename"] = df["filename"].apply(lambda x: os.path.join(path, self.features_subdir, x))
        return df


    @cached_property
    def _df(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.path, "Annotations_BEForest_segments_one-hot.csv")
        )
        df["plot"] = df["filename"].apply(lambda x: x.split("_")[1])
        df["filename"] = df["filename"].apply(lambda x: os.path.join("segments", x))
        return df


    @cached_property
    def df_train(self) -> pd.DataFrame:
        # Load the besound data
        be_train = self._df.loc[
            self._df["plot"].isin(self._df["plot"].value_counts().index[:100])
        ].reset_index()

        # Load the data from the additional domains
        if self.additional_train_domains:
            additional_dfs = []
            for domain, path in self.additional_train_domains.items():
                df = self._load_additional_df(domain, path, "train.csv")
                additional_dfs.append(df)
            combined_df = pd.concat(additional_dfs).reset_index(drop=True)
            be_train = pd.concat([be_train, combined_df]).reset_index(drop=True)
        
        return be_train


    @cached_property
    def df_dev(self) -> pd.DataFrame:
        be_dev = self._df.loc[
            self._df["plot"].isin(self._df["plot"].value_counts().index[100:120])
        ].reset_index()

        # Load the data from the additional domains
        if self.additional_train_domains:
            additional_dfs = []
            for domain, path in self.additional_train_domains.items():
                df = self._load_additional_df(domain, path, "dev.csv")
                additional_dfs.append(df)
            combined_df = pd.concat(additional_dfs).reset_index(drop=True)
            be_dev = pd.concat([be_dev, combined_df]).reset_index(drop=True)
        
        return be_dev


    @cached_property
    def df_test(self) -> pd.DataFrame:
        df_test = self._df.loc[
            self._df["plot"].isin(self._df["plot"].value_counts().index[120:])
        ].reset_index()

        if self.test_paths is not None:
            print("Use the following test csvs: ", self.test_paths)
            df_test["domain"] = "BESound"
            df_test[supercategories] = df_test[supercategories].astype(int)
            df_all_tests = self._all_df_test
            df_test = pd.concat([df_test, df_all_tests]).reset_index(drop=True)

        return df_test

class EDANSAv1(CoarseBalancedSamplingDataset):
    """
    Uses Edansa as "base" dataset and adds additional train/dev data from other specified locations.
    """
    def __init__( 
        self,
        path: str,
        test_paths: List[str],
        additional_train_domains: Dict[str, str],
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: List[str],
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        super().__init__(
            path=path,
            features_subdir=features_subdir,
            test_paths=test_paths,
            additional_train_domains=additional_train_domains,
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

        # TODO: Make this optional! If not chosen, use normal label-based weighted sampling.
        # Get domain frequencies and weights
        self.domains = self.df_train["domain"]
        self.num_samples = len(self.df_train)

        # Count the occurrences of each domain
        self.domain_counts = self.domains.value_counts()
        self.domain_frequencies = self.domain_counts / self.num_samples

        # Compute the inverse frequency weights for each domain
        self.domain_weights = 1.0 / self.domain_frequencies

        # Map the weights back to each sample based on its domain
        self.sample_weights = self.domains.map(self.domain_weights).values
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float)

        print("Initializing the weighted domain sampler...")
        self.sampler = WeightedRandomSampler(
            weights=self.sample_weights, 
            num_samples=self.num_samples, 
            replacement=True, 
            generator=self._generator
        )
        print("Initialized.")
    
    
    def _load_additional_df(self, domain, path, split) -> pd.DataFrame:
        """
        path: Path to the dataset directory containing the train, dev, and test csvs.
        split: The name of the csv, e.g. "train.csv"
        """
        df = pd.read_csv(os.path.join(path, split))
        df = df[selected_columns]
        df["domain"] = domain
        # Make paths absolute in order to not use filepaths relative to path for the additional data.
        df["filename"] = df["filename"].apply(lambda x: os.path.join(path, self.features_subdir, x))
        return df


    @cached_property
    def df_train(self) -> pd.DataFrame:
        # Load the edansa data
        df_train = pd.read_csv(os.path.join(self.path, "train.csv"))
        df_train["domain"] = "edansa"

        # Load the data from the additional domains
        if self.additional_train_domains:
            additional_dfs = []
            for domain, path in self.additional_train_domains.items():
                df = self._load_additional_df(domain, path, "train.csv")
                additional_dfs.append(df)
            combined_df = pd.concat(additional_dfs).reset_index(drop=True)
            df_train = pd.concat([df_train, combined_df]).reset_index(drop=True)
        
        return df_train


    @cached_property
    def df_dev(self) -> pd.DataFrame:
        df_dev = pd.read_csv(os.path.join(self.path, "dev.csv"))
        df_dev["domain"] = "edansa"

        # Load the data from the additional domains
        if self.additional_train_domains:
            additional_dfs = []
            for domain, path in self.additional_train_domains.items():
                df = self._load_additional_df(domain, path, "dev.csv")
                additional_dfs.append(df)
            combined_df = pd.concat(additional_dfs).reset_index(drop=True)
            df_dev = pd.concat([df_dev, combined_df]).reset_index(drop=True)
        
        return df_dev


    @cached_property
    def df_test(self) -> pd.DataFrame:
        df_test = pd.read_csv(os.path.join(self.path, "test.csv"))
        df_test["domain"] = "edansa"

        if self.test_paths is not None:
            print("Use the following test csvs: ", self.test_paths)
            df_test["domain"] = "Edansa"
            df_test[supercategories] = df_test[supercategories].astype(int)
            df_all_tests = self._all_df_test
            df_test = pd.concat([df_test, df_all_tests]).reset_index(drop=True)

        return df_test



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
        dataset = EDANSAv1(
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
    
    dataloader = dataset.train_loader(batch_size=4)
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
