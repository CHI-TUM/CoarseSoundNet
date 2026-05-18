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

class CoarseSamplingDataset(BaseMLClassificationDataset):
    def __init__(
        self,
        path: str,
        test_paths: List[str],
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
        # Apparently I need to call these dataframes in order for the method function to be executed?
        # self.df_train = self.df_train
        # self.df_dev = self.df_dev
        # self.df_test = self.df_test

    
    # Keep the train/dev with the simple structure, where we have already defined train and dev csvs.
    @cached_property
    def df_train(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.path, "train.csv"))

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.path, "dev.csv"))


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
                domain = "BE"
            elif "esc50" in csv_path: # esc50 data
                prependix = "/path/to/data/test_sets/esc50_test/log_mel_32k/"
                domain = "esc50"
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


class BESoundv1(CoarseSamplingDataset):
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
        return self._df.loc[
            self._df["plot"].isin(self._df["plot"].value_counts().index[:100])
        ].reset_index()

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        return self._df.loc[
            self._df["plot"].isin(self._df["plot"].value_counts().index[100:120])
        ].reset_index()

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
        dataset = CoarseBalancedSamplingDataset(
            path=dataset_config.path,
            test_paths=dataset_config.test_paths,
            features_subdir=dataset_config.features_subdir,
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
