import os
import shutil
import json
import ast
import pandas as pd
from typing import Dict, List, Optional, Tuple, TypeVar, Union
from omegaconf import OmegaConf, DictConfig
from autrainer.datasets.abstract_dataset import BaseMLClassificationDataset
from autrainer.datasets.utils import ZipDownloadManager
import torch
from autrainer.transforms import RandomCrop, SmartCompose
from torch.utils.data import DataLoader
from functools import cached_property


def convert_json_to_csv(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert the json files to csv files.
    """
    splits = ["train", "val", "test"]
    dfs = {}
    for split in splits:
        print("\nSplit: ", split)
        fp = os.path.join(path, f"{split}.json")

        with open(fp, "r") as f:
            data = json.load(f)
        
        categories_df = pd.DataFrame(data['categories'])
        audio_df = pd.DataFrame(data['audio'])
        annotations_df = pd.DataFrame(data['annotations'])

        # Merge the sub dataframes into one single dataframe
        annotations_audio_df = pd.merge(annotations_df, audio_df, left_on="audio_id", right_on="id", suffixes=("_annotation", "_audio"))
        full_df = pd.merge(annotations_audio_df, categories_df, left_on="category_id", right_on="id", suffixes=("_audio", "_category"))
        full_df.drop(columns=['id_audio', 'id'], inplace=True)
        full_df.rename(columns={'file_name': 'filename'}, inplace=True)
        # print("\n")
        # print(full_df.head())
        # print(full_df.columns)
        if split == "val":
            split = "dev"

        dfs[split] = full_df
        full_df.to_csv(os.path.join(path, f"df_{split}.csv"), index=False)

    print("Dfs: ", dfs)
    return dfs['train'], dfs['dev'], dfs['test']




class INATURALIST(BaseMLClassificationDataset):
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
        batch_size: int,
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        threshold: float = 0.5,
        label_column: str = "supercategory", # The column which comprises all the target classes which will then form the target_column attribute.
    ) -> None:
        self.label_column = label_column
        # TODO: Right now the target_column still must be handed over to the class. It would be good to directly get them from the train set, based on the label_column.
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
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
            threshold=threshold,
        )


    def load_dataframes(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the csv files for train, dev, and test. 
        Then convert them to a one-hot encoded version.
        """
        if not os.path.exists(os.path.join(self.path, "df_train.csv")):
            df_train, df_dev, df_test = convert_json_to_csv(self.path)
        else:
            df_train = pd.read_csv(os.path.join(self.path, "df_train.csv"))
            df_dev = pd.read_csv(os.path.join(self.path, "df_dev.csv"))
            df_test = pd.read_csv(os.path.join(self.path, "df_test.csv"))
        
        if self.target_column is None:
            self.target_column = df_train[self.label_column].unique().tolist()
        df_train = self.convert_to_multilabel(df=df_train)
        df_dev = self.convert_to_multilabel(df=df_dev)
        df_test = self.convert_to_multilabel(df=df_test)

        return df_train, df_dev, df_test
    

    def convert_to_multilabel(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Converts the dataframe to a one-hot encoded multi-label version based on the target_column column.
        However, the iNaturalits dataset is weakly-labelled, i.e., there is only one label per audio file.
        """
        new_cols = pd.DataFrame(
            [[1 if col == name else 0 for col in self.target_column] for name in df[self.label_column]],
            columns=self.target_column
        )
        df = pd.concat([df, new_cols], axis=1)
        df = df[['id_annotation', 'filename'] + self.target_column]
        return df

    

    


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

    try:
        dataset = INATURALIST(
            # path=dataset_config.path,
            path="/path/to/iNaturalist",
            features_subdir=dataset_config.features_subdir,
            seed=trainer_config.hydra.sweeper.params['+seed'],
            metrics=dataset_config.metrics,
            tracking_metric=dataset_config.tracking_metric,
            index_column=dataset_config.index_column,
            # target_column=dataset_config.target_column,
            target_column=['Aves', 'Amphibia', 'Insecta', 'Mammalia', 'Reptilia'],
            file_type=dataset_config.file_type,
            file_handler=dataset_config.file_handler,
            # batch_size=trainer_config.hydra.sweeper.params['+batch_size'],
            batch_size=8,
            inference_batch_size=trainer_config.inference_batch_size,
            train_transform=sc,
            dev_transform=None,  # Adjust as needed
            test_transform=None,  # Adjust as needed
            stratify=None,  # Adjust as needed
            threshold=0.5,  # Adjust as needed,
            label_column="supercategory",
        )
        print("Dataset initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        exit()
    
    train_set = dataset.df_train
    print("Train set:\n", train_set)
    dev_set = dataset.df_dev
    print("Dev set:\n", dev_set)
    test_set = dataset.df_test
    print("Test set:\n", test_set)