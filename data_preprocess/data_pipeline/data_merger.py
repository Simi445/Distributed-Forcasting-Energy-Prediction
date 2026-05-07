import pandas as pd
from ..my_logger_pkg import my_logger
from typing import Optional
from ..classes import Merger, Loader
import numpy as np

class DataMerger(Merger):
    def __init__(self, data_loader: Loader):
        self.data_obj = data_loader
        self.full_df: Optional[pd.DataFrame] = None

    def merge_tables(self)->None:
        if self.full_df is not None:
            my_logger.info("Data already merged, skipping merge operation")
            return
        
        try:
            train: pd.DataFrame = self.data_obj.get_dataset(data_category="train")
            train = train[train.meter == 0]

            train: pd.DataFrame = train.drop('meter', axis=1)

            building_set: np.ndarray = train['building_id'].unique()

            meta: pd.DataFrame = self.data_obj.get_dataset(data_category="meta")

            meta['floor_missing'] = meta['floor_count'].isna().astype('int8')
            meta['year_missing'] = meta['year_built'].isna().astype('int8')

            meta['year_built'] = meta['year_built'].fillna(meta['year_built'].median())
            meta['year_built'] = meta['year_built'].astype('int16')

            meta['floor_count'] = meta['floor_count'].fillna(meta['floor_count'].median())
            meta['floor_count'] = meta['floor_count'].astype('int16')

            meta: pd.DataFrame = pd.get_dummies(meta, columns=['primary_use'], drop_first=False)

            meta: pd.DataFrame = meta[meta['building_id'].isin(building_set)]


            site_ids: np.ndarray = meta['site_id'].unique()

            weather: pd.DataFrame = self.data_obj.get_dataset(data_category="weather")
            weather: pd.DataFrame = weather[weather['site_id'].isin(site_ids)]

            train_meta: pd.DataFrame = pd.merge(train, meta, on='building_id', how='left')
            self.full_df: pd.DataFrame = pd.merge(train_meta, weather, on=['site_id','timestamp'], how='left')
            my_logger.info(f"Data merged successfully. Shape: {self.full_df.shape}")
            my_logger.info(self.full_df.head(80))
        except KeyError as e:
            my_logger.exception(f"KeyError occurred during merge - missing column: {e}")
            raise ValueError(f"Required column missing in dataset: {e}") from e
        except Exception as e:
            my_logger.exception(f"Exception occurred in merging tables: {e}")
            raise

    def get_full_df(self) -> pd.DataFrame:
        if self.full_df is not None:
            return self.full_df
        raise ValueError("Requested df has no entries.")