import pandas as pd
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score
from typing import Dict, List, Tuple, Any
from .classes import Metrics, SeasonConfig, Mask, Organizer
from .my_logger_pkg import my_logger
import numpy as np

class MetricsImpl(Metrics):
    @staticmethod
    def get_metrics(pred: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
        rmse: float = root_mean_squared_error(test, pred)
        mae: float = mean_absolute_error(test, pred)
        nmae: float = (mae / (test.mean() + 1e-9))
        r_2: float = r2_score(test, pred)
        return {"rmse": rmse, "mae": mae, "nmae": nmae, "r_2": r_2}
    
class SeasonMask(Mask):
    @staticmethod
    def get_season_mask(dataframe:pd.DataFrame, season:str, possible_seasons:SeasonConfig)-> np.ndarray:
        allowed_seasons: Dict[str, List[int]] = possible_seasons.seasonMap
        if season not in allowed_seasons:
            raise ValueError(f"{season} not in the set of possible seasons!")
        idx = dataframe.index
        return idx.month.isin(allowed_seasons[season])
    
class DataOrganizer(Organizer):
    def __init__(self):
        self.dataset_b = {}
        self._train_test_sepparation()

    def _two_fold_selection(self, dataset: pd.DataFrame) -> List[Tuple[Any, Any, Any]]:
        data:pd.DataFrame = dataset
        data = data.sort_index()
        minDate, maxDate = data.index.min(), data.index.max()
        start:pd.Timestamp = minDate + pd.DateOffset(months=8)
        splits = []
        for val_start in pd.date_range(start=start, end=maxDate, freq='MS'):
            train_end:pd.Timestamp = val_start - pd.Timedelta(hours=1)
            valid_end:pd.Timestamp = (val_start+pd.offsets.MonthEnd()) + pd.Timedelta(hours=23)
            if valid_end > maxDate:
                break
            splits.append((train_end,val_start,valid_end))
        return [splits[1],splits[-1]]
    
    def _train_test_sepparation(self) -> None:
        if self.dataset_b:
            my_logger.info("Dataset for each building already initialized")
            return
        
        for _, b in self.final_buildings.items():
            intervals: List[Tuple[Any, Any, Any]] = self._two_fold_selection(self.per_building[b])
            self.dataset_b[b] = {}
            
            for idx, (train_end, val_start, val_end) in enumerate(intervals):
                data: pd.DataFrame = self.per_building[b]
                train_data: pd.DataFrame = data[:train_end]
                val_data: pd.DataFrame = data[val_start:val_end]
                self.dataset_b[b]["autumn" if idx == 0 else "winter"] = {
                    'train': train_data,
                    'test': val_data
                }

    def get_dataset_b(self) -> Dict:
        return self.dataset_b
    
class BaseLineDataOrganizer(DataOrganizer):
    def __init__(self, per_building: Dict[int, pd.DataFrame]):
        super().__init__()
        self.per_building = per_building
    
    def prepare_train_test(self, build_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset: pd.DataFrame = self.per_building[build_id]
        y = dataset["meter_reading"].sort_index()
        all_test = []
        for season in self.dataset_b[build_id]:
            test_data = self.dataset_b[build_id][season]['test']['meter_reading']
            all_test.append(test_data)
        y_test = pd.concat(all_test).sort_index()
        return (y, y_test)