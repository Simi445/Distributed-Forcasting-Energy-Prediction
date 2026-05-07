import pandas as pd
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score
from typing import Dict, List, Tuple, Any, Optional
from classes import Metrics, SeasonConfig, Mask, Organizer, ModelMangerAbstract, FactoryProvider, ModelFactory
from my_logger_pkg import my_logger
import numpy as np
from classes import Model


class MetricsImpl(Metrics):
    @staticmethod
    def get_metrics(pred: pd.Series, test: pd.Series) -> Dict[str, float]:
        my_logger.debug(f"Calculating metrics for {len(pred)} predictions")
        rmse: float = root_mean_squared_error(test, pred)
        mae: float = mean_absolute_error(test, pred)
        nmae: float = (mae / (test.mean() + 1e-9))
        r_2: float = r2_score(test, pred)
        my_logger.info(f"Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, NMAE={nmae:.4f}, R2={r_2:.4f}")
        return {"rmse": rmse, "mae": mae, "nmae": nmae, "r_2": r_2}
    
class SeasonMask(Mask):
    @staticmethod
    def get_season_mask(dataframe:pd.DataFrame, season:str,possible_seasons: Optional[SeasonConfig] = None)-> np.ndarray:
        my_logger.debug(f"Getting season mask for '{season}' on dataframe with {len(dataframe)} rows")
        if possible_seasons is None:
            possible_seasons = SeasonConfig()
        allowed_seasons: Dict[str, List[int]] = possible_seasons.seasonMap
        if season not in allowed_seasons:
            my_logger.error(f"Invalid season '{season}'. Allowed: {list(allowed_seasons.keys())}")
            raise ValueError(f"{season} not in the set of possible seasons!")
        idx = dataframe.index
        mask = idx.month.isin(allowed_seasons[season])
        my_logger.debug(f"Season mask for '{season}': {mask.sum()} rows selected")
        return mask
    
class DataOrganizer(Organizer):
    def __init__(self):
        final_buildings, per_building = Organizer.initialize_organizer()
        my_logger.info(f"Initializing DataOrganizer with {len(final_buildings)} buildings")
        self.final_buildings: Dict[str,int] = final_buildings
        self.per_building: Dict[int,pd.DataFrame] = per_building
        self.dataset_b = {}
        self._train_test_sepparation()
        my_logger.info("DataOrganizer initialization complete")

    def _two_fold_selection(self, dataset: pd.DataFrame) -> List[Tuple[Any, Any, Any]]:
        data:pd.DataFrame = dataset
        data = data.sort_index()
        minDate, maxDate = data.index.min(), data.index.max()
        my_logger.debug(f"Two-fold selection: date range {minDate} to {maxDate}")
        start:pd.Timestamp = minDate + pd.DateOffset(months=8)
        splits = []
        for val_start in pd.date_range(start=start, end=maxDate, freq='MS'):
            train_end:pd.Timestamp = val_start - pd.Timedelta(hours=1)
            valid_end:pd.Timestamp = (val_start+pd.offsets.MonthEnd()) + pd.Timedelta(hours=23)
            if valid_end > maxDate:
                break
            splits.append((train_end,val_start,valid_end))
        selected_splits = [splits[1],splits[-1]]
        my_logger.debug(f"Selected 2 folds from {len(splits)} possible splits")
        for i, (train_end, val_start, val_end) in enumerate(selected_splits):
            my_logger.info(f"  Fold {i}: train_end={train_end}, val_start={val_start}, val_end={val_end}")
        return selected_splits
    
    def _train_test_sepparation(self) -> None:
        if self.dataset_b:
            my_logger.info("Dataset for each building already initialized")
            return
        
        my_logger.info(f"Starting train/test separation for {len(self.final_buildings)} buildings")
        for _, b in self.final_buildings.items():
            intervals: List[Tuple[Any, Any, Any]] = self._two_fold_selection(self.per_building[b])
            self.dataset_b[b] = {}
            
            for idx, (train_end, val_start, val_end) in enumerate(intervals):
                data: pd.DataFrame = self.per_building[b]
                train_data: pd.DataFrame = data[:train_end]
                val_data: pd.DataFrame = data[val_start:val_end]
                season_name = "autumn" if idx == 0 else "winter"
                self.dataset_b[b][season_name] = {
                    'train': train_data,
                    'test': val_data
                }
                my_logger.debug(f"Building {b} - {season_name}: train={len(train_data)}, test={len(val_data)} samples")
        my_logger.info("Train/test separation complete")

    def get_dataset_b(self) -> Dict:
        my_logger.debug(f"Returning dataset_b with {len(self.dataset_b)} buildings")
        return self.dataset_b
    
    def get_per_building(self) -> Dict:
        my_logger.debug(f"Returning per_building with {len(self.per_building)} buildings")
        return self.per_building
    
class BaseLineDataOrganizer(DataOrganizer):
    def __init__(self):
        my_logger.info("Initializing BaseLineDataOrganizer")
        super().__init__()

    def prepare_train_test(self) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        res: Dict = {}
        for _,b in self.final_buildings.items():
            my_logger.info(f"Preparing train/test data for baseline model, build_id={b}")
            dataset: pd.DataFrame = self.per_building[b]
            y = dataset["meter_reading"].sort_index()
            all_test = []
            for season in self.dataset_b[b]:
                test_data = self.dataset_b[b][season]['test']['meter_reading']
                all_test.append(test_data)
                my_logger.debug(f"Added {len(test_data)} test samples from {season}")
            y_test = pd.concat(all_test).sort_index()
            my_logger.info(f"Baseline data prepared: y={len(y)}, y_test={len(y_test)} samples")
            res[b] = (y, y_test)
            my_logger.info(f"Baseline data appended for build_id: {b}")
        return res
    
class ModelManager(ModelMangerAbstract):
    def __init__(self):
        self.models: Dict[Tuple, Model] = {}
    
    def add_model(self, key: Tuple, model: Model) -> None:
        if key in self.models:
            return
        self.models[key] = model
    
    def get_model(self, key: Tuple) -> Model:
        if key not in self.models:
            raise ValueError("Model was not added to manager!")
        return self.models[key]

class DefaultFactoryProvider(FactoryProvider):
      def __init__(self, factories: dict[str, ModelFactory]):
          self._factories = factories

      def get_factory(self, model_name: str) -> ModelFactory:
          try:
              return self._factories[model_name]
          except KeyError as e:
              raise ValueError(f"Unknown model: {model_name}") from e