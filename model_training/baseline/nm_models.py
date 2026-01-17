import pandas as pd
from .my_logger_pkg import my_logger
from typing import Optional, Dict, Tuple, Callable,Any
from functools import reduce

from ..classes import BaselineModel, NaiveModel, SeasonConfig,Organizer,Baseline,BaseLineModelConfig
from model_training.baseline.auxiliary import BaseLineDataOrganizer,SeasonMask,MetricsImpl


class BaselinePredictor(BaselineModel):
    
    def __init__(self, final_buildings: Dict[str, int], per_building: Dict[int, pd.DataFrame]):
        self.final_buildings = final_buildings
        self.per_building = per_building
        
        self.dataset_b: Dict[int, Dict[str, Any]] = {}
        self.data_preparator:Organizer = BaseLineDataOrganizer(final_buildings, per_building)
    
    def get_train_test(self, build_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.data_preparator.prepare_train_test(build_id)
    
    def _apply_seasonal_filter(self, y_pred: pd.DataFrame, y_test: pd.DataFrame, 
                                season: str, possible_seasons: SeasonConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if season not in possible_seasons.seasonMap:
            raise ValueError(f"Incorrect season, available: {list(possible_seasons.seasonMap.keys())}")
        
        m = SeasonMask.get_season_mask(y_test, season, possible_seasons)
        return y_pred[m], y_test[m]


class NaiveImpl(BaselinePredictor):
    
    def baseline_persistance_model(self, build_id: int, season: str, possible_seasons: SeasonConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:

        y, y_test = self.get_train_test(build_id)
        y_hat = y.shift(24).loc[y_test.index]
        return self._apply_seasonal_filter(y_hat, y_test, season, possible_seasons)


class MeanImpl(BaselinePredictor):
    
    def baseline_mean_model(self, build_id: int, season: str, possible_seasons: SeasonConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        y, y_test = self.get_train_test(build_id)
        
        dataframes = [y.shift(i).loc[y_test.index] for i in range(24, 169, 24)]
        
        y_hat = reduce(lambda x, z: x + z, dataframes) / len(dataframes)
        
        return self._apply_seasonal_filter(y_hat, y_test, season, possible_seasons)


class BaselineTrainer:
    def __init__(self, final_buildings: Dict[str, int], per_building: Dict[int, pd.DataFrame], models:BaseLineModelConfig):
        self.final_buildings = final_buildings
        self.per_building = per_building
        self.model_config = models
    
    def train_and_evaluate(self, model_name: str, build_id: int, season: str, 
                          possible_seasons: SeasonConfig, method_name: str) -> Dict[str, float]:

        if model_name not in self.model_config.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
        
        model = self.model_config.models[model_name]
        method = getattr(model, method_name)
        
        y_pred, y_test = method(build_id, season, possible_seasons)
        metrics = MetricsImpl.get_metrics(y_pred, y_test)
        
        my_logger.info(f"{model_name} model metrics for building {build_id}, {season}: {metrics}")
        return metrics
    
