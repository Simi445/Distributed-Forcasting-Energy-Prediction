import pandas as pd
from my_logger_pkg import my_logger
from typing import Dict, Tuple
from seasonality import NoSeasonalityStrategy
from classes import (
    GridSearchModel, SeasonalityStrategy, XGGridParamsConfig, ModelConfigparams,
    TrainingContext
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit


class XGBoost(GridSearchModel):
    """XGBoost implementation using the GridSearchModel base class."""
    
    def __init__(self, param_grid: XGGridParamsConfig, config: ModelConfigparams, 
                 dataset_b: Dict, seasonality_strategy: SeasonalityStrategy = NoSeasonalityStrategy()):
        super().__init__(
            config=config,
            dataset_b=dataset_b,
            seasonality_strategy=seasonality_strategy,
            param_grid=param_grid.parameters
        )
        my_logger.info(f"Initializing XGBoost with state_seed={self.state_seed}, k_fold={self.k_fold}")
    
    def _get_model_name(self)->str:
        return self.__class__.__name__

    def _make_grid(self) -> GridSearchCV:
        return GridSearchCV(
            XGBRegressor(random_state=self.state_seed),
            self.param_grid,
            cv=TimeSeriesSplit(n_splits=self.k_fold),
            return_train_score=False,
            n_jobs=self.n_jobs
        )