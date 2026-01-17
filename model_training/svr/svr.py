import pandas as pd
from .my_logger_pkg import my_logger
from typing import Dict, Tuple, List, Any
from ..seasonality import NoSeasonalityStrategy
from ..classes import Model, SeasonalityStrategy, XGGridParamsConfig, ModelConfigparams, AnalysisConfig
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from ..helpers import TrainTestCache, TrainerImpl
import numpy as np



class SVR(Model):
    def __init__(self,config: ModelConfigparams, 
                 dataset_b: Dict, seasonality_strategy: SeasonalityStrategy = NoSeasonalityStrategy()):
        self.dataset_b: Dict = dataset_b
        self.seasonality_strategy: SeasonalityStrategy = seasonality_strategy
        self.n_jobs = config.n_jobs
        self.k_fold = config.k_fold
        my_logger.info(f"Initializing SVR with k_fold={config.k_fold}")
        
        self._tt_cache = TrainTestCache()
        self._trainer = TrainerImpl()
       
    def _calculate_train_std(self, y_train: pd.Series)->float:
        return np.std(y_train)

    def make_param_grid(self, y_train: pd.Series) -> None:
        train_std:float = self._calculate_train_std(y_train)
        self.param_grid = {
            "epsilon": [0.1 * train_std],
            "C": [10 * train_std],
            "kernel": ['rbf']
            }

    def _make_grid(self) -> GridSearchCV:
        if not self.param_grid:
            raise ValueError("For SVR, please call the make_param_grid method first!")
        return GridSearchCV(SVR(),self.param_grid, cv=TimeSeriesSplit(n_splits=self.k_fold),
          return_train_score=False, n_jobs=self.n_jobs)
    

    def train_test_fn(self, build_id: int, fold_key: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        analysis = AnalysisConfig(
            build_id=build_id,
            fold_key=fold_key,
            dataset_b=self.dataset_b,
            strategy=self.seasonality_strategy
        )
        return self._tt_cache.get(analysis)
    

    def predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, grid_svr:GridSearchCV) -> pd.Series:
        analysis = AnalysisConfig(grid=grid_svr)
        return self._trainer.fit_predict(analysis, X_train, y_train, X_test, y_test)


    
