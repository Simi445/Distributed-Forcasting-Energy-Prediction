import pandas as pd
from .my_logger_pkg import my_logger
from typing import Dict, Tuple
from ..seasonality import NoSeasonalityStrategy
from ..classes import Model, SeasonalityStrategy, XGGridParamsConfig, ModelConfigparams, AnalysisConfig
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from ..helpers import TrainTestCache, TrainerImpl




class XGBoost(Model):
    def __init__(self, param_grid: XGGridParamsConfig, config: ModelConfigparams, 
                 dataset_b: Dict, seasonality_strategy: SeasonalityStrategy = NoSeasonalityStrategy()):
        self.param_grid:XGGridParamsConfig = param_grid
        self.dataset_b: Dict = dataset_b
        self.state_seed: int = config.state_seed
        self.seasonality_strategy: SeasonalityStrategy = seasonality_strategy
        self.n_jobs = config.n_jobs
        self.k_fold = config.k_fold
        my_logger.info(f"Initializing XGBoost with state_seed={self.state_seed}, k_fold={config.k_fold}")
        
        self._tt_cache = TrainTestCache()
        self._trainer = TrainerImpl()
       
    def _make_grid(self) -> GridSearchCV:
        return GridSearchCV(XGBRegressor(
            random_state=self.state_seed,
        ),self.param_grid, cv=TimeSeriesSplit(n_splits=self.k_fold),
          return_train_score=False, n_jobs=self.n_jobs)
    

    def train_test_fn(self, build_id: int, fold_key: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        analysis = AnalysisConfig(
            build_id=build_id,
            fold_key=fold_key,
            dataset_b=self.dataset_b,
            strategy=self.seasonality_strategy
        )
        return self._tt_cache.get(analysis)
    

    def predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, grid_xg:GridSearchCV) -> pd.Series:
        analysis = AnalysisConfig(grid=grid_xg)
        return self._trainer.fit_predict(analysis, X_train, y_train, X_test, y_test)


    
