import pandas as pd
from .my_logger_pkg import my_logger
from typing import Dict, Tuple
from ..seasonality import NoSeasonalityStrategy
from ..classes import TreeModel, SeasonalityStrategy, RFGridParamsConfig, RFConfig, AnalysisConfig
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from .helpers import TrainTestCache, RandomForestTrainer, FeatureAnalyzer


class RandomForest(TreeModel):
    def __init__(self, param_grid: RFGridParamsConfig, config: RFConfig, 
                 dataset_b: Dict, seasonality_strategy: SeasonalityStrategy = NoSeasonalityStrategy()):
        self.dataset_b: Dict = dataset_b
        self.state_seed: int = config.state_seed
        self.seasonality_strategy: SeasonalityStrategy = seasonality_strategy
        my_logger.info(f"Initializing RandomForest with state_seed={self.state_seed}, k_fold={config.k_fold}")
        
        self.grid_rf: GridSearchCV = GridSearchCV(
            RandomForestRegressor(random_state=self.state_seed),
            param_grid, 
            cv=TimeSeriesSplit(n_splits=config.k_fold), 
            return_train_score=False, 
            n_jobs=config.n_jobs
        )
        
        self._tt_cache = TrainTestCache()
        self._trainer = RandomForestTrainer()
        self._analyzer = FeatureAnalyzer(self._trainer)
       
    def train_test_fn(self, build_id: int, fold_key: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        analysis = AnalysisConfig(
            build_id=build_id,
            fold_key=fold_key,
            dataset_b=self.dataset_b,
            strategy=self.seasonality_strategy
        )
        return self._tt_cache.get(analysis)
    
    
    def predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
        analysis = AnalysisConfig(grid_rf=self.grid_rf)
        return self._trainer.fit_predict(analysis, X_train, y_train, X_test, y_test)


    def permutation_importance(self, build_id: int, fold_key: str, iterations: int = 10) -> Dict[str, Dict[str, float]]:
        analysis = AnalysisConfig(
            build_id=build_id,
            fold_key=fold_key,
            dataset_b=self.dataset_b,
            iterations=iterations,
            state_seed=self.state_seed,
            trainer_fn=self.train_test_fn,
            grid_rf=self.grid_rf
        )
        return self._analyzer.permutation_importance(analysis)

    def feature_importance(self, build_id: int, fold_key: str) -> pd.Series:
        analysis = AnalysisConfig(
            build_id=build_id,
            fold_key=fold_key,
            dataset_b=self.dataset_b,
            trainer_fn=self.train_test_fn,
            grid_rf=self.grid_rf
        )
        return self._analyzer.feature_importance(analysis)


    def random_forest(self, build_id: str, fold_key: str) -> Tuple[pd.Series, pd.Series]:
        my_logger.info(f"Starting random_forest for build_id={build_id}, fold_key={fold_key}")
        X_train, y_train, X_test, y_test = self.train_test_fn(build_id,fold_key)
        y_pred = self.predict(X_train, y_train, X_test, y_test)
        return y_pred, y_test
