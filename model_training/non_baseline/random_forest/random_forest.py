import pandas as pd
from my_logger_pkg import my_logger
from typing import Dict, Tuple
from seasonality import NoSeasonalityStrategy
from classes import (
    GridSearchModel, SeasonalityStrategy, RFGridParamsConfig, ModelConfigparams,
    DataContext, TrainingContext, FeatureAnalysisContext
)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from helpers import FeatureAnalyzer


class RandomForestImpl(GridSearchModel):
    """Random Forest implementation using the GridSearchModel base class."""
    
    def __init__(self, param_grid: RFGridParamsConfig, config: ModelConfigparams, 
                 dataset_b: Dict, seasonality_strategy: SeasonalityStrategy = NoSeasonalityStrategy()):
        super().__init__(
            config=config,
            dataset_b=dataset_b,
            seasonality_strategy=seasonality_strategy,
            param_grid=param_grid.parameters
        )
        my_logger.info(f"Initializing RandomForest with state_seed={self.state_seed}, k_fold={self.k_fold}")
        self._analyzer = FeatureAnalyzer(self._trainer)
    
    def _get_model_name(self)->str:
        return self.__class__.__name__

    def _make_grid(self) -> GridSearchCV:
        return GridSearchCV(
            RandomForestRegressor(random_state=self.state_seed),
            self.param_grid,
            cv=TimeSeriesSplit(n_splits=self.k_fold),
            return_train_score=False,
            n_jobs=self.n_jobs
        )

    def permutation_importance(self, build_id: int, fold_key: str, grid_rf: GridSearchCV, strategy_mode: str = "global", iterations: int = 10) -> Dict[str, Dict[str, float]]:
        data_ctx = DataContext(
            build_id=build_id,
            fold_key=fold_key,
            dataset_b=self.dataset_b
        )
        training_ctx = TrainingContext(grid=grid_rf, model_name=self._get_model_name(), strategy=strategy_mode)
        analysis_ctx = FeatureAnalysisContext(
            iterations=iterations,
            state_seed=self.state_seed,
            trainer_fn=self.train_test_fn
        )
        return self._analyzer.permutation_importance(analysis_ctx, data_ctx, training_ctx)

    def feature_importance(self, build_id: int, fold_key: str, grid_rf: GridSearchCV, strategy_mode: str = "global") -> Dict[str, float]:
        data_ctx = DataContext(
            build_id=build_id,
            fold_key=fold_key,
            dataset_b=self.dataset_b
        )
        training_ctx = TrainingContext(grid=grid_rf, model_name=self._get_model_name(), strategy=strategy_mode)
        analysis_ctx = FeatureAnalysisContext(trainer_fn=self.train_test_fn)
        
        mdi = self._analyzer.feature_importance(analysis_ctx, data_ctx, training_ctx)
        return mdi.to_dict()
