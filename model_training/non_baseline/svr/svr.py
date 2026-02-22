import pandas as pd
from my_logger_pkg import my_logger
from typing import Dict, Tuple, List, Any
from seasonality import NoSeasonalityStrategy
from classes import (
    GridSearchModel, SeasonalityStrategy, ModelConfigparams,
    TrainingContext
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR as SklearnSVR
from sklearn.model_selection import TimeSeriesSplit
import numpy as np



class SVRModel(GridSearchModel):
    """SVR implementation using the GridSearchModel base class.
    
    Note: SVR requires calling make_param_grid(y_train) before fit() to 
    configure epsilon and C based on training data statistics.
    """
    
    def __init__(self, config: ModelConfigparams, dataset_b: Dict,
                 seasonality_strategy: SeasonalityStrategy = NoSeasonalityStrategy()):
        super().__init__(
            config=config,
            dataset_b=dataset_b,
            seasonality_strategy=seasonality_strategy,
            param_grid=None  # SVR computes param_grid dynamically
        )
        my_logger.info(f"Initializing SVR with k_fold={self.k_fold}")
    
    def _get_model_name(self)->str:
        return self.__class__.__name__
       
    def _calculate_train_std(self, y_train: pd.Series) -> float:
        return np.std(y_train)

    def _prepare_grid(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Configure parameter grid based on training data statistics.
        
        Must be called before fit() or svr().
        """
        train_std: float = self._calculate_train_std(y_train)
        self.param_grid = {
            "epsilon": [0.1 * train_std],
            "C": [10 * train_std],
            "kernel": ['rbf']
        }
        my_logger.debug(f"SVR param_grid configured with train_std={train_std:.4f}")

    def _make_grid(self) -> GridSearchCV:
        if not self.param_grid:
            raise ValueError("For SVR, please call _prepare_grid(y_train) method first!")
        return GridSearchCV(
            SklearnSVR(),
            self.param_grid,
            cv=TimeSeriesSplit(n_splits=self.k_fold),
            return_train_score=False,
            n_jobs=self.n_jobs
        )
