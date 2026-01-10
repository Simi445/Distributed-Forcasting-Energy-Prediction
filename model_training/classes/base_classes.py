from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Dict, Tuple, Union, Any
import pandas as pd
from .models import SeasonConfig, AnalysisConfig

class BaselineModel(ABC):
    """Abstract base class for models."""
    pass


class TreeModel(ABC):
    """Abstract class for tree based models."""

    @abstractmethod
    def train_test_fn(build_id,fold_key, seasonal = False, is_svm = False):
        pass

    @abstractmethod
    def predict_fn(X_train, y_train, X_test, y_test):
        pass


class Metrics(ABC):
    @abstractmethod
    @staticmethod
    def get_metrics(pred: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
        pass

class Mask(ABC):
    @abstractmethod
    @staticmethod
    def get_season_mask(dataframe:pd.DataFrame, season:str, possible_seasons:SeasonConfig)-> np.ndarray:
        pass

class Organizer(ABC):
    @abstractmethod
    def two_fold_selection(self, dataset: pd.DataFrame) -> List[Tuple[Any, Any, Any]]:
        pass

    @abstractmethod
    def train_test_sepparation(self) -> None:
        pass

class SeasonalityStrategy(ABC):
    """Abstract base for seasonality handling strategies."""
    @abstractmethod
    def apply(self, train_data: pd.DataFrame, fold_key: str) -> pd.DataFrame:
        """Apply seasonality filtering and return processed training data."""
        pass

class TrainTestCache(ABC):
    @abstractmethod
    def get(self, build_id: str, 
            fold_key: str, strategy: SeasonalityStrategy,dataset_b:Dict)->Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            pass
    
class Train(ABC):
    @staticmethod
    @abstractmethod
    def fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
            pass
    
class Analyzer(ABC):
    @abstractmethod
    def permutation_importance(self, build_id:int, fold_key:str, dataset_b:Dict, iterations:int = 10):
        pass

    @abstractmethod
    def feature_importance(self, build_id: id, fold_key: str) -> pd.Series:
        pass