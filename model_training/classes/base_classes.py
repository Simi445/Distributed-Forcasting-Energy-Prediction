from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Dict, Tuple, Union, Any
import pandas as pd
from .models import SeasonConfig, AnalysisConfig

class BaselineModel(ABC):
    """Abstract base class for baseline model implementations.
    
    Provides an interface for implementing simple baseline models that serve
    as reference points for comparing more complex model performance.
    """
    pass


class Model(ABC):
    """Abstract base class for tree-based model implementations.
    
    Defines the core interface for training and prediction operations
    on tree-based regression models.
    """

    @abstractmethod
    def train_test_fn(self, build_id: int, fold_key: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Get train/test split for a specific building and fold.
        
        Args:
            build_id: The building identifier.
            fold_key: The fold identifier (e.g., 'autumn', 'winter').
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: 
            (X_train, y_train, X_test, y_test) data split.
        """
        pass

    @abstractmethod
    def predict(self, X_train, y_train, X_test, y_test, grid_rf)-> pd.Series:
        """Train and make predictions on test data.
        
        Args:
            X_train: Training feature matrix.
            y_train: Training target values.
            X_test: Test feature matrix.
            y_test: Test target values (for evaluation).
            grid_rf: Grid for the model.
            
        Returns:
            Predictions on the test set.
        """
        pass


class Metrics(ABC):
    """Abstract base class for metric calculation components."""
    
    @abstractmethod
    @staticmethod
    def get_metrics(pred: pd.Series, test: pd.Series) -> Dict[str, float]:
        """Calculate evaluation metrics comparing predictions to test values.
        
        Args:
            pred: Predicted values.
            test: Actual test values.
            
        Returns:
            Dictionary of metric names to their calculated values.
        """
        pass

class Mask(ABC):
    """Abstract base class for seasonal data masking components."""
    
    @abstractmethod
    @staticmethod
    def get_season_mask(dataframe: pd.DataFrame, season: str, possible_seasons: SeasonConfig) -> np.ndarray:
        """Generate a boolean mask for data belonging to a specific season.
        
        Args:
            dataframe: The input DataFrame with datetime information.
            season: The target season (e.g., 'autumn', 'winter').
            possible_seasons: Configuration specifying season definitions.
            
        Returns:
            Boolean numpy array indicating which rows belong to the season.
        """
        pass

class Organizer(ABC):
    """Abstract base class for data organization components.
    
    Handles structuring raw data into organized train/test splits
    and fold selections for cross-validation.
    """
    
    @abstractmethod
    def _two_fold_selection(self, dataset: pd.DataFrame) -> List[Tuple[Any, Any, Any]]:
        """Create two-fold splits from dataset.
        
        Args:
            dataset: The input DataFrame to split.
            
        Returns:
            List of tuples representing fold selections.
        """
        pass

    @abstractmethod
    def _train_test_sepparation(self) -> None:
        """Separate organized data into train and test sets."""
        pass

class SeasonalityStrategy(ABC):
    """Abstract base for seasonality handling strategies.
    
    Defines the interface for applying different seasonality filters
    to training data during model development.
    """
    
    @abstractmethod
    def apply(self, train_data: pd.DataFrame, fold_key: str) -> pd.DataFrame:
        """Apply seasonality filtering and return processed training data.
        
        Args:
            train_data: The raw training DataFrame.
            fold_key: The current fold identifier.
            
        Returns:
            Filtered training data with seasonality applied.
        """
        pass

class TrainTest(ABC):
    """Abstract base class for train/test data caching and retrieval.
    
    Manages efficient access to pre-split train/test data with caching
    to avoid redundant data processing.
    """
    
    @abstractmethod
    def get(self, build_id: str, fold_key: str, strategy: SeasonalityStrategy, dataset_b: Dict) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Retrieve train/test split with seasonality applied.
        
        Args:
            build_id: Building identifier.
            fold_key: Fold identifier.
            strategy: Seasonality strategy to apply.
            dataset_b: Dictionary containing all dataset splits.
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            (X_train, y_train, X_test, y_test) with seasonality applied.
        """
        pass
    
class Train(ABC):
    """Abstract base class for model training and prediction.
    
    Provides the interface for training models and making predictions
    on new data.
    """
    
    @abstractmethod
    def fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
        """Train model and make predictions on test set.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets (for reference).
            
        Returns:
            Predictions on the test set.
        """
        pass
    
    @abstractmethod
    def predict(self, analysis: AnalysisConfig, X_test: pd.DataFrame) -> pd.Series:
        """Make predictions on new data using trained model.
        
        Args:
            analysis: Analysis configuration containing model state.
            X_test: Test features.
            
        Returns:
            Predictions on the test set.
        """
        pass
    
class Analyzer(ABC):
    """Abstract base class for feature analysis components.
    
    Provides interfaces for analyzing feature importance and impact
    on model predictions.
    """
    
    @abstractmethod
    def permutation_importance(self, build_id: int, fold_key: str, dataset_b: Dict, iterations: int = 10) -> Dict[str, Dict[str, float]]:
        """Calculate permutation-based feature importance.
        
        Args:
            build_id: Building identifier.
            fold_key: Fold identifier.
            dataset_b: Dataset dictionary.
            iterations: Number of permutation iterations.
            
        Returns:
            Dictionary of feature groups with importance metrics.
        """
        pass

    @abstractmethod
    def feature_importance(self, build_id: int, fold_key: str) -> pd.Series:
        """Calculate model-based feature importance scores.
        
        Args:
            build_id: Building identifier.
            fold_key: Fold identifier.
            
        Returns:
            Series of feature importance scores sorted descending.
        """
        pass