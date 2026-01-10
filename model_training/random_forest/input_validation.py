from ..classes import AnalysisConfig
import pandas as pd
from typing import List,Callable,Tuple,Dict,Any
from .my_logger_pkg import my_logger

class ValidationDecorator:
    @staticmethod
    def validate_analysis_context(arguments:List[str]):
        def decorator(function:Callable):
            def wrapper(*args:Tuple, **kwargs:Dict[str,Any]):
                analysis = next((arg for arg in args if type(arg) == AnalysisConfig), None)
                if analysis is None:
                    raise ValueError("No AnalysisConfig found in arguments")

                for argument in arguments:
                    try:
                        validator_method = getattr(InputValidator, f"validate_{argument}")
                        validator_method(analysis)
                    except Exception as e:
                        my_logger.exception(f"Exception occurred in validation; {e}")
                        raise
                result = function(*args, **kwargs)
                return result
            return wrapper
        return decorator


class InputValidator:
    """Validation for input parameters and data"""
        
    @staticmethod
    def validate_analysis_config(analysis: AnalysisConfig) -> None:
        if analysis is None:
            raise ValueError("AnalysisConfig cannot be None")
    
    @staticmethod
    def validate_build_fold_keys(analysis: AnalysisConfig) -> None:
        if analysis.build_id is None or analysis.fold_key is None:
            raise ValueError(f"build_id and fold_key cannot be None: build_id={analysis.build_id}, fold_key={analysis.fold_key}")
    
    @staticmethod
    def validate_dataset_b(analysis: AnalysisConfig) -> None:
        if analysis.dataset_b is None:
            raise ValueError("dataset_b cannot be None")
        if analysis.build_id not in analysis.dataset_b:
            raise KeyError(f"build_id {analysis.build_id} not found in dataset_b")
        if analysis.fold_key not in analysis.dataset_b[analysis.build_id]:
            raise KeyError(f"fold_key '{analysis.fold_key}' not found for build_id {analysis.build_id}")
    
    @staticmethod
    def validate_strategy(analysis: AnalysisConfig) -> None:
        if analysis.strategy is None:
            raise ValueError("strategy cannot be None")
    
    @staticmethod
    def validate_grid_rf(analysis: AnalysisConfig) -> None:
        if analysis.grid_rf is None:
            raise ValueError("grid_rf cannot be None")
        if not hasattr(analysis.grid_rf, 'best_estimator_'):
            raise AttributeError("grid_rf does not have best_estimator_ attribute")
    
    @staticmethod
    def validate_trainer_fn(analysis: AnalysisConfig) -> None:
        if analysis.trainer_fn is None:
            raise ValueError("trainer_fn cannot be None")
    
    @staticmethod
    def validate_iterations(analysis: AnalysisConfig) -> None:
        if analysis.iterations is None or analysis.iterations <= 0:
            raise ValueError(f"iterations must be positive integer, got {analysis.iterations}")
    
    @staticmethod
    def validate_state_seed(analysis: AnalysisConfig) -> None:
        if analysis.state_seed is None or analysis.state_seed < 0:
            raise ValueError(f"state_seed must be non-negative integer, got {analysis.state_seed}")
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, name: str) -> None:
        if df is None or df.empty:
            raise ValueError(f"{name} cannot be None or empty")
    
    @staticmethod
    def validate_series(series: pd.Series, name: str) -> None:
        if series is None or series.empty:
            raise ValueError(f"{name} cannot be None or empty")
    
    @staticmethod
    def validate_train_test_lengths(X_train: pd.DataFrame, y_train: pd.Series, 
                                    X_test: pd.DataFrame, y_test: pd.Series) -> None:
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
        if len(X_test) != len(y_test):
            raise ValueError(f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}")
    
    @staticmethod
    def validate_autumn_train_data(analysis: AnalysisConfig) -> None:
        if 'autumn' not in analysis.dataset_b[analysis.build_id]:
            raise KeyError(f"'autumn' fold not found for build_id {analysis.build_id}")
        if 'train' not in analysis.dataset_b[analysis.build_id]['autumn']:
            raise KeyError(f"'train' data not found for build_id {analysis.build_id} in autumn fold")
