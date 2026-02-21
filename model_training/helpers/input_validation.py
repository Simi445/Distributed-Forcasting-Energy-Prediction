from classes import DataContext, TrainingContext, FeatureAnalysisContext
import pandas as pd
from typing import List,Callable,Tuple,Dict,Any,Union
from my_logger_pkg import my_logger

AnyContext = Union[DataContext, TrainingContext, FeatureAnalysisContext]

class ValidationDecorator:
    @staticmethod
    def decorator(function:Callable):
        def wrapper(*args:Tuple, **kwargs:Dict[str,Any]):
            valid_args = [arg for arg in args if isinstance(arg, (DataContext, TrainingContext, FeatureAnalysisContext))]
            if len(valid_args) < 1:
                raise ValueError("Context object (DataContext, TrainingContext, FeatureAnalysisContext) not found in all arguments")
            for arg in valid_args:
                try:
                    class_name = arg.__class__.__name__
                    validator_method = getattr(InputValidator, f"validate_{class_name}")
                    validator_method(arg)
                except Exception as e:
                    my_logger.exception(f"Exception occurred in validation; {e}")
                    raise
            result = function(*args, **kwargs)
            return result
        return wrapper


class InputValidator:
    """Validation for input parameters and data"""

    @staticmethod
    def validate_FeatureAnalysisContext(context: AnyContext) -> None:
        variables = {
            'iterations': getattr(context, 'iterations', None),
            'state_seed': getattr(context, 'state_seed', None),
            'trainer_fn': getattr(context, 'trainer_fn', None)
        }
        
        empty_vars = [name for name, value in variables.items() if value is None]
        if len(empty_vars) == len(variables):
            raise ValueError(f"Error, following variables are empty: {empty_vars}")
        
    @staticmethod
    def validate_DataContext(context: AnyContext) -> None:
        variables = {
            'dataset_b': getattr(context, 'dataset_b', None),
            'build_id': getattr(context, 'build_id', None),
            'fold_key': getattr(context, 'fold_key', None)
        }
        
        empty_vars = [name for name, value in variables.items() if value is None]
        if len(empty_vars) == len(variables):
            raise ValueError(f"Error, following variables are empty: {empty_vars}")
    
    @staticmethod
    def validate_TrainingContext(context: AnyContext) -> None:
        variables = {
            'grid': getattr(context, 'grid', None),
            'strategy': getattr(context, 'strategy', None)
        }
        
        empty_vars = [name for name, value in variables.items() if value is None]
        if len(empty_vars) == len(variables):
            raise ValueError(f"Error, following variables are empty: {empty_vars}")
    
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
