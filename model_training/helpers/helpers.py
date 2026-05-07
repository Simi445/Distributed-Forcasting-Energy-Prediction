import os

import pandas as pd
import pickle
import requests
from my_logger_pkg import my_logger
from typing import Dict, Tuple, Callable, Any, Union, Optional
from seasonality import NoSeasonalityStrategy
from classes import (
    DataContext, TrainingContext, FeatureAnalysisContext,
    Analyzer, Train, SeasonalityStrategy, RFGridParamsConfig, TrainTest
)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from helpers.input_validation import InputValidator, ValidationDecorator
import numpy as np

WEATHER_COL_RANGE = slice(24, 31)
TIME_COL_RANGE = slice(31, 37)

ENDPOINT = os.environ.get('CLOUD_ENDPOINT','http://data-cloud:8000')

class GridNetworkManager:
    """Manages grid serialization, cloud upload/download operations."""
    
    def __init__(self):
        self.cloud_url = ENDPOINT
    
    
    def download_from_cloud(self, model_file: str, local_path: str) -> bool:
        """Download grid from cloud and save to local path."""
        try:
            my_logger.info(f"Downloading grid {model_file} from cloud...")
            response: requests.Response = requests.get(
                f"{self.cloud_url}/download_grid_file/{model_file}",
                timeout=30
            )
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            my_logger.info(f"Grid {model_file} downloaded successfully from cloud.")
            return True
        except requests.RequestException as e:
            my_logger.warning(f"Failed to download grid from cloud: {e}")
            return False
    
    def upload_to_cloud(self, model_file: str, local_path: str) -> bool:
        """Upload grid to cloud storage."""
        try:
            my_logger.info("Uploading grid to cloud...")
            with open(local_path, 'rb') as f:
                response: requests.Response = requests.post(
                    f"{self.cloud_url}/upload_grid_file/{model_file}",
                    files={'grid': f},
                    timeout=30
                )
                response.raise_for_status()
            
            my_logger.info("Grid uploaded to cloud successfully!")
            return True
        except requests.RequestException as e:
            my_logger.warning(f"Failed to upload grid to cloud: {e}")
            return False

class TrainTestCache(TrainTest):
    def __init__(self):
        self._tt_cache: Dict[Tuple[int, str, str], Any] = {}

    @ValidationDecorator.decorator
    def get(self, data_ctx: DataContext, training_ctx: TrainingContext) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        fold_key, build_id = data_ctx.fold_key, data_ctx.build_id
        dataset_b, strategy = data_ctx.dataset_b, training_ctx.strategy
        strategy_name: str = type(strategy).__name__
        
        key: Tuple[int, str, str] = (build_id, fold_key, strategy_name)
        if key in self._tt_cache:
            my_logger.debug(f"Cache hit for build_id={build_id}, fold_key={fold_key}, strategy={strategy_name}")
            X_tr, y_tr, X_te, y_te = self._tt_cache[key]
            return (X_tr.copy(), y_tr.copy(), X_te.copy(), y_te.copy())
        
        my_logger.debug(f"Cache miss for build_id={build_id}, fold_key={fold_key}, strategy={strategy_name}. Computing train_test split.")
        train_data: pd.DataFrame = dataset_b[build_id][fold_key]['train']
        test_data: pd.DataFrame = dataset_b[build_id][fold_key]['test']
        
        train_data = strategy.apply(train_data, fold_key)

        X_train:pd.DataFrame = train_data.drop('meter_reading', axis=1)
        y_train: pd.Series = train_data['meter_reading']
        InputValidator.validate_series(series=y_train, name="y_train")

        X_test:pd.DataFrame = test_data.drop('meter_reading', axis=1)
        y_test: pd.Series = test_data['meter_reading']
        InputValidator.validate_series(series=y_test, name="y_test")

        InputValidator.validate_train_test_lengths(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        self._tt_cache[key] = (X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
        my_logger.debug(f"Cached train_test split: X_train.shape={X_train.shape}, X_test.shape={X_test.shape}")
        return (X_train,y_train,X_test,y_test)

class TrainerImpl(Train):
    def __init__(self):
        self.network_manager = GridNetworkManager()

    def _grid_artifact_naming(self,model_name:str, strategy:Any, build_id:int, fold_key:str) -> Tuple[str,str]:
        model_file: str = f'{model_name}_{strategy}_{build_id}_{fold_key}_grid.pkl'

        grid_dir = os.environ.get('GRID_DIR', '/app/grids')
        os.makedirs(grid_dir, exist_ok=True)

        local_path: str = os.path.join(grid_dir, f'{model_file}')

        return model_file, local_path

    def _grid_retrieval(self, model_file:str, local_path:str, grid:Any, X_train: pd.DataFrame, y_train: pd.Series)->Tuple[Any, Dict]:
        
        grid_exists_locally = os.path.isfile(local_path)
        grid_loaded = False
        
        if grid_exists_locally:
            my_logger.info(f"Loading cached grid from {local_path}")
            try:
                with open(local_path, 'rb') as f:
                    grid = pickle.load(f)
                best_params: Dict = grid.best_params_
                my_logger.info(f"Grid loaded. Best params: {best_params}")
                grid_loaded = True
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                my_logger.warning(f"Failed to load cached grid from {local_path}: {e}. Will attempt to download from cloud or refit.")
        
        if not grid_loaded:
            grid_downloaded = self.network_manager.download_from_cloud(model_file, local_path)
            
            if grid_downloaded:
                try:
                    with open(local_path, 'rb') as f:
                        grid = pickle.load(f)
                    best_params: Dict = grid.best_params_
                    my_logger.info(f"Grid loaded from cloud. Best params: {best_params}")
                    grid_loaded = True
                except (EOFError, pickle.UnpicklingError, Exception) as e:
                    my_logger.warning(f"Failed to load grid from cloud: {e}. Will refit GridSearchCV.")
            
            if not grid_loaded:
                my_logger.info(f"Grid not found locally or in cloud, or file was corrupted. Fitting GridSearchCV...")
                grid.fit(X_train, y_train)
                best_params: Dict = grid.best_params_
                
                with open(local_path, 'wb') as f:
                    pickle.dump(grid, f)
                my_logger.info(f"Grid saved locally to {local_path}. Best params: {best_params}")
                
                self.network_manager.upload_to_cloud(model_file, local_path)
        return grid, best_params 
    
    @ValidationDecorator.decorator
    def fit_predict(self, training_ctx: TrainingContext, data_ctx:DataContext, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.Series,Dict]:
        my_logger.info(f"Starting GridSearchCV.fit with training set size {len(y_train)}")

        model_file, local_path = self._grid_artifact_naming(training_ctx.model_name, training_ctx.strategy, data_ctx.build_id, data_ctx.fold_key)
        training_ctx.grid, best_params = self._grid_retrieval(model_file, local_path, training_ctx.grid, X_train, y_train)

        y_pred = pd.Series(training_ctx.grid.predict(X_test), index=y_test.index)
        InputValidator.validate_series(series=y_pred, name="y_pred")

        my_logger.debug(f"Predictions generated for test set size {len(y_test)}")
        return y_pred, best_params

    @ValidationDecorator.decorator
    def nofit_predict(self, training_ctx: TrainingContext, X_test: pd.DataFrame) -> pd.Series:
        InputValidator.validate_dataframe(X_test, "X_test")
        y_pred = pd.Series(training_ctx.grid.predict(X_test))
        InputValidator.validate_series(series=y_pred, name="y_pred")
        
        return y_pred
    

class FeatureAnalyzer(Analyzer):
    def __init__(self, trainer: Train):
        self.trainer: Train = trainer

    @ValidationDecorator.decorator
    def _get_correlated_columns(self, data_ctx: DataContext) -> Dict[str, list]:
        dataset_b, build_id = data_ctx.dataset_b, data_ctx.build_id
        data: pd.DataFrame = pd.Series(dataset_b[build_id]['autumn']['train'].columns).to_frame("y")
        primary_uses_cols: list = data[data['y'].str.contains('primary_use')]['y'].tolist()
        weather_cols: list = data.iloc[WEATHER_COL_RANGE]['y'].tolist()
        time_cols: list = data.iloc[TIME_COL_RANGE]['y'].tolist()
        lag_columns: list = data[data['y'].str.contains('load')]['y'].tolist()
        nonlin_columns: list = data[data['y'].str.contains('hour_')]['y'].tolist()
        return {
            "usage": primary_uses_cols,
            "weather": weather_cols,
            "time": time_cols,
            "lag": lag_columns,
            "nonlinear": nonlin_columns
        }  

    @ValidationDecorator.decorator
    def permutation_importance(self, analysis_ctx: FeatureAnalysisContext, data_ctx: DataContext, 
                               training_ctx: TrainingContext) -> Dict[str, Dict[str, float]]:
        build_id, fold_key = data_ctx.build_id, data_ctx.fold_key
        trainer_fn = analysis_ctx.trainer_fn
        iterations, state_seed = analysis_ctx.iterations, analysis_ctx.state_seed
        
        my_logger.info(f"Computing permutation_importance for build_id={build_id}, fold_key={fold_key}, iterations={iterations}")
        X_train, y_train, X_test, y_test = trainer_fn(build_id, fold_key)
        y_pred, _ = self.trainer.fit_predict(training_ctx,data_ctx, X_train, y_train, X_test, y_test)
        
        res: Dict[str, Dict[str, float]] = {}
        columns: Dict[str, list] = self._get_correlated_columns(data_ctx)
        for key in columns:
            group_cols: list = columns[key]
            maes: list = []
            for r in range(iterations):
                rng: np.random.RandomState = np.random.RandomState(state_seed + r)
                X_test_new: pd.DataFrame = X_test.copy()
                perm_idx: np.ndarray = rng.permutation(len(X_test_new))

                X_test_new[group_cols] = X_test_new[group_cols].iloc[perm_idx].to_numpy()
                y_perm: np.ndarray = self.trainer.nofit_predict(training_ctx, X_test_new)
                mae_permuted: float = mean_absolute_error(y_test, y_perm)
                maes.append(mae_permuted)

            mae_permuted = np.mean(maes)
            mae_std = np.std(maes)
            mae_original = mean_absolute_error(y_test, y_pred)
            res[key] = {
                "mae_permuted": mae_permuted,
                "mae_std": mae_std,
                "mae_original": mae_original,
                "mae_rel_increase": (mae_permuted - mae_original) / mae_original
            }
            my_logger.debug(f"Feature group '{key}': mae_rel_increase={res[key]['mae_rel_increase']:.4f}")
        my_logger.info(f"Permutation importance computation completed with {len(res)} feature groups")
        if not res:
            my_logger.warning("No feature groups were analyzed in permutation importance")
        return res

    @ValidationDecorator.decorator
    def feature_importance(self, analysis_ctx: FeatureAnalysisContext, data_ctx: DataContext,
                          training_ctx: TrainingContext) -> pd.Series:
        build_id, fold_key = data_ctx.build_id, data_ctx.fold_key
        trainer_fn, grid = analysis_ctx.trainer_fn, training_ctx.grid
        
        my_logger.info(f"Computing feature_importance for build_id={build_id}, fold_key={fold_key}")
        X_train, y_train, _, _ = trainer_fn(build_id, fold_key)
        
        grid.fit(X_train, y_train)
        
        model: RandomForestRegressor = grid.best_estimator_
        mdi: pd.Series = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        InputValidator.validate_series(series=mdi, name="mdi")

        my_logger.debug(f"Top 5 important features: {mdi.head().to_dict()}")
        return mdi