from .model_factories import ModelFactory, RandomForestFactory, SVRFactory, XGBoostFactory
from my_logger_pkg import my_logger
from typing import Dict, List, Tuple, Optional
from auxiliary import DataOrganizer, MetricsImpl, ModelManager
from seasonality import NoSeasonalityStrategy, SeasonMaskStrategy
from classes import SeasonalityStrategy, Model, GlobalInfoRF, GlobalInfo, SeasonConfig, Pipeline
from data_plot import PlotHelperModel

class NonBaselinePipeline(Pipeline):
    def __init__(self, state_manager: ModelManager, possible_seasons: SeasonConfig = SeasonConfig()):
        self.state = state_manager
        self.possible_seasons = possible_seasons
        self._data_organizer: Optional[DataOrganizer] = None
        self._fit_cache: Dict[Tuple[str, int, str, str], Tuple] = {}
        self._importance_cache: Dict[Tuple[str, int, str, str], Tuple[Dict, Dict]] = {}
        self.model_factories: Dict[str, ModelFactory] = {
            "random_forest": RandomForestFactory(),
            "svr": SVRFactory(),
            "xgboost": XGBoostFactory(),
        }
    
    @property
    def data_organizer(self) -> DataOrganizer:
        if self._data_organizer is None:
            self._data_organizer = DataOrganizer()
        return self._data_organizer

    def run(self, model_name: str, season: Optional[str] = None) -> Dict:
        is_seasonal = season is not None

        if model_name not in self.model_factories:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_factories.keys())}")
        
        if is_seasonal and season not in self.possible_seasons.possible_seasons:
            raise ValueError(f"Unknown season: {season}. Available: {self.possible_seasons.possible_seasons}")
        
        season_list = [season] if is_seasonal else self.possible_seasons.possible_seasons
        strat: SeasonalityStrategy = SeasonMaskStrategy() if is_seasonal else NoSeasonalityStrategy()
        strategy_mode:str = "seasonal" if is_seasonal else "global"
        
        results = {}
        for current_season in season_list:
            key: Tuple[str, str, str] = (model_name, current_season, strategy_mode)
            model: Optional[Model] = None
            
            try:
                model = self.state.get_model(key)
            except ValueError:
                my_logger.info(f"Model {model_name} for {current_season} and strategy {strategy_mode} not found, creating new one")
                try:
                    dataset_b: Dict = self.data_organizer.get_dataset_b()
                    model = self.model_factories[model_name].create(dataset_b, strat)
                    self.state.add_model(key, model)
                    my_logger.info(f"Added model: {model_name}, season: {current_season}, strategy: {strategy_mode}")
                except Exception as e:
                    my_logger.exception(f"Failed to create model: {e}")
                    raise RuntimeError(f"Failed to create model {model_name} for season {current_season}") from e
            
            dataset_b: Dict = self.data_organizer.get_dataset_b()
            for building_id in dataset_b:
                cache_key: Tuple[str, int, str, str] = (model_name, building_id, current_season, strategy_mode)
                
                if cache_key in self._fit_cache:
                    my_logger.debug(f"Cache hit: {model_name}, building {building_id}, season {current_season}")
                    pred, test, grid, best_params = self._fit_cache[cache_key]
                else:
                    my_logger.debug(f"Cache miss: {model_name}, building {building_id}, season {current_season}, fitting model")
                    pred, test, grid, best_params = model.fit(building_id, current_season)
                    self._fit_cache[cache_key] = (pred, test, grid, best_params)
                
                html:str = PlotHelperModel.plot_model_pred(build_id=building_id, pred=pred,test=test)
                res_key: Tuple = (building_id, current_season, strategy_mode)
                
                base_metrics = MetricsImpl.get_metrics(pred, test)
                if model_name == "random_forest":
                    if cache_key in self._importance_cache:
                        my_logger.debug(f"Importance cache hit: {model_name}, building {building_id}, season {current_season}")
                        p_importance, f_importance = self._importance_cache[cache_key]
                    else:
                        my_logger.debug(f"Importance cache miss: {model_name}, building {building_id}, season {current_season}, computing importance")
                        p_importance = model.permutation_importance(building_id, current_season, grid)
                        f_importance = model.feature_importance(building_id, current_season, grid)
                        self._importance_cache[cache_key] = (p_importance, f_importance)
                    results[res_key] = GlobalInfoRF(
                        metrics=base_metrics, 
                        p_importance=p_importance,
                        f_importance=f_importance,
                        html = html
                    )
                else:
                    results[res_key] = GlobalInfo(
                        metrics=base_metrics,
                        html = html
                    )
        
        return results