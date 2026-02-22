from .model_factories import ModelFactory, RandomForestFactory, SVRFactory, XGBoostFactory
from my_logger_pkg import my_logger
from typing import Dict, Tuple, Optional
from auxiliary import DataOrganizer, MetricsImpl, ModelManager
from seasonality import NoSeasonalityStrategy, SeasonMaskStrategy
from classes import SeasonalityStrategy, Model, GlobalInfoRF, GlobalInfo, SeasonConfig, Pipeline

class NonBaselinePipeline(Pipeline):
    def __init__(self, state_manager: ModelManager, possible_seasons: SeasonConfig = SeasonConfig()):
        self.state = state_manager
        self.possible_seasons = possible_seasons
        self._data_organizer: Optional[DataOrganizer] = DataOrganizer()
        self.model_factories: Dict[str, ModelFactory] = {
            "random_forest": RandomForestFactory(),
            "svr": SVRFactory(),
            "xgboost": XGBoostFactory(),
        }
    
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
                    model = self.model_factories[model_name].create(self._data_organizer.get_dataset_b(), strat)
                    self.state.add_model(key, model)
                    my_logger.info(f"Added model: {model_name}, season: {current_season}, strategy: {strategy_mode}")
                except Exception as e:
                    my_logger.exception(f"Failed to create model: {e}")
                    raise RuntimeError(f"Failed to create model {model_name} for season {current_season}") from e
            
            dataset_b: Dict = self._data_organizer.get_dataset_b()
            for building_id in dataset_b:
                pred, test, grid = model.fit(building_id, current_season)
                res_key: Tuple = (building_id, current_season, strategy_mode)
                
                base_metrics = MetricsImpl.get_metrics(pred, test)
                if model_name == "random_forest":
                    results[res_key] = GlobalInfoRF(
                        metrics=base_metrics, 
                        p_importance=model.permutation_importance(building_id, current_season, grid),
                        f_importance=model.feature_importance(building_id, current_season, grid)
                    )
                else:
                    results[res_key] = GlobalInfo(
                        metrics=base_metrics
                    )
        
        return results