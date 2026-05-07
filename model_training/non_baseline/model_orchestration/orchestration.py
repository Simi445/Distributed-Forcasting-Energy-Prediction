from my_logger_pkg import my_logger
from typing import Dict, Tuple, Optional, Any
from auxiliary import DataOrganizer, MetricsImpl, FactoryProvider
from seasonality import NoSeasonalityStrategy, SeasonMaskStrategy
from classes import SeasonalityStrategy, Model, GlobalInfoRF, GlobalInfo, SeasonConfig, Pipeline, ModelMangerAbstract, Organizer, ModelFactory
from data_plot import PlotHelperModel

class NonBaselinePipeline(Pipeline):
    def __init__(self, factory_provider: FactoryProvider, state_manager: ModelMangerAbstract, data_organizer: Organizer, possible_seasons: SeasonConfig = SeasonConfig()):
        self.state = state_manager
        self.possible_seasons = possible_seasons
        self.data_organizer = data_organizer
        self._fit_cache: Dict[Tuple[str, int, str, str], Tuple] = {}
        self._importance_cache: Dict[Tuple[str, int, str, str], Tuple[Dict, Dict]] = {}
        self.factory_provider: FactoryProvider = factory_provider
        

    def _retrieve_importances(
            self,
            cache_key:Tuple, 
            model_name:str, 
            current_season:str,
            building_id:int,
            grid:Any,
            model: Any,
            strategy_mode: str
        ):
        if cache_key in self._importance_cache:
            my_logger.debug(f"Importance cache hit: {model_name}, building {building_id}, season {current_season}")
            p_importance, f_importance = self._importance_cache[cache_key]
        else:
            my_logger.debug(f"Importance cache miss: {model_name}, building {building_id}, season {current_season}, computing importance")
            p_importance = model.permutation_importance(building_id, current_season, grid, strategy_mode)
            f_importance = model.feature_importance(building_id, current_season, grid, strategy_mode)
            self._importance_cache[cache_key] = (p_importance, f_importance)
        return self._importance_cache[cache_key]

    def _get_or_create_model(
        self,
        model_name: str,
        current_season: str,
        strategy_mode: str,
        dataset_b: Dict,
        strategy: SeasonalityStrategy,
    ) -> Model:
        key: Tuple[str, str, str] = (model_name, current_season, strategy_mode)

        try:
            return self.state.get_model(key)
        except ValueError:
            my_logger.info(
                f"Model {model_name} for {current_season} and strategy {strategy_mode} not found, creating new one"
            )
            try:
                factory = self.factory_provider.get_factory(model_name)
                model = factory.create(dataset_b, strategy)
                self.state.add_model(key, model)
                my_logger.info(f"Added model: {model_name}, season: {current_season}, strategy: {strategy_mode}")
                return model
            except Exception as e:
                my_logger.exception(f"Failed to create model: {e}")
                raise RuntimeError(f"Failed to create model {model_name} for season {current_season}") from e

    def run(self, model_name: str, season: Optional[str] = None) -> Dict:
        is_seasonal = season is not None
        self.factory_provider.get_factory(model_name) #early quit
        
        if is_seasonal and season not in self.possible_seasons.possible_seasons:
            raise ValueError(f"Unknown season: {season}. Available: {self.possible_seasons.possible_seasons}")
        
        season_list = [season] if is_seasonal else self.possible_seasons.possible_seasons
        strat: SeasonalityStrategy = SeasonMaskStrategy() if is_seasonal else NoSeasonalityStrategy()
        strategy_mode:str = "seasonal" if is_seasonal else "global"
        dataset_b: Dict = self.data_organizer.get_dataset_b()
        
        results = {}
        for current_season in season_list:
            model = self._get_or_create_model(model_name, current_season, strategy_mode, dataset_b, strat)
            for building_id in dataset_b:
                cache_key: Tuple[str, int, str, str] = (model_name, building_id, current_season, strategy_mode)
                
                if cache_key in self._fit_cache:
                    my_logger.debug(f"Cache hit: {model_name}, building {building_id}, season {current_season}, strategy {strategy_mode}")
                    pred, test, grid, best_params = self._fit_cache[cache_key]
                else:
                    my_logger.debug(f"Cache miss: {model_name}, building {building_id}, season {current_season}, strategy {strategy_mode}, fitting model")
                    pred, test, grid, best_params = model.fit(building_id, current_season,strategy_mode)
                    self._fit_cache[cache_key] = (pred, test, grid, best_params)
                
                html:str = PlotHelperModel.plot_model_pred(build_id=building_id, pred=pred,test=test)
                res_key: Tuple = (building_id, current_season, strategy_mode)
                
                base_metrics = MetricsImpl.get_metrics(pred, test)
                
                base_args = {
                    "metrics": base_metrics,
                    "html": html
                }
                
                if model_name == "random_forest":
                    p_importance, f_importance = self._retrieve_importances(cache_key,model_name,current_season,
                                                                            building_id,grid,model,strategy_mode)
                    results[res_key] = GlobalInfoRF(
                        **base_args,
                        p_importance=p_importance,
                        f_importance=f_importance
                    )
                else:
                    results[res_key] = GlobalInfo(**base_args)
        
        return results
