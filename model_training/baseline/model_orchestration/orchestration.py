from my_logger_pkg import my_logger
from typing import Dict, Any, Tuple, Optional
from classes import SeasonConfig, Pipeline
from auxiliary import MetricsImpl, ModelManager
from .model_factories import NaiveFactory, MeanFactory

class BaselinePipeline(Pipeline):
    def __init__(self, state_manager: ModelManager, possible_seasons: SeasonConfig = SeasonConfig()):
        self.state = state_manager
        self.possible_seasons = possible_seasons
        self.model_factories: Dict[str, Any] = {
            "naive": NaiveFactory(),
            "mean": MeanFactory(),
        }

    def _get_or_create_model(self, model_name: str, season: str) -> Any:
        key: Tuple[str, str] = (model_name, season)
        try:
            return self.state.get_model(key)
        except ValueError:
            my_logger.info(f"Model {model_name} for {season} not found, creating new one")
            try:
                model = self.model_factories[model_name].create()
                self.state.add_model(key, model)
                my_logger.info(f"Added model: {model_name}, season: {season}")
                return model
            except Exception as e:
                my_logger.exception(f"Failed to create model: {e}")
                raise RuntimeError(f"Failed to create model {model_name} for season {season}") from e


    def run(self, model_name: str, season: Optional[str] = None) -> Dict[int, Dict[str, float]]:

        if season is None:
            raise ValueError("Baseline models require a season argument.")

        if model_name not in self.model_factories:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_factories.keys())}")

        if season not in self.possible_seasons.possible_seasons:
            raise ValueError(f"Unknown season: {season}. Available: {self.possible_seasons.possible_seasons}")

        model: Any = self._get_or_create_model(model_name, season)

        all_metrics: Dict = {}
        train_res: Dict = model.train(season)
        for build_id, (y_pred, y_test) in train_res.items():
            metrics = MetricsImpl.get_metrics(y_pred, y_test)
            all_metrics[build_id] = metrics
            my_logger.info(f"{model_name} model metrics for building {build_id}, {season}: {metrics}")
        return all_metrics
    
