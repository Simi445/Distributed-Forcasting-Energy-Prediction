from data_plot.plot_helper import PlotHelperModel
from my_logger_pkg import my_logger
from typing import Dict, Any, Tuple, Optional
from classes import SeasonConfig, Pipeline, ModelMangerAbstract, ModelFactory, Organizer
from auxiliary import MetricsImpl, FactoryProvider, SeasonMask

class BaselinePipeline(Pipeline):
    def __init__(
        self,
        factory_provider: FactoryProvider,
        state_manager: ModelMangerAbstract,
        data_organizer: Organizer,
        possible_seasons: SeasonConfig = SeasonConfig(),
    ):
        self.state = state_manager
        self.factory_provider = factory_provider
        self.data_organizer = data_organizer
        self.possible_seasons = possible_seasons

    def _get_or_create_model(self, model_name: str, season: str) -> Any:
        key: Tuple[str, str] = (model_name, season)
        try:
            return self.state.get_model(key)
        except ValueError:
            my_logger.info(f"Model {model_name} for {season} not found, creating new one")
            try:
                factory = self.factory_provider.get_factory(model_name)
                from seasonality import SeasonMaskStrategy
                model = factory.create(organizer=self.data_organizer, strategy=SeasonMaskStrategy())
                self.state.add_model(key, model)
                my_logger.info(f"Added model: {model_name}, season: {season}")
                return model
            except Exception as e:
                my_logger.exception(f"Failed to create model: {e}")
                raise RuntimeError(f"Failed to create model {model_name} for season {season}") from e


    def run(self, model_name: str, season: Optional[str] = None) -> Dict[int, Dict[str, float]]:

        if season is None:
            raise ValueError("Baseline models require a season argument.")

        self.factory_provider.get_factory(model_name)

        if season not in self.possible_seasons.possible_seasons:
            raise ValueError(f"Unknown season: {season}. Available: {self.possible_seasons.possible_seasons}")

        model: Any = self._get_or_create_model(model_name, season)

        all_metrics: Dict = {}
        train_res: Dict = model.train(season)
        for build_id, (y_pred, y_test) in train_res.items():
            metrics = MetricsImpl.get_metrics(y_pred, y_test)
            html:str = PlotHelperModel.plot_model_pred(build_id=build_id, pred=y_pred,test=y_test)
            
            all_metrics[(build_id,season)] = {"metrics": metrics, "html": html}
            my_logger.info(f"{model_name} model metrics for building {build_id}, {season}: {metrics}")
        return all_metrics
    
