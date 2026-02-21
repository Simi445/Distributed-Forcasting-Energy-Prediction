import pandas as pd
from model_orchestration.model_factories import ModelFactory, RandomForestFactory, SVRFactory, XGBoostFactory
from my_logger_pkg import my_logger
from typing import Dict, Tuple
from model_orchestration.model_manager import ModelManager
from auxiliary import DataOrganizer, MetricsImpl
from typing import Dict, Optional
import os
from fastapi import HTTPException
import requests as rq
from seasonality import NoSeasonalityStrategy, SeasonMaskStrategy
from classes import SeasonalityStrategy, Model, GlobalInfoRF, GlobalInfo

HOSTNAME = os.environ.get('DOCKER_DEP', 'data-preprocess')
PORT = os.environ.get('DOCKER_DEP_PORT', '8001')
    

class ModelPipeline:
    def __init__(self, state_manager: ModelManager):
        self.state = state_manager
        self._data_organizer: Optional[DataOrganizer] = self.initialize_organizer()
        self.model_factories: Dict[str, ModelFactory] = {
            "random_forest": RandomForestFactory(),
            "svr": SVRFactory(),
            "xgboost": XGBoostFactory(),
        }


    def initialize_organizer(self)->Optional[DataOrganizer]:
        try:
            res = rq.get((f"http://{HOSTNAME}:{PORT}/final_selection"))
            final_buildings: Dict = res.json()["response"]
            
            per_building: Dict[int, pd.DataFrame] = {}
            for category, building_id in final_buildings.items():
                my_logger.info(f"Fetching data for building {building_id} ({category})")
                res = rq.get(f"http://{HOSTNAME}:{PORT}/building/{building_id}")
                building_data = res.json()
                df = pd.DataFrame(building_data["data"])
                df.index = pd.to_datetime(building_data["index"])
                per_building[building_id] = df
                
            return DataOrganizer(final_buildings, per_building)
        except Exception as e:
            my_logger.exception(f"An initialization exception occured: {e}")

    
    def model(self, model_name: str, season: Optional[str] = None) -> Dict:
        if model_name not in self.model_factories:
            raise ValueError("Invalid model, not supported!")
        
        season_list = [season] if season else ["autumn", "winter"]
        
        strat: SeasonalityStrategy = SeasonMaskStrategy() if season else NoSeasonalityStrategy()
        
        results = {}
        for current_season in season_list:
            key: Tuple[str, str] = (model_name, current_season)
            model: Optional[Model] = None
            
            try:
                model = self.state.get_model(key)
            except Exception as e:
                my_logger.info(f"Model {model_name} for {current_season} not found, creating new one")
                try:
                    model = self.model_factories[model_name].create(self._data_organizer.get_dataset_b(), strat)
                    self.state.add_model(key, model)
                    my_logger.info(f"Added model: {model_name}, season: {current_season}")
                except Exception as e:
                    my_logger.exception(f"Failed to create model: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to train model for {current_season}: {str(e)}")
            
            dataset_b: Dict = self._data_organizer.get_dataset_b()
            for building_id in dataset_b:
                pred, test, grid = model.fit(building_id, current_season)
                
                base_metrics = MetricsImpl.get_metrics(pred, test)
                if model_name == "random_forest":
                    results[(building_id, current_season)] = GlobalInfoRF(
                        metrics=base_metrics, 
                        p_importance=model.permutation_importance(building_id, current_season, grid),
                        f_importance=model.feature_importance(building_id, current_season, grid)
                    )
                else:
                    results[(building_id, current_season)] = GlobalInfo(
                        metrics=base_metrics
                    )
        
        return results