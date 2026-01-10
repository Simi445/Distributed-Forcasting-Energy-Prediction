from ..my_logger_pkg import my_logger
from ..classes import ModelBuildings, State, Loader, Merger, Engineer, Selection, FeatureEngineerConfig
from ..data_pipeline import DataLoader, FeatureEngineer, DataMerger
from ..data_selection import EnhancedSelection, ModelBSelection
from typing import Dict, List, Any, Tuple
import pandas as pd

class StateManager(State):
    def __init__(self):
        self.states: Dict[str, Any] = {}

    def add_state(self, name: str, value: Any) -> None:
        self.states[name] = value

    def get_state(self, name: str) -> Any:
        return self.states.get(name)

class PreprocessingPipeline:    
    def __init__(self, state_manager: State):
        self.state = state_manager

    def ensure_loaded(self) -> None:
        my_logger.info("Load data endpoint called")
        if self.state.get_state("loader") is None:
            loader: Loader = DataLoader()
            self.state.add_state("loader", loader)
        my_logger.info("Data loaded successfully")

    def ensure_merged(self) -> pd.DataFrame:
        my_logger.info("Merge data endpoint called")

        full_df = self.state.get_state("full_df")
        if full_df is not None:
            my_logger.info("Returning cached merged data")
            return full_df
        
        self.ensure_loaded()

        merger = self.state.get_state("merger")
        if merger is None:
            loader = self.state.get_state("loader")
            merger: Merger = DataMerger(loader)
            self.state.add_state("merger", merger)
        else:
            merger: Merger = merger

        merger.merge_tables()
        full_df = merger.get_full_df()
        self.state.add_state("full_df", full_df)
        my_logger.info("Data merged successfully")
        return full_df

    def ensure_engineered(self) -> Dict[int, pd.DataFrame]:
        my_logger.info("Feature engineer endpoint called")
        
        per_building = self.state.get_state("per_building")
        if per_building is not None:
            return per_building
        
        full_df = self.ensure_merged()

        engineer = self.state.get_state("engineer")
        if engineer is None:
            engineer: Engineer = FeatureEngineer()
            self.state.add_state("engineer", engineer)
        else:
            engineer: Engineer = engineer

        per_building: Dict[int, pd.DataFrame] = engineer.feature_engineer_buildings(full_df, FeatureEngineerConfig())
        self.state.add_state("per_building", per_building)

        my_logger.info("Features engineered successfully")
        return per_building

    def ensure_acf_calculated(self) -> Tuple[List[str], List[float]]:
        my_logger.info("ACF endpoint called")
        
        median_buildings = self.state.get_state("median_buildings")
        if median_buildings is not None:
            my_logger.info("Returning cached ACF data")
            return self.extract_acf_data(median_buildings)
        
        per_building = self.ensure_engineered()

        median_acf = self.state.get_state("median_acf")
        if median_acf is None:
            median_acf: Selection = EnhancedSelection()
            self.state.add_state("median_acf", median_acf)
        else:
            median_acf: Selection = median_acf

        building_type: Dict[int, str] = median_acf.category_buildings(per_building)
        self.state.add_state("building_type", building_type)

        lag_acf: Tuple[Dict[str, Any], Dict[int, float]] = median_acf.calculate_lag_acf(building_type, per_building)
        acf24_per_type, acf24_per_building = lag_acf

        self.state.add_state("acf24_per_type", acf24_per_type)
        self.state.add_state("acf24_per_building", acf24_per_building)

        median_buildings = median_acf.median_per_type(acf24_per_type)
        self.state.add_state("median_buildings", median_buildings)

        my_logger.info("ACF calculated successfully")
        return self.extract_acf_data(median_buildings)
    
    def extract_acf_data(self, median_buildings: Dict[str, Tuple[int, float]]) -> Tuple[List[str], List[float]]:
        categories: List[str] = list(median_buildings.keys())
        values: List[float] = [acf_value for _, acf_value in median_buildings.values()]
        return categories, values

    def ensure_final_selection(self) -> Dict[str, int]:
        my_logger.info("Final selection endpoint called")

        final_buildings = self.state.get_state("final_buildings")
        if final_buildings is not None:
            my_logger.info("Returning cached final selection")
            my_logger.info(f"The final buildings: {final_buildings}")
            return final_buildings
        
        self.ensure_acf_calculated()

        per_building = self.state.get_state("per_building")
        median_buildings = self.state.get_state("median_buildings")
        model_build: ModelBuildings = ModelBSelection()

        final_buildings = model_build.select_final_buildings(per_building, median_buildings)
        self.state.add_state("final_buildings", final_buildings)

        my_logger.info("Final selection completed successfully")
        my_logger.info(f"The final buildings: {final_buildings}")
        return final_buildings