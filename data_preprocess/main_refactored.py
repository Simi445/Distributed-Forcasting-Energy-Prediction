from .my_logger_pkg import my_logger
from fastapi import FastAPI, Depends
from starlette import status
from .classes import ModelBuildings, State, Loader, Merger, Engineer, Selection, HealthCheck, LoadResponse, MergeResponse, \
FeatureEngineerResponse, ACFResponse, FeatureEngineerConfig, MedianResponse, Plot, PlotResponse
from .data_pipeline import DataLoader, FeatureEngineer, DataMerger
from .data_selection import EnhancedSelection, ModelBSelection
from .data_plot import PlotHelper
from typing import Dict, List, Any, Tuple
import pandas as pd

class StateManager(State):
    def __init__(self):
        self.states: Dict[str, Any] = {}

    def add_state(self, name: str, value: Any) -> None:
        self.states[name] = value

    def check_state(self, name: str) -> bool:
        return name in self.states

    def get_state(self, name: str) -> Any:
        return self.states.get(name)

    def get_median_acf_data(self) -> tuple[List[str], List[float]]:
        median_buildings = self.get_state("median_buildings")
        if median_buildings is None:
            raise ValueError("median_buildings is not set")
        categories: List[str] = list(median_buildings.keys())
        values: List[float] = [acf_value for _, acf_value in median_buildings.values()]
        return categories, values

class PreprocessingPipeline:    
    def __init__(self, state_manager: State):
        self.state = state_manager

    def ensure_loaded(self) -> None:
        if not self.state.check_state("loader"):
            loader: Loader = DataLoader()
            self.state.add_state("loader", loader)
        my_logger.info("Data loaded")

    def ensure_merged(self) -> pd.DataFrame:
        if self.state.check_state("full_df"):
            return self.state.get_state("full_df")
        self.ensure_loaded()
        if not self.state.check_state("merger"):
            loader = self.state.get_state("loader")
            merger: Merger = DataMerger(loader)
            self.state.add_state("merger", merger)
        merger = self.state.get_state("merger")
        merger.merge_tables()
        full_df = merger.get_full_df()
        self.state.add_state("full_df", full_df)
        my_logger.info("Data merged")
        return full_df

    def ensure_engineered(self) -> Dict[int, pd.DataFrame]:
        if self.state.check_state("per_building"):
            return self.state.get_state("per_building")
        full_df = self.ensure_merged()
        if not self.state.check_state("engineer"):
            engineer: Engineer = FeatureEngineer()
            self.state.add_state("engineer", engineer)
        engineer = self.state.get_state("engineer")
        per_building: Dict[int, pd.DataFrame] = engineer.feature_engineer_buildings(full_df, FeatureEngineerConfig())
        self.state.add_state("per_building", per_building)
        my_logger.info("Features engineered")
        return per_building

    def ensure_acf_calculated(self) -> Tuple[List[str], List[float]]:
        if self.state.check_state("median_buildings"):
            return self.state.get_median_acf_data()
        per_building = self.ensure_engineered()
        if not self.state.check_state("median_acf"):
            acf: Selection = EnhancedSelection()
            self.state.add_state("median_acf", acf)
        median_acf = self.state.get_state("median_acf")
        building_type: Dict[int, str] = median_acf.category_buildings(per_building)
        self.state.add_state("building_type", building_type)
        lag_acf: Tuple[Dict[str, Any], Dict[int, float]] = median_acf.calculate_lag_acf(building_type, per_building)
        acf24_per_type, acf24_per_building = lag_acf
        self.state.add_state("acf24_per_type", acf24_per_type)
        self.state.add_state("acf24_per_building", acf24_per_building)
        median_buildings = median_acf.median_per_type(acf24_per_type)
        self.state.add_state("median_buildings", median_buildings)
        my_logger.info("ACF calculated")
        return self.state.get_median_acf_data()

    def ensure_final_selection(self) -> Dict[str, int]:
        if self.state.check_state("final_buildings"):
            return self.state.get_state("final_buildings")
        self.ensure_acf_calculated()
        per_building = self.state.get_state("per_building")
        median_buildings = self.state.get_state("median_buildings")
        model_build: ModelBuildings = ModelBSelection()
        final_buildings = model_build.select_final_buildings(per_building, median_buildings)
        self.state.add_state("final_buildings", final_buildings)
        my_logger.info("Final selection completed")
        return final_buildings

app_state = StateManager()
pipeline = PreprocessingPipeline(app_state)

app = FastAPI(
    title="Data Preprocessing API",
    description="Backend API for preprocessing the dataset",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "Data Preprocessing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "load_data": "/load_data",
            "merge_data": "/merge_data",
            "feature_engineer": "/feature_engineer",
            "acf": "/acf",
            "final_selection": "/final_selection",
            "plot": "/plot"
        }
    }

@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="HTTP Status Code 200",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    my_logger.info("Health check requested")
    return HealthCheck(status="OK")

@app.get("/load_data", tags=["preprocessing"], response_model=LoadResponse)
def load_data(pipeline: PreprocessingPipeline = Depends(lambda: pipeline)) -> LoadResponse:
    pipeline.ensure_loaded()
    return LoadResponse(status="Data loaded")

@app.get("/merge_data", tags=["preprocessing"], response_model=MergeResponse)
def merge_data(pipeline: PreprocessingPipeline = Depends(lambda: pipeline)) -> MergeResponse:
    full_df = pipeline.ensure_merged()
    return MergeResponse(status="Data merged", shape=full_df.shape)

@app.get("/feature_engineer", tags=["preprocessing"], response_model=FeatureEngineerResponse)
def feature_engineer(pipeline: PreprocessingPipeline = Depends(lambda: pipeline)) -> FeatureEngineerResponse:
    per_building = pipeline.ensure_engineered()
    return FeatureEngineerResponse(status="Features engineered", buildings=list(per_building.keys()))

@app.get("/acf", tags=["preprocessing"], response_model=ACFResponse)
def get_acf(pipeline: PreprocessingPipeline = Depends(lambda: pipeline)) -> ACFResponse:
    categories, values = pipeline.ensure_acf_calculated()
    return ACFResponse(status="Median ACF categories calculated", graph_data={"acf_data": (categories, values)})

@app.get("/final_selection", tags=["preprocessing"], response_model=MedianResponse)
def get_final_selection(pipeline: PreprocessingPipeline = Depends(lambda: pipeline)) -> MedianResponse:
    final_buildings = pipeline.ensure_final_selection()
    return MedianResponse(status="Final selection completed", response=final_buildings)

@app.get("/plot", tags=["preprocessing"], response_model=PlotResponse)
def get_plot(pipeline: PreprocessingPipeline = Depends(lambda: pipeline)) -> PlotResponse:
    final_buildings = pipeline.ensure_final_selection()
    plot_helper: Plot = PlotHelper(final_buildings)
    plots: Dict[str, str] = {}
    if pipeline.state.check_state("acf24_per_building") and pipeline.state.check_state("per_building"):
        plots["acf"] = plot_helper.plot_acf(pipeline.state.get_state("acf24_per_building"))
        plots["energy"] = plot_helper.plot_energy(pipeline.state.get_state("per_building"))
    return PlotResponse(status="Plots plotted!", response=plots)