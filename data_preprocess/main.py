from .my_logger_pkg import my_logger
from fastapi import FastAPI, HTTPException, Depends
from starlette import status
from .classes import ModelBuildings, State, Loader, Merger, Engineer, Selection, HealthCheck, LoadResponse, MergeResponse, \
FeatureEngineerResponse, ACFResponse, FeatureEngineerConfig, MedianResponse, Plot, PlotResponse
from .data_pipeline import DataLoader, FeatureEngineer, DataMerger
from .data_selection import EnhancedSelection, ModelBSelection
from .data_plot import PlotHelper
from typing import Dict, List, Any, Tuple
import pandas as pd
from .data_orchestration import StateManager, PreprocessingPipeline

app = FastAPI(
    title="Data Preprocessing API",
    description="Backend API for preprocessing the dataset",
    version="1.0.0"
)

_state_manager = StateManager()

def get_pipeline() -> PreprocessingPipeline:
    return PreprocessingPipeline(_state_manager)

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
def load_data(pipeline: PreprocessingPipeline = Depends(get_pipeline)) -> LoadResponse:
    try:
        pipeline.ensure_loaded()
        return LoadResponse(status="Data loaded")
    except Exception as e:
        my_logger.exception(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")

@app.get("/merge_data", tags=["preprocessing"], response_model=MergeResponse)
def merge_data(pipeline: PreprocessingPipeline = Depends(get_pipeline)) -> MergeResponse:
    try:
        full_df = pipeline.ensure_merged()
        return MergeResponse(status="Data merged", shape=full_df.shape)
    except Exception as e:
        my_logger.exception(f"Error merging data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to merge data: {str(e)}")


@app.get("/feature_engineer", tags=["preprocessing"], response_model=FeatureEngineerResponse)
def feature_engineer(pipeline: PreprocessingPipeline = Depends(get_pipeline)) -> FeatureEngineerResponse:
    try:
        per_building = pipeline.ensure_engineered()
        return FeatureEngineerResponse(status="Features engineered", buildings=list(per_building.keys()))
    except Exception as e:
        my_logger.exception(f"Error in feature engineering: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to engineer features: {str(e)}")

@app.get("/acf", tags=["preprocessing"], response_model=ACFResponse)
def get_acf(pipeline: PreprocessingPipeline = Depends(get_pipeline)) -> ACFResponse:
    try:
        categories, values = pipeline.ensure_acf_calculated()
        return ACFResponse(status="Median ACF categories calculated", graph_data={"acf_data": (categories, values)})
    except Exception as e:
        my_logger.exception(f"Error calculating ACF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate ACF: {str(e)}")


@app.get("/final_selection", tags=["preprocessing"], response_model=MedianResponse)
def get_final_selection(pipeline: PreprocessingPipeline = Depends(get_pipeline)) -> MedianResponse:
    try:
        final_buildings = pipeline.ensure_final_selection()
        return MedianResponse(status="Final selection completed", response=final_buildings)
    except Exception as e:
        my_logger.exception(f"Error in final selection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to finalize building selection: {str(e)}")


@app.get("/plot", tags=["preprocessing"], response_model=PlotResponse)
def get_plot(pipeline: PreprocessingPipeline = Depends(get_pipeline)) -> PlotResponse:
    try:
        final_buildings = pipeline.ensure_final_selection()

        plot_helper: Plot = PlotHelper(final_buildings)
        plots: Dict[str, str] = {}
        
        acf24_per_building = pipeline.state.get_state("acf24_per_building")
        per_building = pipeline.state.get_state("per_building")
        if acf24_per_building is not None and per_building is not None:
            plots["acf"] = plot_helper.plot_acf(acf24_per_building)
            plots["energy"] = plot_helper.plot_energy(per_building)
        return PlotResponse(status="Plots plotted!", response=plots)
    except Exception as e:
        my_logger.exception(f"Error generating plots: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate plots: {str(e)}")
