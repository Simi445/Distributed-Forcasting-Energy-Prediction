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

app = FastAPI(
    title="Model Training API",
    description="Backend API for model training",
    version="1.0.0"
)

# _state_manager = StateManager()

# def get_pipeline() -> PreprocessingPipeline:
#     return PreprocessingPipeline(_state_manager)

@app.get("/")
async def root():
    return {
        "message": "Model Training API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
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


# @app.get("/load_data", tags=["preprocessing"], response_model=LoadResponse)
# def load_data(pipeline: PreprocessingPipeline = Depends(get_pipeline)) -> LoadResponse:
#     try:
#         pipeline.ensure_loaded()
#         return LoadResponse(status="Data loaded")
#     except Exception as e:
#         my_logger.exception(f"Error loading data: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")


@app.get("/random_forest", tags=["model_training"], response_model=LoadResponse)
def load_data(pipeline: PreprocessingPipeline = Depends(get_pipeline)) -> LoadResponse:
    try:
        pipeline.ensure_loaded()
        return LoadResponse(status="Data loaded")
    except Exception as e:
        my_logger.exception(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")