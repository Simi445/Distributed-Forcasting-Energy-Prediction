from my_logger_pkg import my_logger
from fastapi import FastAPI, HTTPException, Depends
from auxiliary import MetricsImpl, ModelManager
from classes import HealthCheck, ModelResponse, Pipeline, SeasonConfig
from starlette import status
from non_baseline import NonBaselinePipeline
from typing import Optional
from baseline import BaselinePipeline

app = FastAPI(
    title="Model Training API",
    description="Backend API for model training",
    version="1.0.0"
)


_state_manager_non_baseline = ModelManager()
_state_manager_baseline = ModelManager()

def get_non_baseline_pipeline() -> Pipeline:
    return NonBaselinePipeline(_state_manager_non_baseline)

def get_baseline_pipeline() -> Pipeline:
    return BaselinePipeline(_state_manager_baseline)

@app.get("/")
async def root():
    return {
        "message": "Model Training API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "random_forest": "/rf/name"
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


@app.get("/rf", tags=["model_training"], response_model=ModelResponse)
def rf_model(season: Optional[str] = None, manager: Pipeline = Depends(get_non_baseline_pipeline)) -> ModelResponse:
    try:
        results = manager.run("random_forest", season)
    except Exception as e:
        my_logger.exception(f"RF exception occured!")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
    return ModelResponse(status="RF training completed successfully!", response=results)
            
@app.get("/svr", tags=["model_training"], response_model=ModelResponse)
def svr_model(season: Optional[str] = None, manager: Pipeline = Depends(get_non_baseline_pipeline)) -> ModelResponse:
    try:
        results = manager.run("svr", season)
    except Exception as e:
        my_logger.exception(f"SVR exception occured!")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
    return ModelResponse(status="SVR training completed successfully!", response=results)


@app.get("/xg_boost", tags=["model_training"], response_model=ModelResponse)
def xgboost_model(season: Optional[str] = None, manager: Pipeline = Depends(get_non_baseline_pipeline)) -> ModelResponse:
    try:
        results = manager.run("xgboost", season)
    except Exception as e:
        my_logger.exception(f"XGBoost exception occured!")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
    return ModelResponse(status="XGBoost training completed successfully!", response=results)


@app.get("/naive/{season}", tags=["model_training"], response_model=ModelResponse)
def naive_model(season: str, manager: Pipeline = Depends(get_baseline_pipeline)) -> ModelResponse:
    try:
        results = manager.run("naive", season)
    except Exception as e:
        my_logger.exception(f"Naive exception occured!")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
    return ModelResponse(status="Naive training completed successfully!", response=results)

@app.get("/mean/{season}", tags=["model_training"], response_model=ModelResponse)
def mean_model(season: str, manager: Pipeline = Depends(get_baseline_pipeline)) -> ModelResponse:
    try:
        results = manager.run("mean", season)
    except Exception as e:
        my_logger.exception(f"Mean exception occured!")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
    return ModelResponse(status="Mean training completed successfully!", response=results)