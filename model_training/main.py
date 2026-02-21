from my_logger_pkg import my_logger
from fastapi import FastAPI, HTTPException, Depends
from auxiliary import DataOrganizer, MetricsImpl
from classes import SeasonalityStrategy, HealthCheck, ModelResponse, Model, GlobalInfoRF, ModelConfigparams, RFGridParamsConfig
from seasonality import NoSeasonalityStrategy, SeasonMaskStrategy
from starlette import status
from random_forest import RandomForestImpl
from typing import Dict,Tuple,Optional
from model_orchestration import ModelPipeline, ModelManager

app = FastAPI(
    title="Model Training API",
    description="Backend API for model training",
    version="1.0.0"
)

_state_manager = ModelManager()

def get_pipeline() -> ModelPipeline:
    return ModelPipeline(_state_manager)

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
def rf_model(season: Optional[str] = None, manager: ModelManager = Depends(get_pipeline)) -> ModelResponse:
    try:
        results = manager.model("random_forest", season)
    except Exception as e:
        my_logger.exception(f"RF exception occured!")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
    return ModelResponse(status="RF training completed successfully!", response=results)
            
@app.get("/svr", tags=["model_training"], response_model=ModelResponse)
def svr_model(season: Optional[str] = None, manager: ModelManager = Depends(get_pipeline)) -> ModelResponse:
    try:
        results = manager.model("svr", season)
    except Exception as e:
        my_logger.exception(f"SVR exception occured!")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
    return ModelResponse(status="SVR training completed successfully!", response=results)