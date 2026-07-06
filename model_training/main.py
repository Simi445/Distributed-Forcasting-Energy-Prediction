from my_logger_pkg import my_logger
from fastapi import FastAPI, HTTPException, Depends
from auxiliary import ModelManager, DefaultFactoryProvider, DataOrganizer, BaseLineDataOrganizer
from classes import HealthCheck, ModelResponse, Pipeline, SeasonConfig
from starlette import status
from non_baseline import NonBaselinePipeline, RandomForestFactory, SVRFactory, XGBoostFactory
from baseline import BaselinePipeline, NaiveFactory, MeanFactory
from typing import Optional

app = FastAPI(
    title="Model Training API",
    description="Backend API for model training",
    version="1.0.0"
)

_non_baseline_factory_provider = DefaultFactoryProvider({
      "random_forest": RandomForestFactory(),
      "svr": SVRFactory(),
      "xgboost": XGBoostFactory(),
  })

_baseline_factory_provider = DefaultFactoryProvider({
      "naive": NaiveFactory(),
      "mean": MeanFactory(),
  })


class LazyOrganizer:
    def __init__(self, factory, name: str):
        self._factory = factory
        self._name = name
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            my_logger.info(f"Initializing {self._name} organizer on first model request")
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self._get_instance(), name)


_state_manager_non_baseline = ModelManager()
_state_manager_baseline = ModelManager()

_non_baseline_organizer = LazyOrganizer(DataOrganizer, "non-baseline")
_baseline_organizer = LazyOrganizer(BaseLineDataOrganizer, "baseline")

_non_baseline_pipeline = NonBaselinePipeline(_non_baseline_factory_provider, _state_manager_non_baseline, _non_baseline_organizer)
_baseline_pipeline = BaselinePipeline(_baseline_factory_provider, _state_manager_baseline, _baseline_organizer)

def get_non_baseline_pipeline() -> Pipeline:
    return _non_baseline_pipeline

def get_baseline_pipeline() -> Pipeline:
    return _baseline_pipeline

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
