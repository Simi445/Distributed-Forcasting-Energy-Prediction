from typing import Dict, Optional

from numpy import byte
from classes import HealthCheck, DownloadResponse, UploadResponse
from data_facade import DataFacade
from my_logger_pkg import my_logger
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from starlette import status


_state_manager = DataFacade()

def get_methods() -> DataFacade:
    return _state_manager


app = FastAPI(
    title="Data Transfer ASHRAE API",
    description="Backend API for migrating ASHRAE csv data to parquet format in AWS S3",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {
        "message": "Data transfer API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "main": "/main"
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
    return HealthCheck(status="OK")


@app.get(
    "/download_dataset_files",
    tags=["download_dataset_files"],
    summary="Perform the data download",
    response_description="HTTP Status Code 200",
    status_code=status.HTTP_200_OK,
    response_model=DownloadResponse,
)
async def download_dataset_files(facade: DataFacade = Depends(get_methods)) -> DownloadResponse:
    try:
       parquet_files: Dict[str, str] = facade.download_parquet_files()
    except Exception as e:
        my_logger.exception(f'An exception occurred: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    return DownloadResponse(status=f"Download operation successful! Retrieved {len(parquet_files)} files.", response=parquet_files)


@app.post(
    "/upload_dataset_files",
    tags=["upload_dataset_files"],
    summary="Perform the data upload",
    response_description="HTTP Status Code 200",
    status_code=status.HTTP_200_OK,
    response_model=UploadResponse,
)
async def upload_dataset_files(facade: DataFacade = Depends(get_methods)) -> UploadResponse:
    try:
        uploaded_files:list[str] = facade.upload_parquet_files()
    except Exception as e:
        my_logger.exception(f'An exception occurred: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    return UploadResponse(status=f"Upload operation successful! {len(uploaded_files)} files uploaded.")


@app.post(
    "/upload_grid_file/{model_file}",
    tags=["upload_grid_file"],
    summary="Perform the grid upload",
    response_description="HTTP Status Code 200",
    status_code=status.HTTP_200_OK,
    response_model=UploadResponse,
)
async def upload_grid_file(model_file:str, facade: DataFacade = Depends(get_methods)) -> UploadResponse:
    try:
        facade.upload_grid(model_file)
    except Exception as e:
        my_logger.exception(f'An exception occurred: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    return UploadResponse(status=f"Upload operation successful! Grid file {model_file} uploaded.")


@app.get(
    "/download_grid_file/{model_file}",
    tags=["download_grid_file"],
    summary="Perform the grid download",
    response_description="Binary grid file",
    status_code=status.HTTP_200_OK,
)
async def download_grid_file(model_file:str, facade: DataFacade = Depends(get_methods)):
    try:
        grid_path = f'/app/grids/{model_file}'
        result:Optional[byte] = facade.download_grid(model_file)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Grid file {model_file} not found.")
        return FileResponse(path=grid_path, media_type="application/octet-stream", filename=model_file)
    except HTTPException:
        raise
    except Exception as e:
        my_logger.exception(f'An exception occurred: {e}')
        raise HTTPException(status_code=500, detail=str(e))