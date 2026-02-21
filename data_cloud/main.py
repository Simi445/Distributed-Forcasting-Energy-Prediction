from classes import Converter, Storage, FileNames, HealthCheck, DownloadResponse, UploadResponse
from class_concrete import CsvToParquet, CloudFlareR2
from my_logger_pkg import my_logger
from fastapi import FastAPI, HTTPException
from starlette import status

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
    "/download_files",
    tags=["download_files"],
    summary="Perform the data download",
    response_description="HTTP Status Code 200",
    status_code=status.HTTP_200_OK,
    response_model=DownloadResponse,
)
async def download_files() -> DownloadResponse:
    try:
        storage: Storage = CloudFlareR2("my-bucket-sim")
        parquet_files:dict[str,str] = storage.read_parquet_files()
        my_logger.info(f"Finished downloading.")
    except Exception as e:
        my_logger.exception(f'An exception occurred: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    return DownloadResponse(status="Download operation successful!", response=parquet_files)

@app.post(
    "/upload_files",
    tags=["upload_files"],
    summary="Perform the data upload",
    response_description="HTTP Status Code 200",
    status_code=status.HTTP_200_OK,
    response_model=UploadResponse,
)
async def upload_files() -> UploadResponse:
    try:
        names = FileNames(train='train', meta='building_metadata', weather='weather_train')
        my_logger.info(f"Created the names object with values {names.model_dump()}.")
        converter: Converter = CsvToParquet(names)
        my_logger.info(f"Created the parquet object.")
        parquet_locations:list[str] = converter.data_to_parquet(names)
        storage: Storage = CloudFlareR2("my-bucket-sim")
        for v in parquet_locations:
            storage.upload_parquet_file(file_name=v)
            my_logger.info(f"File {v} uploaded in s3.")
        my_logger.info(f"Finished execution.")
    except Exception as e:
        my_logger.exception(f'An exception occurred: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    return UploadResponse(status="Upload operation successful!")