from typing import Dict, Optional
import os
import base64
from classes import BulkConverter,Storage
from my_logger_pkg import my_logger
from class_concrete import CsvToParquet,CloudFlareR2,ConcreteParquet


class DataFacade:
    def __init__(self):
        self.cfr2:CloudFlareR2 = CloudFlareR2.get_instance()
        self.client = self.cfr2.get_client()
        self.bucket_name = self.cfr2.get_bucket_name()
        self.bucket = self.cfr2.get_bucket()

    def upload_parquet_files(self) -> list[str]:
        converter: BulkConverter = CsvToParquet(train='train', meta='building_metadata', weather='weather_train')
        my_logger.info(f"Created the parquet object.")
        parquet_locations:list[str] = converter.data_to_data(train='train', meta='building_metadata', weather='weather_train')
        storage: Storage = ConcreteParquet(self.bucket_name, self.client,self.bucket)
        uploaded_files:list[str] = []

        for v in parquet_locations:
            upload_status:bool = storage.upload_file_bucket(file_name=v)
            if upload_status:
                uploaded_files.append(v)
            else:
                my_logger.warning(f"File {v} didn't get uploaded in R2.")

        my_logger.info(f"Finished execution. Uploaded {len(uploaded_files)} files.")
        return uploaded_files

    def upload_grid(self, model_file:str) -> None:
        storage: Storage = ConcreteParquet(self.bucket_name, self.client,self.bucket)
        file_path = f'/app/grids/{model_file}'
        upload_status:bool = storage.upload_file_bucket(file_name=file_path, object_name=model_file)
        if upload_status:
            my_logger.info(f"Finished execution. Uploaded grid file for {model_file}.")
        else:
            my_logger.warning(f"Grid file {model_file} didn't get uploaded in R2.")

    def download_grid(self, model_file:str) -> Optional[bytes]:
        storage: Storage = ConcreteParquet(self.bucket_name, self.client,self.bucket)
        for obj in self.bucket.objects.all():
            if obj.key == model_file:
                content = storage.read_file_bucket(obj.key)
                my_logger.info(f"Found grid file {model_file}.")
                grid_path = f'/app/grids/{model_file}'
                os.makedirs('/app/grids', exist_ok=True)
                with open(grid_path, 'wb') as f:
                    f.write(base64.b64decode(content))
                my_logger.info(f"Saved grid file to {grid_path}")
                return base64.b64decode(content)
        my_logger.warning(f"Failed to find grid file {model_file}.")
        return None

    def download_parquet_files(self) -> Dict[str, str]:
        storage: Storage = ConcreteParquet(self.bucket_name, self.client,self.bucket)
        parquet_files:dict[str,str] = {}
        for obj in self.bucket.objects.all():
            key = obj.key

            if not key.endswith('.parquet'):
                continue
            content = storage.read_file_bucket(key)
            if content:
                parquet_files[key] = content
            else:
                my_logger.warning(f"Failed to read or empty content for {key}.")
        my_logger.info(f"Read and encoded {len(parquet_files)} parquet files from R2.")
        my_logger.info(f"Finished downloading.")
        return parquet_files