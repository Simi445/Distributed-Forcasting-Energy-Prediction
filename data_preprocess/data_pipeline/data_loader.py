import requests
import base64
import io
import pandas as pd
from ..my_logger_pkg import my_logger
from fastapi import HTTPException
from typing import Optional, Dict
from ..classes import Loader

class DataLoader(Loader):
    def __init__(self, ip:str = "data-cloud", port:str = "8000"):
        try:
            data: Dict[str, str] = self.__download_request(ip, port)

            counter: int = 0
            while not data:
                if counter > 3:
                    raise ConnectionError("Failed to fetch data from S3 after 3 retry attempts. Data upload may have failed.")
                my_logger.warning(f"No data found, attempting upload and retry (attempt {counter + 1}/3)")
                self.__upload_request(ip, port)
                data: Dict[str, str] = self.__download_request(ip, port)
                counter += 1

            self.datasets: Dict[str, pd.DataFrame] = {}  
            for filename, encoded_data in data.items():
                file_bytes = base64.b64decode(encoded_data)
                self.datasets[filename] = pd.read_parquet(io.BytesIO(file_bytes))
            my_logger.info(f"Data loader initialized successfully with {len(self.datasets)} datasets")
        except ConnectionError as e:
            my_logger.exception(f"Connection error in DataLoader: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            my_logger.exception(f"An exception occurred in DataLoader constructor: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    def __download_request(self, ip: str, port: str) -> Dict[str, str]:
        my_logger.info("Trying to download data...")
        response: requests.Response = requests.get(f"http://{ip}:{port}/download_dataset_files")
        response.raise_for_status()
        my_logger.info("Data downloaded.")
        return response.json()["response"] 
    
    def __upload_request(self, ip:str, port:str)->None:
        my_logger.info("No data found in bucket, calling for upload...")
        response:requests.Response = requests.post(f"http://{ip}:{port}/upload_dataset_files")
        response.raise_for_status()
        my_logger.info("Data uploaded!")
    
    def get_dataset(self, data_category: str) -> Optional[pd.DataFrame]:
        for key, df in self.datasets.items():
            if data_category in key:
                return df
        raise ValueError(f"No such '{data_category}' dataset exists")