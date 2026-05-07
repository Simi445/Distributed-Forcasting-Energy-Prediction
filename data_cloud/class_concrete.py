from typing import Any, Dict, Optional

from classes import BulkConverter,Storage
import pandas as pd
import boto3
import os
import glob
from my_logger_pkg import my_logger
from botocore.exceptions import ClientError
import base64
import configparser

class CloudFlareR2:
    _instance = None

    @staticmethod
    def get_instance():
        if CloudFlareR2._instance is None:
            CloudFlareR2._instance = CloudFlareR2()
        return CloudFlareR2._instance

    def __init__(self)->None:
        self.bucket_name = os.environ.get('R2_BUCKET_NAME', 'my-bucket-sim')
        cred_file:str = os.environ.get('CRED_FILE', '/root/.cloudflare/credentials')
        if os.path.exists(cred_file):
            config = configparser.ConfigParser()
            config.read(cred_file)
            if 'default' in config:
                os.environ['AWS_ACCESS_KEY_ID'] = config['default'].get('aws_access_key_id', '')
                os.environ['AWS_SECRET_ACCESS_KEY'] = config['default'].get('aws_secret_access_key', '')
                os.environ['AWS_DEFAULT_REGION'] = config['default'].get('region', 'us-east-1')
            my_logger.info("Loaded CF credentials from cred file.")
        else:
            my_logger.warning("Cred file not found; using existing environment variables or IAM roles.")
        
        self.endpoint_url = os.environ.get('R2_ENDPOINT_URL', 'https://39209c3b1c9ee2eaac1dd5c876180d59.r2.cloudflarestorage.com')

        try:
            self.client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                region_name='auto'
            )
            self.bucket = boto3.resource(
                's3',
                endpoint_url=self.endpoint_url,
                region_name='auto'
            ).Bucket(self.bucket_name)
            my_logger.info("Initialized Cloudflare R2 client and bucket.")
        except Exception as e:
            my_logger.exception(f"Failed to initialize CloudFlareR2: {e}")
            raise

    def get_bucket_name(self) -> str:
        if hasattr(self, 'bucket_name'):
            return self.bucket_name
        return None

    def get_client(self):
        if hasattr(self, 'client'):
            return self.client
        return None

    def get_bucket(self):
        if hasattr(self, 'bucket'):
            return self.bucket
        return None

class CsvToParquet(BulkConverter):
    def __init__(self,**kwargs)->None:
        names:Dict = kwargs
        try:
            self.locations: dict[str,str] = self.__get_migration_files(kwargs)
            
            self.train:pd.DataFrame = pd.read_csv(
                self.locations[names.train],
                delimiter=',',
                dtype={
                    'building_id': 'int32',
                    'meter': 'int8'
                },
                parse_dates=['timestamp']
            )
            my_logger.info(f"Loaded train dataframe with {len(self.train)} rows.")
            self.meta:pd.DataFrame = pd.read_csv(
                self.locations[names.meta],
                delimiter=',',
                dtype={
                    'building_id': 'int32',
                    'site_id': 'int8',  
                }
            )
            my_logger.info(f"Loaded meta dataframe with {len(self.meta)} rows.")
            self.weather:pd.DataFrame = pd.read_csv(
                self.locations[names.weather],
                delimiter=',',
                dtype={
                    'site_id': 'int8'
                },
                parse_dates=['timestamp'])
            my_logger.info(f"Loaded weather dataframe with {len(self.weather)} rows.")
        except Exception as e:
            my_logger.exception(f'An exception occurred: {e}')

    def __detect_file_changes(self, file1:str, file2:str) -> bool:
        file1_mod: float = os.path.getmtime(file1)
        file2_mod: float = os.path.getmtime(file2)
        return file1_mod > file2_mod

    def data_to_data(self, location:str = 'parquet', extension:str = 'parquet', **kwargs) -> list[str]:
        created_files: list[str] = []
        dfs = {value: getattr(self, key) for key, value in kwargs.items()}
        for _, value in kwargs.items():
            path = f'{location}/{value}.{extension}'
            if not os.path.isfile(path) or self.__detect_file_changes(self.locations[value], path):
                dfs[value].to_parquet(path)
                my_logger.info(f"Created the parquet file at {path}.")
                created_files.append(path)
        return created_files

    
    def __get_migration_files(self,items:Dict, csv_location:str ='csv_files') -> dict[str]:
        res:dict[str,str] = {}
        for key,value in items.items():
            files = glob.glob(f'{csv_location}/{value}.csv', recursive=True)
            if not files:
                raise ValueError(f"No files found for {key}: {value}")
            res[value] = files[0]
            my_logger.info(f"Found file for {key}: {files[0]}")
        return res


class ConcreteParquet(Storage):
    def __init__(self, bucket_name:str, client:Any,bucket:Any):
        self.client:Any = client
        self.bucket_name:str = bucket_name
        self.bucket:Any = bucket

    def upload_file_bucket(self, file_name:str, object_name:str = None)->bool:
        if object_name is None:
            object_name = os.path.basename(file_name)
        try:
            self.client.upload_file(file_name,self.bucket_name,object_name)
            my_logger.info(f"Successfully uploaded {file_name} to {self.bucket_name} as {object_name}.")
            return True
        except ClientError as e:
            my_logger.exception(e)
            return False
    
    def read_file_bucket(self,file_name:str)->Optional[str]:
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=file_name)
            body = response['Body'].read()
            encoded_body:str = base64.b64encode(body).decode('utf-8')
            my_logger.info(f"Read and encoded parquet file {file_name} from R2.")
            return encoded_body
        except Exception as e:
            my_logger.exception(f"An exception occurred when reading file from R2: {e}")
            return None