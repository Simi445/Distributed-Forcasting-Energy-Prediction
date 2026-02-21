from classes import Converter,Storage
from classes.models import FileNames
import pandas as pd
import boto3
import os
import glob
from my_logger_pkg import my_logger
from botocore.exceptions import ClientError
import base64
import configparser

class CsvToParquet(Converter):
    def __init__(self,names:FileNames)->None:
        self.locations: dict[str,str] = self.__get_migration_files(names)
        
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

    def __detect_file_changes(self, file1:str, file2:str) -> bool:
        file1_mod: float = os.path.getmtime(file1)
        file2_mod: float = os.path.getmtime(file2)
        return file1_mod > file2_mod

    def data_to_parquet(self, names: FileNames) -> list[str]:
        created_files: list[str] = []
        dfs = {names.train: self.train, names.meta: self.meta, names.weather: self.weather}

        for _, value in names.model_dump().items():
            path = f'parquet/{value}.parquet'
            if not os.path.isfile(path) or self.__detect_file_changes(self.locations[value], path):
                dfs[value].to_parquet(path)
                my_logger.info(f"Created the parquet file at {path}.")
                created_files.append(path)

        return created_files

    
    def __get_migration_files(self, names:FileNames) -> dict[str]:
        res:dict[str,str] = {}
        for key,value in names.model_dump().items():
            files = glob.glob(f'csv_files/{value}.csv', recursive=True)
            if not files:
                raise ValueError(f"No files found for {key}: {value}")
            res[value] = files[0]
            my_logger.info(f"Found file for {key}: {files[0]}")
        return res
    

class CloudFlareR2(Storage):
    def __init__(self,bucket_name:str)->None:
        cred_file = '/root/.cloudflare/credentials'
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
        
        self.bucket_name = bucket_name
        self.endpoint_url = os.environ.get('R2_ENDPOINT_URL', 'https://39209c3b1c9ee2eaac1dd5c876180d59.r2.cloudflarestorage.com')
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            region_name='auto'
        )
        self.bucket = boto3.resource(
            's3',
            endpoint_url=self.endpoint_url,
            region_name='auto'
        ).Bucket(bucket_name)
        my_logger.info("Initialized Cloudflare R2 client and bucket.")

    def upload_parquet_file(self, file_name:str, object_name:str = None)->bool:
        if object_name is None:
            object_name = os.path.basename(file_name)
        try:
            self.client.upload_file(file_name,self.bucket_name,object_name)
            my_logger.info(f"Successfully uploaded {file_name} to {self.bucket_name} as {object_name}.")
            return True
        except ClientError as e:
            my_logger.exception(e)
            return False
    
    def read_parquet_files(self)->dict[str,str]:
        parquet_files:dict[str,str] = {}
        for obj in self.bucket.objects.all():
            key = obj.key
            if ".parquet" not in key:
                continue
            body = obj.get()['Body'].read()
            encoded_body:str = base64.b64encode(body).decode('utf-8')
            parquet_files[key.replace('.parquet', '')] = encoded_body
        my_logger.info(f"Read and encoded {len(parquet_files)} parquet files from S3.")
        return parquet_files