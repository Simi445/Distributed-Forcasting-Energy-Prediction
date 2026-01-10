from pydantic import BaseModel, field_validator
from typing import Optional

#Misc
class FileNames(BaseModel):
    train: str
    meta: str
    weather: str

    @field_validator("train", "meta", "weather")
    @classmethod
    def check_location_length(cls, value: str):
        if value and len(value) > 25:
            raise ValueError(f'Name \'{value}\' is too long!')
        return value
    
#Health check
class HealthCheck(BaseModel):
    status: str = "OK"

#Responses
class UploadResponse(BaseModel):
    status: str

class DownloadResponse(BaseModel):
    status: str
    response: dict[str,bytes]