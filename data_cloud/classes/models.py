from pydantic import BaseModel, field_validator

#Health check
class HealthCheck(BaseModel):
    status: str = "OK"

#Responses
class UploadResponse(BaseModel):
    status: str

class DownloadResponse(BaseModel):
    status: str
    response: dict[str,bytes]