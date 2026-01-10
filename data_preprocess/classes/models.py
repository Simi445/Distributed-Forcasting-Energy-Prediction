from pydantic import BaseModel, ConfigDict
from typing import List, Tuple, Dict, Optional
import pandas as pd

class BaseConfig(BaseModel):
    pass

#Health check
class HealthCheck(BaseModel):
    status: str

#Responses
class LoadResponse(BaseModel):
    status: str

class MergeResponse(BaseModel):
    status: str
    shape: tuple

class FeatureEngineerResponse(BaseModel):
    status: str
    buildings: List[int]

class ACFResponse(BaseModel):
    status: str
    graph_data: Dict[str, Tuple[List[str], List[float]]]

class MedianResponse(BaseModel):
    status: str
    response: Dict[str, int]

class PlotResponse(BaseModel):
    status: str
    response: Dict[str, str]

#Configs
class FeatureEngineerConfig(BaseModel):
    weather_cols: Optional[List[str]] = None
    lag_configs: Optional[Dict[str, int]] = None
    freq: str = 'h'
    interp_limit: int = 6

class FilterConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_gap_th: pd.Timedelta = pd.Timedelta(days=7)
    miss_ratio_th: float = 0.1
    days_th_min: int = 310
    k: int = 3

class ACFConfig(BaseModel):
    hours: int = 24
    nr_days: int = 2

