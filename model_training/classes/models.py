from sklearn.model_selection import GridSearchCV
from pydantic import BaseModel
from typing import Callable, List, Tuple, Dict, Optional,Set,Any
import pandas as pd


#Health check
class HealthCheck(BaseModel):
    status: str

    
#Responses
class ModelResponse(BaseModel):
    status: str
    response: Dict

#DTO
class GlobalInfo(BaseModel):
    metrics:Dict[str,float]

class GlobalInfoRF(GlobalInfo):
    p_importance:Dict
    f_importance:Dict

#Configs
class SeasonConfig(BaseModel):
    possible_seasons: List[str] = ["winter", "autumn"]
    seasonMap: Dict[str, List[int]] = {"autumn":[9,10,11], "winter":[1,2,12]}

class RFGridParamsConfig(BaseModel):
    parameters: Dict[str, List[Any]] = {
        'min_samples_leaf': [10,15],
        'n_estimators': [100,200],
        'max_depth': [None]
        }
    
class XGGridParamsConfig(BaseModel):
    parameters: Dict[str, List[Any]] = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    
class ModelConfigparams(BaseModel):
    k_fold: int = 3
    n_jobs: int = 4
    state_seed: int = 42

#Model Contexts
class DataContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    build_id: Optional[int] = None
    fold_key: Optional[str] = None
    dataset_b: Optional[Dict] = None


class TrainingContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    grid: Optional[Any] = None
    strategy: Optional[Any] = None


class FeatureAnalysisContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    iterations: Optional[int] = None
    state_seed: Optional[int] = None
    trainer_fn: Optional[Callable] = None



class BaseLineModelConfig(BaseModel):
    models:Dict[str,Any]
    

