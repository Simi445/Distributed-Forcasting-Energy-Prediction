from sklearn.model_selection import GridSearchCV
from pydantic import BaseModel
from typing import Callable, List, Tuple, Dict, Optional,Set,Any
import pandas as pd
from base_classes import BaselineModel, SeasonalityStrategy

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
    
class RFConfig(BaseModel):
    k_fold: int = 3
    n_jobs: int = 4
    state_seed: int = 42

class BaseLineModelConfig(BaseModel):
    models:Dict[str,BaselineModel]
    
class AnalysisConfig(BaseModel):
    build_id: int
    fold_key: str
    dataset_b: Dict
    grid_rf: GridSearchCV
    trainer_fn: Callable
    strategy: SeasonalityStrategy
    iterations: int = 10
    state_seed: int = 42