from abc import ABC, abstractmethod
from typing import Dict, Optional
from classes import (
    SeasonalityStrategy, RFGridParamsConfig, XGGridParamsConfig, ModelConfigparams,ModelFactory
)
from ..random_forest import RandomForestImpl
from ..svr import SVRModel
from ..xg_boost import XGBoost

class RandomForestFactory(ModelFactory):
    def __init__(self, param_grid: Optional[RFGridParamsConfig] = None, 
                 config: Optional[ModelConfigparams] = None):
        self.param_grid = param_grid or RFGridParamsConfig()
        self.config = config or ModelConfigparams()
    
    def create(self, dataset_b: Dict, strategy: SeasonalityStrategy) -> RandomForestImpl:
        return RandomForestImpl(self.param_grid, self.config, dataset_b, strategy)

class SVRFactory(ModelFactory):
    def __init__(self, config: Optional[ModelConfigparams] = None):
        self.config = config or ModelConfigparams()
    
    def create(self, dataset_b: Dict, strategy: SeasonalityStrategy) -> SVRModel:
        return SVRModel(self.config, dataset_b, strategy)

class XGBoostFactory(ModelFactory):
    def __init__(self, param_grid: Optional[XGGridParamsConfig] = None,
                 config: Optional[ModelConfigparams] = None):
        self.param_grid = param_grid or XGGridParamsConfig()
        self.config = config or ModelConfigparams()
    
    def create(self, dataset_b: Dict, strategy: SeasonalityStrategy) -> XGBoost:
        return XGBoost(self.param_grid, self.config, dataset_b, strategy)