
from typing import Dict, Tuple
from classes import Model


class ModelManager:
    def __init__(self):
        self.models: Dict[Tuple[str,str], Model] = {}
    
    def add_model(self, key: Tuple[str,str], model:Model) -> None:
        if key in self.models:
            return
        self.models[key] = model
    
    def get_model(self,key: Tuple[str,str]) -> Model:
        if key not in self.models:
            raise ValueError("Model was not added to manager!")
        return self.models[key]