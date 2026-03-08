from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class Plot(ABC):
    """Abstract base class for building data plotting."""
    
    @abstractmethod
    def plot_acf(self, acf24_per_building: Dict[int, float]) -> str:
        """Plot acf of buildings.
        
        Args:
            acf24_per_building: 24 h acf for each building.

        Returns:
            html string object
        """
        pass

    @abstractmethod
    def plot_energy(self, per_building: Dict[int, pd.DataFrame]) -> str:
        """Plots energy of buildings.
        
        Args:
            per_building: DF of buildings.

        Returns:
            Html string object
        """
        pass


class ModelPlot(ABC):
    """Abstract base class for model prediction plotting."""

    @staticmethod
    @abstractmethod
    def plot_model_pred(build_id: int, pred: pd.Series, test: pd.Series) -> str:
        """Plots predicted versus actual values per season.
        
        Args:
            build_id: ID of the building.
            pred: Series data of the building predicted data.
            test: Series data of the test data.

        Returns:
            Html string object
        """
        pass