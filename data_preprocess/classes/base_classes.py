from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Union, Any
import pandas as pd

class Loader(ABC):
    """Abstract base class for data loading components."""
    
    @abstractmethod
    def get_dataset(self, data_category: str) -> Optional[pd.DataFrame]:
        """Retrieve a dataset by category.
        
        Args:
            data_category: The type of dataset to retrieve (e.g., 'train', 'meta').
            
        Returns:
            The requested DataFrame or None if not found.
        """
        pass 

class Merger(ABC):
    """Abstract base class for data merging components."""
    
    @abstractmethod
    def merge_tables(self) -> None:
        """Merge multiple data tables into a single dataset."""
        pass 
    
    @abstractmethod
    def get_full_df(self) -> pd.DataFrame:
        """Get the merged full DataFrame.
        
        Returns:
            The complete merged DataFrame.
        """
        pass 

class Engineer(ABC):
    """Abstract base class for feature engineering components."""
    
    @abstractmethod
    def feature_engineer_buildings(self, full_df: pd.DataFrame, config: 'FeatureEngineerConfig') -> Dict[int, pd.DataFrame]:
        """Engineer features for each building.
        
        Args:
            full_df: The full merged DataFrame.
            config: Configuration for feature engineering.
            
        Returns:
            Dictionary of building IDs to their feature-engineered DataFrames.
        """
        pass

class Selection(ABC):
    """Abstract base class for data selection components."""
    
    @abstractmethod
    def category_buildings(self, per_building: Dict[int, pd.DataFrame]) -> Dict[int, str]:
        """Categorize buildings by type.
        
        Args:
            per_building: Dictionary of building DataFrames.
            
        Returns:
            Dictionary mapping building IDs and their categories.
        """
        pass
    
    @abstractmethod
    def calculate_lag_acf(self, building_type: Dict[int, str], 
                          per_building: Dict[int, pd.DataFrame],
                          acf_config: Optional['ACFConfig'] = None) -> Tuple[Dict[str, Any], Dict[int, float]]:
        """Calculate lag autocorrelation for building categories.
        
        Args:
            building_type: Building type mappings (building_id -> building_type).
            per_building: Building DataFrames.
            acf_config: Optional ACF configuration. If None, uses defaults.
            
        Returns:
            Tuple[Dict[str, Any], Dict[int, float]]: A tuple containing:
            - acf24_per_type: Autocorrelation at lag 24 aggregated per building type.
            - acf24_per_building: Autocorrelation at lag 24 for each individual building.
        """
        pass
    
    @abstractmethod
    def median_per_type(self, acf24_per_type: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Tuple[int, float]]:
        """Calculate median ACF per building type.
        
        Args:
            acf24_per_type: ACF data per type.
            
        Returns:
            Median building and ACF per type.
        """
        pass

class Filter(ABC):
    """Abstract base class for data filtering components."""
    
    @abstractmethod
    def quality_check(self, df_b: pd.DataFrame, freq: str = 'h') -> Dict[str, Union[float, pd.Timedelta]]:
        """Check data quality for a building.
        
        Args:
            df_b: Building DataFrame.
            freq: Frequency for quality check.
            
        Returns:
            Quality metrics.
        """
        pass
    
    @abstractmethod
    def filter_buildings(self, build_dict: Dict[str, int], df_b: pd.DataFrame, config: 'FilterConfig') -> List[Tuple[Tuple[float, pd.Timedelta, float], str, int]]:
        """Filter buildings based on quality.
        
        Args:
            build_dict: Building dictionary.
            df_b: Building DataFrame.
            config: Filtering configuration.
            
        Returns:
            Filtered building list.
        """
        pass

class ModelBuildings(ABC):
    """Abstract base class for model building selection components."""
    
    @abstractmethod
    def select_final_buildings(self, per_building: Dict[int, pd.DataFrame], median_buildings: Dict[str, Tuple[int, float]], k: int = 3) -> Dict[str, int]:
        """Select final buildings for modeling.
        
        Args:
            per_building: Dictionary of building DataFrames.
            median_buildings: Median buildings per category with ACF values.
            k: Number of buildings to select per category.
            
        Returns:
            Selected final buildings.
        """
        pass


class State(ABC):
    """Abstract base class for state management."""
    
    @abstractmethod
    def add_state(self, name:str, value:Any) -> None:
        """Add state to the manager.
        
        Args:
            name: Name of the state.
            value: The data the state holds.

        Returns:
            Nothing
        """
        pass
    
    @abstractmethod
    def get_state(self, name: str) -> Any:
        """Fetches state data if it exists.
        
        Args:
            name: Name of the state.

        Returns:
            The value that the state holds.
        """
        pass

class Plot(ABC):
    """Abstract base class for plotting."""
    
    @abstractmethod
    def plot_acf(self, acf24_per_building:Dict[int, float])->str:
        """Plot acf of buildings.
        
        Args:
            acf24_per_building: 24 h acf for each building.

        Returns:
            html string object
        """
        pass

    @abstractmethod
    def plot_energy(self,per_building:Dict[int, pd.DataFrame])->str:
        """Plots energy of buildings.
        
        Args:
            per_building: DF of buildings.

        Returns:
            Html string object
        """
        pass