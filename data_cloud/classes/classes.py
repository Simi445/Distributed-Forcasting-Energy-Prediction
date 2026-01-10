from abc import ABC, abstractmethod
from typing import List, Dict
from .models import FileNames

class Converter(ABC):
    """Abstract base class for data conversion components.
    
    Handles converting data files to Parquet format.
    """
    
    @abstractmethod
    def data_to_parquet(self, names: FileNames) -> List[str]:
        """Convert data files to Parquet format.
        
        Args:
            names: File names configuration for conversion.
            
        Returns:
            List of converted Parquet file names.
        """
        pass

class Storage(ABC):
    """Abstract base class for storage components.
    
    Manages uploading and reading Parquet files from storage (e.g., S3).
    """
    
    @abstractmethod
    def upload_parquet_file(self, file_name: str, object_name: str = None) -> bool:
        """Upload a Parquet file to storage.
        
        Args:
            file_name: Local path to the Parquet file.
            object_name: Optional name for the object in storage.
            
        Returns:
            True if upload successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def read_parquet_files(self) -> Dict[str, str]:
        """Read Parquet files from storage.
        
        Returns:
            Dictionary mapping file names to their content or paths.
        """
        pass