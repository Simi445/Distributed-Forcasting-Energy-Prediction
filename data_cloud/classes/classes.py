from abc import ABC, abstractmethod
from typing import Optional

class BulkConverter(ABC):
    """Abstract base class for data conversion components.
    
    Handles converting data files to Parquet format.
    """
    
    @abstractmethod
    def data_to_data(self, location: str, extension: str, **kwargs) -> list[str]:
        """Convert data files to other data extentions.
        
        Args:
            location: The folder where the files lie.
            extension: The new extansion of the files.
        Returns:
            List of converted files.
        """
        pass

class Storage(ABC):
    """Abstract base class for storage components.
    
    Manages uploading and reading Parquet files from storage (e.g., S3).
    """
    
    @abstractmethod
    def upload_file_bucket(self, file_name: str, object_name: str = None) -> bool:
        """Upload a file to storage.
        
        Args:
            file_name: Local path to the Parquet file.
            object_name: Optional name for the object in storage.
            
        Returns:
            True if upload successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def read_file_bucket(self, file_name: str) -> Optional[str]:
        """Read a file from storage.
        
        Args:
            file_name: Name of the file to read from storage.
            
        Returns:
            Base64-encoded file content if successful, None otherwise.
        """
        pass