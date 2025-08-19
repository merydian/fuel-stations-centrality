"""
Configuration file for fuel station centrality analysis.
"""

import os
from pathlib import Path


class Config:
    """Configuration class for analysis parameters."""
    
    # Analysis parameters
    PLACE = "Heidelberg, Germany"
    MAX_DISTANCE = 20000  # meters
    N_REMOVE = 5
    K_NN = 5
    REMOVAL_KIND = "knn_dist"
    
    # Data limits
    MAX_STATIONS_FOR_ANALYSIS = 100
    MIN_STATIONS_REQUIRED = 10
    MAX_NODES_FOR_BETWEENNESS = 10000
    MAX_NODES_FOR_STRAIGHTNESS = 20000
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Directories
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("output")
    CACHE_DIR = Path("cache")
    
    # File names
    LOG_FILE = "fuel_stations.log"
    
    # API configuration
    ORS_API_KEY = os.getenv("ORS_API_KEY")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.CACHE_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_road_filename(cls, place: str) -> str:
        """Generate standardized road network filename."""
        safe_name = place.lower().replace(", ", "_").replace(" ", "_")
        return f"{safe_name}_road.graphml"
    
    @classmethod
    def validate_config(cls):
        """Validate configuration parameters."""
        if not cls.ORS_API_KEY:
            raise ValueError("ORS_API_KEY environment variable is required")
        
        if cls.N_REMOVE <= 0:
            raise ValueError("N_REMOVE must be positive")
        
        if cls.K_NN <= 0:
            raise ValueError("K_NN must be positive")
        
        if cls.MAX_DISTANCE <= 0:
            raise ValueError("MAX_DISTANCE must be positive")
