"""
Configuration file for fuel station centrality analysis.
"""

import os
from pathlib import Path


class Config:
    """Configuration class for analysis parameters."""

    # Analysis parameters
    PLACE = "Cook"
    MAX_DISTANCE = 2  # meters
    N_REMOVE = 2
    K_NN = 8
    REMOVAL_KIND = "knn_dist"

    LOCAL_PBF_PATH = Path(__file__).parent / "data" / "cook-latest.osm"
    SIMPLIFY_ROAD_NETWORK = True

    # Station clustering parameters
    CLUSTER_RADIUS = 50_000  # meters - stations within this radius are combined

    # Graph sampling parameter
    SAMPLE_NODES = None  # Number of nodes to sample from road network (None = no sampling)

    # Data limits
    MIN_STATIONS_REQUIRED = 2
    MAX_STATIONS = 300  # Maximum number of stations to use (None = no limit)

    # Random seed for reproducibility
    RANDOM_SEED = 42

    # Directories
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("output")
    CACHE_DIR = Path("cache")

    # File names
    LOG_FILE = "fuel_stations.log"
    STATS_FILE = OUTPUT_DIR / "stats.json"  # Path to save all stats

    # Coordinate Reference System (CRS) configuration
    WGS84_EPSG = 4326  # Input CRS from OSM data
    EPSG_CODE = 32642  # Target projected CRS for analysis (UTM Zone 48N for Mongolia)

    # Logging configuration
    LOG_LEVEL = "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Centrality calculation control
    CALCULATE_DEGREE_CENTRALITY = True
    CALCULATE_CLOSENESS_CENTRALITY = False
    CALCULATE_BETWEENNESS_CENTRALITY = False
    CALCULATE_EIGENVECTOR_CENTRALITY = False
    CALCULATE_STRAIGHTNESS_CENTRALITY = True
    CALCULATE_GLOBAL_STRAIGHTNESS = True

    @classmethod
    def get_wgs84_crs(cls):
        """Get WGS84 CRS string."""
        return f"EPSG:{cls.WGS84_EPSG}"

    @classmethod
    def get_target_crs(cls):
        """Get target projected CRS string."""
        return f"EPSG:{cls.EPSG_CODE}"

    @classmethod
    def ensure_target_crs(cls, gdf, name="data"):
        """
        Ensure GeoDataFrame is in target projected CRS.

        Args:
            gdf: GeoDataFrame to check/transform
            name: Name for logging purposes

        Returns:
            GeoDataFrame in target CRS
        """
        import logging

        logger = logging.getLogger(__name__)

        if gdf.crs is None:
            logger.info(f"Setting {name} CRS to WGS84 (assumed)")
            gdf.crs = cls.get_wgs84_crs()

        if str(gdf.crs) != cls.get_target_crs():
            logger.info(f"Projecting {name} from {gdf.crs} to {cls.get_target_crs()}")
            gdf = gdf.to_crs(cls.get_target_crs())
        else:
            logger.debug(f"{name} already in target CRS {cls.get_target_crs()}")

        return gdf

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.CACHE_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_road_filename(cls) -> str:
        """Generate standardized road network filename."""
        safe_name = cls.PLACE.lower().replace(", ", "_").replace(" ", "_")
        return Path(f"{safe_name}_road.graphml")

    @classmethod
    def validate_config(cls):
        """Validate configuration parameters."""

        if cls.N_REMOVE <= 0:
            raise ValueError("N_REMOVE must be positive")

        if cls.K_NN <= 0:
            raise ValueError("K_NN must be positive")

        if cls.MAX_DISTANCE <= 0:
            raise ValueError("MAX_DISTANCE must be positive")
            
        if cls.MAX_STATIONS is not None and cls.MAX_STATIONS <= 0:
            raise ValueError("MAX_STATIONS must be positive or None")
            
        if cls.MAX_STATIONS is not None and cls.MAX_STATIONS < cls.MIN_STATIONS_REQUIRED:
            raise ValueError(f"MAX_STATIONS ({cls.MAX_STATIONS}) must be >= MIN_STATIONS_REQUIRED ({cls.MIN_STATIONS_REQUIRED})")
            
        if cls.MAX_STATIONS is not None and cls.N_REMOVE >= cls.MAX_STATIONS:
            raise ValueError(f"N_REMOVE ({cls.N_REMOVE}) must be < MAX_STATIONS ({cls.MAX_STATIONS})")

        if cls.SAMPLE_NODES is not None and cls.SAMPLE_NODES <= 0:
            raise ValueError("SAMPLE_NODES must be positive or None")

    @classmethod
    def get_road_filepath(cls) -> Path:
        """Get the file path for the road network."""
        road_filename = cls.get_road_filename()
        return cls.DATA_DIR / road_filename
