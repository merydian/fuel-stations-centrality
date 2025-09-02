"""
Configuration file for fuel station centrality analysis.
"""

from pathlib import Path


class Config:
    """Configuration class for analysis parameters."""

    # Analysis parameters (defaults)
    PLACE = "Iran"
    MAX_DISTANCE = 100_000
    N_REMOVE = 50
    K_NN = 5
    REMOVAL_KIND = "knn_dist"
    STATIONS_MAX_RADIUS = 500  # meters

    # CRS configuration
    EPSG_CODE = 32642  # Target projected CRS

    # Data and output directories
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path(f"output/{PLACE.lower()}")
    CACHE_DIR = Path("cache")

    # File paths
    LOCAL_PBF_PATH = DATA_DIR / f"{PLACE.lower()}-latest.osm"
    SIMPLIFY_ROAD_NETWORK = True

    # Sampling and limits
    SAMPLE_NODES = None  # None = no sampling
    MIN_STATIONS_REQUIRED = 2
    MAX_STATIONS = None  # None = no limit

    CALCULATE_CENTRALITY = False

    # Random seed
    RANDOM_SEED = 42

    def __init__(self, args=None):
        """Initialize config from args (e.g., argparse.Namespace)."""
        if args:
            for key in vars(args):
                if hasattr(self, key.upper()):
                    value = getattr(args, key)
                    # Only set the attribute if the value is not None
                    if value is not None:
                        setattr(self, key.upper(), getattr(args, key))

        # Update dependent paths if PLACE changed
        self.OUTPUT_DIR = Path(f"output/{self.PLACE.lower()}")
        self.LOCAL_PBF_PATH = self.DATA_DIR / f"{self.PLACE.lower()}-latest.osm"

    @classmethod
    def get_target_crs(cls):
        """Get target projected CRS string."""
        return f"EPSG:{cls.EPSG_CODE}"

    @classmethod
    def ensure_target_crs(cls, gdf, name="data"):
        """Ensure GeoDataFrame is in target projected CRS."""
        import logging

        logger = logging.getLogger(__name__)
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
        cls.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        cls.CACHE_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_road_filename(cls) -> str:
        """Generate standardized road network filename."""
        safe_name = cls.PLACE.lower().replace(", ", "_").replace(" ", "_")
        return f"{safe_name}_road.graphml"

    @classmethod
    def get_road_filepath(cls) -> Path:
        """Get the file path for the road network."""
        return cls.DATA_DIR / cls.get_road_filename()

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
        if (
            cls.MAX_STATIONS is not None
            and cls.MAX_STATIONS < cls.MIN_STATIONS_REQUIRED
        ):
            raise ValueError(
                f"MAX_STATIONS ({cls.MAX_STATIONS}) must be >= MIN_STATIONS_REQUIRED ({cls.MIN_STATIONS_REQUIRED})"
            )
        if cls.MAX_STATIONS is not None and cls.N_REMOVE >= cls.MAX_STATIONS:
            raise ValueError(
                f"N_REMOVE ({cls.N_REMOVE}) must be < MAX_STATIONS ({cls.MAX_STATIONS})"
            )
        if cls.SAMPLE_NODES is not None and cls.SAMPLE_NODES <= 0:
            raise ValueError("SAMPLE_NODES must be positive or None")
