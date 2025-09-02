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

    CALCULATE_CENTRALITY = True

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

    def get_target_crs(self):
        """Get target projected CRS string."""
        return f"EPSG:{self.EPSG_CODE}"

    def ensure_target_crs(self, gdf, name="data"):
        """Ensure GeoDataFrame is in target projected CRS."""
        import logging

        logger = logging.getLogger(__name__)
        if str(gdf.crs) != self.get_target_crs():
            logger.info(f"Projecting {name} from {gdf.crs} to {self.get_target_crs()}")
            gdf = gdf.to_crs(self.get_target_crs())
        else:
            logger.debug(f"{name} already in target CRS {self.get_target_crs()}")
        return gdf

    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.CACHE_DIR.mkdir(exist_ok=True)

    def get_road_filename(self) -> str:
        """Generate standardized road network filename."""
        safe_name = self.PLACE.lower().replace(", ", "_").replace(" ", "_")
        return f"{safe_name}_road.graphml"

    def get_road_filepath(self) -> Path:
        """Get the file path for the road network."""
        return self.DATA_DIR / self.get_road_filename()

    def validate_config(self):
        """Validate configuration parameters."""
        if self.N_REMOVE <= 0:
            raise ValueError("N_REMOVE must be positive")
        if self.K_NN <= 0:
            raise ValueError("K_NN must be positive")
        if self.MAX_DISTANCE <= 0:
            raise ValueError("MAX_DISTANCE must be positive")
        if self.MAX_STATIONS is not None and self.MAX_STATIONS <= 0:
            raise ValueError("MAX_STATIONS must be positive or None")
        if (
            self.MAX_STATIONS is not None
            and self.MAX_STATIONS < self.MIN_STATIONS_REQUIRED
        ):
            raise ValueError(
                f"MAX_STATIONS ({self.MAX_STATIONS}) must be >= MIN_STATIONS_REQUIRED ({self.MIN_STATIONS_REQUIRED})"
            )
        if self.MAX_STATIONS is not None and self.N_REMOVE >= self.MAX_STATIONS:
            raise ValueError(
                f"N_REMOVE ({self.N_REMOVE}) must be < MAX_STATIONS ({self.MAX_STATIONS})"
            )
        if self.SAMPLE_NODES is not None and self.SAMPLE_NODES <= 0:
            raise ValueError("SAMPLE_NODES must be positive or None")
