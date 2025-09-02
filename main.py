import argparse
import logging
import sys
from analysis import analysis
from config import Config
from graph_builder import build_graph
import urllib.request

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with cmd args")
    parser.add_argument("--place", type=str, help="Place name for the analysis")
    parser.add_argument("--max-distance", type=float, help="Maximum distance for pruning (in meters)")
    parser.add_argument("--n-remove", type=int, help="Number of stations to remove")
    parser.add_argument("--epsg-code", type=str, help="EPSG code for the coordinate reference system")
    parser.add_argument("--max-stations", type=int, help="Number of stations to use (None = all)")

    args = parser.parse_args()

    Config = Config(args)

    Config.ensure_directories()
    Config.validate_config()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Print to console
            logging.FileHandler(Config.OUTPUT_DIR / f'fuel_stations_analysis_{Config.PLACE.lower()}.log')  # Also save to file
        ]
    )

    build_graph(Config)

    analysis(Config)