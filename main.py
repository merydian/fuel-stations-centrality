import argparse
import logging
import sys
from analysis import analysis
from config import Config
from graph_builder import build_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with cmd args")
    parser.add_argument("--place", type=str, help="Place name for the analysis")
    parser.add_argument(
        "--max-distance", type=float, help="Maximum distance for pruning (in meters)"
    )
    parser.add_argument("--n-remove", type=int, help="Number of stations to remove")
    parser.add_argument(
        "--epsg-code", type=int, help="EPSG code for the coordinate reference system"
    )
    parser.add_argument(
        "--max-stations", type=int, help="Number of stations to use (None = all)"
    )

    args = parser.parse_args()

    config = Config(args)

    print(f"Configuration: {config.__dict__}")

    config.ensure_directories()
    config.validate_config()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),  # Print to console
            logging.FileHandler(
                config.OUTPUT_DIR / f"fuel_stations_analysis_{config.PLACE.lower()}.log"
            ),
        ],
    )

    if not Config.get_road_filepath().exists():
        logging.info(f"Graph file not found at {config.LOCAL_PBF_PATH}, building graph...")
        build_graph(config)
        logging.info("Graph building completed")
    else:
        logging.info(f"Graph file already exists at {config.LOCAL_PBF_PATH}, skipping graph building")

    logging.info("Starting analysis...")
    analysis(config)
    logging.info("Analysis completed")