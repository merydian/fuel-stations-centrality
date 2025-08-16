import osmnx as ox


def get_fuel_stations(place):
    """
    Downloads fuel stations from OSM using osmnx for the given place.
    Returns a GeoDataFrame of fuel station locations.
    """

    ox.settings.overpass_settings = '[maxsize:20000000000]'

    tags = {"amenity": "fuel"}
    G = ox.graph_from_place(place, network_type="drive")
    fuel_gdf = ox.features_from_place(place, tags)
    return fuel_gdf


if __name__ == "__main__":
    place = "Iceland"
    fuel_stations = get_fuel_stations(place)
    print(f"Found {len(fuel_stations)} fuel stations in {place}")