import osmnx as ox


def get_fuel_stations(place):
    """
    Downloads fuel stations from OSM using osmnx for the given place.
    Returns a GeoDataFrame of fuel station locations.
    """
    tags = {"amenity": "fuel"}
    G = ox.graph_from_place(place, network_type="drive")
    fuel_gdf = ox.features_from_place(place, tags)
    return fuel_gdf
