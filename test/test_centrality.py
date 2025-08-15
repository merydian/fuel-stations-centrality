"""
Tests for centrality.py module
"""

import pytest
import pandas as pd
import geopandas as gpd
from unittest.mock import patch, MagicMock
from shapely.geometry import Point
import networkx as nx
import osmnx as ox
import openrouteservice as ors

from centrality import farness_centrality
from ors_router import get_ors_distance_matrix
from downloader import get_fuel_stations


class TestGetFuelStations:
    """Test cases for the get_fuel_stations function"""

    @patch("centrality.ox.features_from_place")
    @patch("centrality.ox.graph_from_place")
    def test_get_fuel_stations_success(
        self, mock_graph_from_place, mock_features_from_place
    ):
        """Test successful fuel station retrieval"""
        # Mock the graph creation
        mock_graph = MagicMock()
        mock_graph_from_place.return_value = mock_graph

        # Create mock fuel station data
        mock_fuel_data = gpd.GeoDataFrame(
            {
                "amenity": ["fuel", "fuel", "fuel"],
                "name": ["Station A", "Station B", "Station C"],
                "geometry": [
                    Point(-74.0059, 40.7128),  # NYC coordinates
                    Point(-74.0159, 40.7228),
                    Point(-74.0259, 40.7328),
                ],
            }
        )
        mock_features_from_place.return_value = mock_fuel_data

        # Test the function
        place = "New York, NY, USA"
        result = get_fuel_stations(place)

        # Verify the mocks were called with correct parameters
        mock_graph_from_place.assert_called_once_with(place, network_type="drive")
        mock_features_from_place.assert_called_once_with(place, {"amenity": "fuel"})

        # Verify the result
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 3
        assert all(result["amenity"] == "fuel")

    @patch("centrality.ox.features_from_place")
    @patch("centrality.ox.graph_from_place")
    def test_get_fuel_stations_empty_result(
        self, mock_graph_from_place, mock_features_from_place
    ):
        """Test when no fuel stations are found"""
        # Mock the graph creation
        mock_graph = MagicMock()
        mock_graph_from_place.return_value = mock_graph

        # Create empty mock data
        mock_empty_data = gpd.GeoDataFrame(columns=["amenity", "geometry"])
        mock_features_from_place.return_value = mock_empty_data

        # Test the function
        place = "Remote Area"
        result = get_fuel_stations(place)

        # Verify the result is empty but still a GeoDataFrame
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    @patch("centrality.ox.features_from_place")
    @patch("centrality.ox.graph_from_place")
    def test_get_fuel_stations_osm_error(
        self, mock_graph_from_place, mock_features_from_place
    ):
        """Test handling of OSM API errors"""
        # Mock an exception during graph creation
        mock_graph_from_place.side_effect = Exception("OSM API Error")

        place = "Invalid Place"

        # Test that the exception is propagated
        with pytest.raises(Exception, match="OSM API Error"):
            get_fuel_stations(place)

    @patch("centrality.ox.features_from_place")
    @patch("centrality.ox.graph_from_place")
    def test_get_fuel_stations_features_error(
        self, mock_graph_from_place, mock_features_from_place
    ):
        """Test handling of errors during features retrieval"""
        # Mock successful graph creation
        mock_graph = MagicMock()
        mock_graph_from_place.return_value = mock_graph

        # Mock an exception during features retrieval
        mock_features_from_place.side_effect = Exception("Features API Error")

        place = "Test Place"

        # Test that the exception is propagated
        with pytest.raises(Exception, match="Features API Error"):
            get_fuel_stations(place)

    def test_get_fuel_stations_correct_tags(self):
        """Test that the function uses correct OSM tags"""
        with (
            patch("centrality.ox.graph_from_place") as mock_graph,
            patch("centrality.ox.features_from_place") as mock_features,
        ):
            mock_graph.return_value = MagicMock()
            mock_features.return_value = gpd.GeoDataFrame()

            place = "Test City"
            get_fuel_stations(place)

            # Verify that the correct tags are used
            expected_tags = {"amenity": "fuel"}
            mock_features.assert_called_once_with(place, expected_tags)


class TestFarnessCentrality:
    """Test cases for the farness_centrality function"""

    def test_farness_centrality_simple_graph(self):
        """Test farness centrality on a simple graph"""
        # Create a simple graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])

        result = farness_centrality(G)

        # Check that all nodes are in the result
        assert set(result.keys()) == {1, 2, 3, 4}

        # Check that farness values are positive numbers
        for node, farness in result.items():
            assert isinstance(farness, (int, float))
            assert farness >= 0

        # In this linear graph, middle nodes should have lower farness (better centrality)
        assert result[2] < result[1]
        assert result[3] < result[4]

    def test_farness_centrality_single_node(self):
        """Test farness centrality on a single node"""
        G = nx.Graph()
        G.add_node(1)

        result = farness_centrality(G)

        assert len(result) == 1
        assert result[1] == 0  # Single node has no distances to sum

    def test_farness_centrality_weighted_graph(self):
        """Test farness centrality with weighted edges"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 1.0), (2, 3, 2.0), (3, 4, 1.0)])

        result_unweighted = farness_centrality(G)
        result_weighted = farness_centrality(G, weight="weight")

        # Results should be different when weights are considered
        assert result_unweighted != result_weighted

        # Both should contain all nodes
        assert set(result_unweighted.keys()) == {1, 2, 3, 4}
        assert set(result_weighted.keys()) == {1, 2, 3, 4}

    def test_farness_centrality_disconnected_graph(self):
        """Test farness centrality on a disconnected graph"""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (3, 4)])  # Two disconnected components

        result = farness_centrality(G)

        # Should still work, but unreachable nodes won't contribute to the sum
        assert len(result) == 4
        for farness in result.values():
            assert isinstance(farness, (int, float))
            assert farness >= 0


class TestGetOrsDistanceMatrix:
    """Test cases for the get_ors_distance_matrix function"""

    def test_get_ors_distance_matrix_empty_locations(self):
        """Test error handling for empty locations list"""
        with pytest.raises(ValueError, match="Locations list cannot be empty"):
            get_ors_distance_matrix([], "test-api-key")

    def test_get_ors_distance_matrix_single_location(self):
        """Test error handling for single location"""
        locations = [(-74.0059, 40.7128)]
        with pytest.raises(ValueError, match="At least 2 locations are required"):
            get_ors_distance_matrix(locations, "test-api-key")

    def test_get_ors_distance_matrix_invalid_coordinates(self):
        """Test error handling for invalid coordinates"""
        # Invalid longitude
        locations = [(-200, 40.7128), (-74.0059, 40.7128)]
        with pytest.raises(ValueError, match="Invalid coordinates at index 0"):
            get_ors_distance_matrix(locations, "test-api-key")

        # Invalid latitude
        locations = [(-74.0059, 100), (-74.0059, 40.7128)]
        with pytest.raises(ValueError, match="Invalid coordinates at index 0"):
            get_ors_distance_matrix(locations, "test-api-key")

    @patch("ors_router.ors.Client")
    @patch("ors_router.ors.distance_matrix.distance_matrix")
    def test_get_ors_distance_matrix_success(self, mock_distance_matrix, mock_client):
        """Test successful distance matrix retrieval"""
        # Mock the ORS client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock successful response
        mock_response = {
            "distances": [[0, 1.5], [1.5, 0]],
            "durations": [[0, 120], [120, 0]],
            "sources": [
                {"location": [-74.0059, 40.7128]},
                {"location": [-73.9857, 40.7484]},
            ],
            "destinations": [
                {"location": [-74.0059, 40.7128]},
                {"location": [-73.9857, 40.7484]},
            ],
        }
        mock_distance_matrix.return_value = mock_response

        # Test the function
        locations = [(-74.0059, 40.7128), (-73.9857, 40.7484)]
        api_key = "test-api-key"
        result = get_ors_distance_matrix(locations, api_key)

        # Verify the client was created with the correct API key
        mock_client.assert_called_once_with(key=api_key)

        # Verify the distance matrix function was called with correct parameters
        mock_distance_matrix.assert_called_once_with(
            client=mock_client_instance,
            locations=[[-74.0059, 40.7128], [-73.9857, 40.7484]],
            profile="driving-car",
            metrics=["distance", "duration"],
            units="km",
        )

        # Verify the result
        assert result == mock_response
        assert "distances" in result
        assert "durations" in result

    @patch("ors_router.ors.Client")
    @patch("ors_router.ors.distance_matrix.distance_matrix")
    def test_get_ors_distance_matrix_custom_parameters(
        self, mock_distance_matrix, mock_client
    ):
        """Test distance matrix with custom parameters"""
        # Mock the ORS client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock successful response
        mock_response = {"distances": [[0, 2.3], [2.3, 0]]}
        mock_distance_matrix.return_value = mock_response

        # Test with custom parameters
        locations = [(-74.0059, 40.7128), (-73.9857, 40.7484)]
        api_key = "test-api-key"
        result = get_ors_distance_matrix(
            locations, api_key, profile="foot-walking", metrics=["distance"], units="m"
        )

        # Verify the distance matrix function was called with custom parameters
        mock_distance_matrix.assert_called_once_with(
            client=mock_client_instance,
            locations=[[-74.0059, 40.7128], [-73.9857, 40.7484]],
            profile="foot-walking",
            metrics=["distance"],
            units="m",
        )

        assert result == mock_response

    @patch("ors_router.ors.Client")
    @patch("ors_router.ors.distance_matrix.distance_matrix")
    def test_get_ors_distance_matrix_api_error(self, mock_distance_matrix, mock_client):
        """Test handling of ORS API errors"""
        # Mock the ORS client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock API error
        mock_distance_matrix.side_effect = ors.exceptions.ApiError("API quota exceeded")

        locations = [(-74.0059, 40.7128), (-73.9857, 40.7484)]
        api_key = "test-api-key"

        with pytest.raises(
            Exception, match="OpenRouteService API error: API quota exceeded"
        ):
            get_ors_distance_matrix(locations, api_key)

    @patch("ors_router.ors.Client")
    @patch("ors_router.ors.distance_matrix.distance_matrix")
    def test_get_ors_distance_matrix_timeout_error(
        self, mock_distance_matrix, mock_client
    ):
        """Test handling of ORS timeout errors"""
        # Mock the ORS client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock timeout error
        mock_distance_matrix.side_effect = ors.exceptions.Timeout("Request timed out")

        locations = [(-74.0059, 40.7128), (-73.9857, 40.7484)]
        api_key = "test-api-key"

        with pytest.raises(
            Exception, match="OpenRouteService timeout error: Request timed out"
        ):
            get_ors_distance_matrix(locations, api_key)

    @patch("ors_router.ors.Client")
    @patch("ors_router.ors.distance_matrix.distance_matrix")
    def test_get_ors_distance_matrix_general_error(
        self, mock_distance_matrix, mock_client
    ):
        """Test handling of general errors"""
        # Mock the ORS client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock general error
        mock_distance_matrix.side_effect = Exception("Network error")

        locations = [(-74.0059, 40.7128), (-73.9857, 40.7484)]
        api_key = "test-api-key"

        with pytest.raises(
            Exception, match="Error getting distance matrix: Network error"
        ):
            get_ors_distance_matrix(locations, api_key)

    @patch("ors_router.ors.Client")
    def test_get_ors_distance_matrix_client_creation_error(self, mock_client):
        """Test handling of client creation errors"""
        # Mock client creation error
        mock_client.side_effect = Exception("Invalid API key")

        locations = [(-74.0059, 40.7128), (-73.9857, 40.7484)]
        api_key = "invalid-key"

        with pytest.raises(
            Exception, match="Error getting distance matrix: Invalid API key"
        ):
            get_ors_distance_matrix(locations, api_key)


if __name__ == "__main__":
    pytest.main([__file__])
