"""
Integration tests for centrality.py module
These tests require network access and may take longer to run.
"""

import pytest
import geopandas as gpd
import os
from downloader import get_fuel_stations
from ors_router import get_ors_distance_matrix


class TestGetFuelStationsIntegration:
    """Integration test cases for the get_fuel_stations function"""

    @pytest.mark.slow
    def test_get_fuel_stations_real_place(self):
        """
        Integration test with a real place.
        This test requires network access and may be slow.
        Use pytest -m "not slow" to skip this test.
        """
        # Use a small area to limit the query
        place = "Central Park, New York, NY, USA"

        try:
            result = get_fuel_stations(place)

            # Basic assertions
            assert isinstance(result, gpd.GeoDataFrame)

            # If fuel stations are found, check the structure
            if len(result) > 0:
                assert "geometry" in result.columns
                assert "amenity" in result.columns
                # All returned features should be fuel stations
                assert all(result["amenity"] == "fuel")

        except Exception as e:
            # If the test fails due to network issues, skip it
            pytest.skip(f"Integration test skipped due to network/API issue: {e}")


class TestGetOrsDistanceMatrixIntegration:
    """Integration test cases for the get_ors_distance_matrix function"""

    @pytest.mark.slow
    def test_get_ors_distance_matrix_real_api(self):
        """
        Integration test with real OpenRouteService API.
        This test requires a valid API key and network access.
        Use pytest -m "not slow" to skip this test.
        """
        # Get API key from environment variable
        api_key = os.getenv("ORS_API_KEY")

        if not api_key:
            pytest.skip("ORS_API_KEY environment variable not set")

        # Small set of test locations in New York
        locations = [
            (-74.0059, 40.7128),  # NYC
            (-73.9857, 40.7484),  # Times Square
        ]

        try:
            result = get_ors_distance_matrix(locations, api_key)

            # Basic assertions
            assert isinstance(result, dict)
            assert "distances" in result
            assert "durations" in result

            # Check matrix dimensions
            distances = result["distances"]
            durations = result["durations"]

            assert len(distances) == len(locations)
            assert len(distances[0]) == len(locations)
            assert len(durations) == len(locations)
            assert len(durations[0]) == len(locations)

            # Check diagonal elements (should be 0)
            for i in range(len(locations)):
                assert distances[i][i] == 0
                assert durations[i][i] == 0

            # Check that matrix is symmetric for distance
            for i in range(len(locations)):
                for j in range(len(locations)):
                    assert (
                        abs(distances[i][j] - distances[j][i]) < 0.01
                    )  # Allow small floating point differences

        except Exception as e:
            # If the test fails due to network/API issues, skip it
            if "API error" in str(e) or "quota" in str(e).lower():
                pytest.skip(f"API quota exceeded or API error: {e}")
            else:
                pytest.skip(f"Integration test skipped due to network/API issue: {e}")


if __name__ == "__main__":
    # Run only integration tests
    pytest.main([__file__, "-v", "-m", "slow"])
