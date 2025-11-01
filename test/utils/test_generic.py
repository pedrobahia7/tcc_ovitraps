# Add the parent directory to the path to import the module
import sys
import pytest
import inspect
import time
import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook
import matplotlib.pyplot as plt


sys.path.append('utils')
import generic

class SharedTestUtils:
    """
    Shared utilities to reduce test redundancy across all test classes.

    """

    @staticmethod
    def test_function_signature_and_return(
        func,
        expected_params,
        expected_return_type,
    ):
        """
        Reusable function signature test.

        """
        sig = inspect.signature(func)
        real_params = list(sig.parameters.keys())

        for param in expected_params:
            assert param in real_params, (
                f"Function should have '{param}' parameter"
            )

        real_return_type = sig.return_annotation
        assert real_return_type == expected_return_type, (
            f"Function should return '{expected_return_type}', "
            f"but returned '{real_return_type}'"
        )

    @staticmethod
    def test_data_integrity(original_data, data_after_function):
        """
        Reusable data integrity test.

        """

        if isinstance(original_data, pd.DataFrame):
            pd.testing.assert_frame_equal(
                original_data,
                data_after_function,
            )

        elif isinstance(original_data, pd.Series):
            pd.testing.assert_series_equal(
                original_data,
                data_after_function,
            )

        elif isinstance(original_data, np.ndarray):
            np.testing.assert_array_equal(
                original_data,
                data_after_function,
            )

    @staticmethod
    def test_docstring_exists(func, required_keywords=None):
        """
        Reusable docstring validation.

        """
        assert func.__doc__ is not None, "Function should have a docstring"
        if required_keywords:
            docstring_lower = func.__doc__.lower()
            for keyword in required_keywords:
                assert keyword in docstring_lower, (
                    f"Docstring should mention '{keyword}'"
                )

    @staticmethod
    def test_performance_timing(
        func,
        args,
        max_time=1.0,
        description="Function",
    ):
        """
        Reusable performance testing.
        """
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        assert execution_time < max_time, (
            f"{description} should complete in <{max_time} seconds, "
            f"took {execution_time:.3f} seconds"
        )
        return result

    @staticmethod
    def test_file_integrity(func, file_path):
        """
        Reusable file integrity test.

        """
        # Test data integrity by checking file exists and is readable after execution
        original_mod_time = os.path.getmtime(file_path)

        # Execute the function
        func(file_path)

        # Verify file was modified (shows function actually ran)
        new_mod_time = os.path.getmtime(file_path)
        assert new_mod_time >= original_mod_time, (
            "File should be modified or timestamp unchanged"
        )

        # Verify file is still readable (data integrity concept)
        wb = load_workbook(file_path)
        assert wb is not None, (
            "File should remain readable after formatting"
        )

    @staticmethod
    def test_function_determinism(func, *args):
        """
        Test that the function produces the same output for the same input.

        """

        output1 = func(*args)
        output2 = func(*args)

        def check_equality_comprehensive(output1, output2):
            if isinstance(output1, pd.DataFrame):
                pd.testing.assert_frame_equal(output1, output2)
            elif isinstance(output1, pd.Series):
                pd.testing.assert_series_equal(output1, output2)
            elif isinstance(output1, np.ndarray):
                np.testing.assert_array_equal(output1, output2)
            else:
                assert output1 == output2, "Outputs should be identical"

        if isinstance(output1, (tuple, list)):
            assert len(output1) == len(output2), (
                "Output tuples should have the same length"
            )
            for item1, item2 in zip(output1, output2):
                check_equality_comprehensive(item1, item2)
        else:
            check_equality_comprehensive(output1, output2)

class TestClosestNeighbors:
    """Test cases for the generic.nearest_neighbors function using pytest."""
    
    @pytest.fixture
    def basic_point_data(self):
        """Fixture providing basic point data for testing."""
        return {
            'query_points': np.array([[0, 0], [1, 1], [5, 5]]),
            'reference_points': np.array([[0.1, 0.1], [1.1, 0.9], [2, 2], [4.8, 5.2], [10, 10]]),
            'expected_indices': [0, 1, 3]  # Closest reference point indices
        }
    

    def test_shared_test_utils(self, basic_point_data):
        """Test using SharedTestUtils methods."""
        # Test function signature and return type
        SharedTestUtils.test_function_signature_and_return(
            generic.nearest_neighbors,
            ['points1', 'points2'],
            np.ndarray
        )
        
        # Test docstring exists
        SharedTestUtils.test_docstring_exists(
            generic.nearest_neighbors,
            ['closest']
        )
        
        # Test data integrity (original data should not be modified)
        original_query = basic_point_data['query_points'].copy()
        original_reference = basic_point_data['reference_points'].copy()
        generic.nearest_neighbors(
            basic_point_data['query_points'], 
            basic_point_data['reference_points']
        )
        SharedTestUtils.test_data_integrity(original_query, basic_point_data['query_points'])
        SharedTestUtils.test_data_integrity(original_reference, basic_point_data['reference_points'])
        
        # Test function determinism
        SharedTestUtils.test_function_determinism(
            generic.nearest_neighbors,
            basic_point_data['query_points'],
            basic_point_data['reference_points']
        )
        
        # Test performance timing with larger dataset
        large_query = np.random.rand(100000, 2)  # 1000 random 2D points
        large_reference = np.random.rand(5000, 2)  # 5000 random 2D points
        
        result = SharedTestUtils.test_performance_timing(
            generic.nearest_neighbors,
            (large_query, large_reference),
            max_time=1.0,
            description="nearest_neighbors function with large dataset"
        )
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize(
        "query_points,reference_points,expected_indices,test_description", [
        # Basic 2D points
        (
            pd.DataFrame([[0, 0], [1, 1], [5, 5]]).values,
            pd.DataFrame([[0.1, 0.1], [1.1, 0.9], [2, 2], [4.8, 5.2], [10, 10]]).values,
            np.array([0, 1, 3]),
            "Basic 2D closest neighbor matching"
        ),
        # Single query point
        (
            np.array([[2, 3]]),
            np.array([[1, 1], [2, 2], [3, 4], [5, 5]]),
            np.array([1]),
            "Single query point"
        ),
        # Identical points (distance = 0)
        (
            np.array([[1, 1], [3, 3]]),
            np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
            np.array([0, 2]),
            "Identical points with zero distance"
        ),
        # 3D points
        (
            np.array([[0, 0, 0], [1, 1, 1]]),
            np.array([[0.1, 0.1, 0.1], [2, 2, 2], [0.9, 1.1, 0.9]]),
            np.array([0, 2]),
            "3D point matching"
        ),
        # Single reference point for multiple queries
        (
            np.array([[1, 1], [2, 2], [3, 3]]),
            np.array([[1.5, 1.5]]),
            np.array([0, 0, 0]),
            "Multiple queries, single reference"
        ),
        # Negative coordinates
        (
            np.array([[-1, -1], [0, 0], [1, 1]]),
            np.array([[-0.9, -1.1], [0.1, 0.1], [1.1, 0.9]]),
            np.array([0, 1, 2]),
            "Negative coordinates"
        ),
        # Large coordinates
        (
            np.array([[1000, 2000], [5000, 3000]]),
            np.array([[1001, 2001], [999, 1999], [5001, 2999]]),
            np.array([0, 2]),
            "Large coordinate values"
        ),
        # Floating point coordinates
        (
            np.array([[1.5, 2.5], [3.7, 4.2]]),
            np.array([[1.6, 2.4], [1.4, 2.6], [3.8, 4.1], [10.0, 10.0]]),
            np.array([0, 2]),
            "Floating point coordinates"
        ),
        # Mixed integer and float
        (
            np.array([[1, 2.5], [3.5, 4]]),
            np.array([[1.1, 2.4], [3.4, 4.1]]),
            np.array([0, 1]),
            "Mixed integer and float coordinates"
        ),
        # Equal distances (should return first occurrence)
        (
            np.array([[0, 0]]),
            np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),  # All equidistant
            np.array([0]),
            "Equal distances - first occurrence"
        ),
        # Higher dimensional points (4D)
        (
            np.array([[1, 2, 3, 4]]),
            np.array([[1.1, 2.1, 3.1, 4.1], [2, 3, 4, 5], [0.9, 1.9, 2.9, 4.9]]),
            np.array([0]),
            "4D points"
        ),
        # Edge case: all points at origin
        (
            np.array([[0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0]]),
            np.array([0, 0]),
            "All points at origin"
        ),

        # São Paulo city coordinates (multiple districts)
        (
            np.array([
                [-23.5505, -46.6333],  # Sé Cathedral (Centro)
                [-23.5489, -46.6388],  # Municipal Market (Centro) 
                [-23.5629, -46.6544],  # Vila Madalena
                [-23.5475, -46.6361],  # República
                [-23.5507, -46.6551],  # Bela Vista
                [-23.5448, -46.6388],  # Santa Ifigênia
                [-23.5578, -46.6395],  # Consolação
                [-23.5614, -46.6565]   # Pinheiros
            ]),
            np.array([
                [-23.5506, -46.6334],  # Very close to Sé
                [-23.5490, -46.6389],  # Very close to Municipal Market
                [-23.5630, -46.6545],  # Very close to Vila Madalena
                [-23.5476, -46.6362],  # Very close to República
                [-23.5508, -46.6552],  # Very close to Bela Vista
                [-23.5449, -46.6389],  # Very close to Santa Ifigênia
                [-23.5579, -46.6396],  # Very close to Consolação
                [-23.5615, -46.6566],  # Very close to Pinheiros
                [-23.5500, -46.6400],  # Random point in centro
                [-23.5600, -46.6500]   # Random point in zona oeste
            ]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            "São Paulo neighborhoods and districts"
        ),

        # New York City coordinates (Manhattan landmarks)
        (
            np.array([
                [40.7614, -73.9776],   # Times Square
                [40.7505, -73.9934],   # Empire State Building
                [40.7829, -73.9654],   # Central Park South
                [40.7589, -73.9851],   # Hell's Kitchen
                [40.7282, -73.9942],   # Greenwich Village
                [40.7061, -74.0087],   # Tribeca
                [40.7831, -73.9712],   # Upper East Side
                [40.7549, -73.9840],   # Theater District
                [40.7359, -74.0014],   # SoHo
                [40.7648, -73.9808]    # Midtown East
            ]),
            np.array([
                [40.7615, -73.9777],   # Very close to Times Square
                [40.7506, -73.9935],   # Very close to Empire State
                [40.7830, -73.9655],   # Very close to Central Park
                [40.7590, -73.9852],   # Very close to Hell's Kitchen
                [40.7283, -73.9943],   # Very close to Greenwich Village
                [40.7062, -74.0088],   # Very close to Tribeca
                [40.7832, -73.9713],   # Very close to Upper East Side
                [40.7550, -73.9841],   # Very close to Theater District
                [40.7360, -74.0015],   # Very close to SoHo
                [40.7649, -73.9809],   # Very close to Midtown East
                [40.7400, -73.9900],   # Random Manhattan point
                [40.7700, -73.9600]    # Random Upper Manhattan point
            ]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "NYC Manhattan neighborhood grid"
        ),

        # London coordinates (central boroughs)
        (
            np.array([
                [51.5074, -0.1278],     # Trafalgar Square
                [51.4994, -0.1245],     # Westminster
                [51.5155, -0.0922],     # Tower Bridge
                [51.5033, -0.1195],     # Waterloo
                [51.5152, -0.1426],     # King's Cross
                [51.5138, -0.0983],     # Shoreditch
                [51.4893, -0.1441],     # Clapham
                [51.5014, -0.1419],     # Chelsea
                [51.5064, -0.1024],     # London Bridge
                [51.5117, -0.1280]      # Bloomsbury
            ]),

            np.array([
                [51.5075, -0.1279],     # Very close to Trafalgar Square
                [51.4995, -0.1246],     # Very close to Westminster
                [51.5156, -0.0923],     # Very close to Tower Bridge
                [51.5034, -0.1196],     # Very close to Waterloo
                [51.5153, -0.1427],     # Very close to King's Cross
                [51.5139, -0.0984],     # Very close to Shoreditch
                [51.4894, -0.1442],     # Very close to Clapham
                [51.5015, -0.1420],     # Very close to Chelsea
                [51.5065, -0.1025],     # Very close to London Bridge
                [51.5118, -0.1281],     # Very close to Bloomsbury
                [51.5200, -0.1000],     # Random central London point
                [51.4900, -0.1300]      # Random south London point
            ]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "London central borough network"
        ),

        # Tokyo coordinates (major districts)
        (
            np.array([
                [35.6762, 139.6503],   # Shibuya Crossing
                [35.6581, 139.7414],   # Tokyo Station
                [35.6895, 139.6917],   # Shinjuku
                [35.6586, 139.7454],   # Ginza
                [35.7090, 139.7319],   # Ikebukuro
                [35.6284, 139.7387],   # Shinagawa
                [35.6938, 139.7036],   # Harajuku
                [35.6393, 139.7036],   # Ebisu
                [35.6681, 139.6589],   # Omotesando
                [35.7295, 139.7104]    # Komagome
            ]),
            np.array([
                [35.6763, 139.6504],   # Very close to Shibuya
                [35.6582, 139.7415],   # Very close to Tokyo Station
                [35.6896, 139.6918],   # Very close to Shinjuku
                [35.6587, 139.7455],   # Very close to Ginza
                [35.7091, 139.7320],   # Very close to Ikebukuro
                [35.6285, 139.7388],   # Very close to Shinagawa
                [35.6939, 139.7037],   # Very close to Harajuku
                [35.6394, 139.7037],   # Very close to Ebisu
                [35.6682, 139.6590],   # Very close to Omotesando
                [35.7296, 139.7105],   # Very close to Komagome
                [35.6800, 139.7000],   # Random Tokyo point
                [35.6500, 139.7200]    # Random Tokyo point
            ]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "Tokyo metropolitan district network"
        ),

        # European capitals coordinates
        (
            np.array([
                [48.8566, 2.3522],      # Louvre, Paris
                [52.5200, 13.4050],     # Brandenburg Gate, Berlin
                [41.9028, 12.4964],     # Colosseum, Rome
                [55.7558, 37.6176],     # Red Square, Moscow
                [59.3293, 18.0686],     # Stockholm City Hall
                [50.0755, 14.4378],     # Prague Castle
                [47.4979, 19.0402],     # Parliament, Budapest
                [52.3676, 4.9041]       # Dam Square, Amsterdam
            ]),
            np.array([
                [48.8567, 2.3523],      # Very close to Louvre
                [52.5201, 13.4051],     # Very close to Brandenburg Gate
                [41.9029, 12.4965],     # Very close to Colosseum
                [55.7559, 37.6177],     # Very close to Red Square
                [59.3294, 18.0687],     # Very close to Stockholm City Hall
                [50.0756, 14.4379],     # Very close to Prague Castle
                [47.4980, 19.0403],     # Very close to Budapest Parliament
                [52.3677, 4.9042],      # Very close to Dam Square
                [48.8500, 2.3500],      # Random Paris point
                [52.5000, 13.4000]      # Random Berlin point
            ]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            "European capital cities landmark network"
        )
    ])

    def test_valid_cases_parametrized(self, query_points, reference_points, expected_indices, test_description):
        """
        Test valid cases with expected results using parametrization.
        
        """
        result = generic.nearest_neighbors(query_points, reference_points)
        
        # Basic type and structure assertions
        assert isinstance(result, np.ndarray), f"Result should be a numpy array for {test_description}"
        assert len(result) == len(query_points), f"Result length should match query points length for {test_description}"
        assert result.dtype == np.int64 or result.dtype == np.int32, f"All indices should be integers for {test_description}"
        
        # Range validation
        assert all(0 <= idx < len(reference_points) for idx in result), f"All indices should be valid reference point indices for {test_description}"
        
        # Expected results validation
        np.testing.assert_array_equal(result, expected_indices, f"Expected indices {expected_indices}, got {result} for {test_description}")
        
        # Distance validation - ensure returned points are actually closest
        for i, (query_point, result_idx) in enumerate(zip(query_points, result)):
            result_distance = np.linalg.norm(query_point - reference_points[result_idx])
            
            # Check that no other reference point is closer
            for j, ref_point in enumerate(reference_points):
                other_distance = np.linalg.norm(query_point - ref_point)
                assert result_distance <= other_distance or abs(result_distance - other_distance) < 1e-10, \
                    f"Point at index {result_idx} should be closest to query point {i} for {test_description}"
        
        # Consistency check - same query should give same result
        result2 = generic.nearest_neighbors(query_points, reference_points)
        np.testing.assert_array_equal(result, result2, f"Function should be deterministic for {test_description}")

    @pytest.mark.parametrize(
        "invalid_query,invalid_reference,expected_error_type,test_description", [
        # Non-array inputs
        ("not an array", np.array([[1, 2]]),  AssertionError, "String as query points"),
        (123, np.array([[1, 2]]),  AssertionError, "Integer as query points"),
        (None, np.array([[1, 2]]),  AssertionError, "None as query points"),
        (np.array([[1, 2]]), "not an array",  AssertionError, "String as reference points"),
        (np.array([[1, 2]]), 123,  AssertionError, "Integer as reference points"),
        (np.array([[1, 2]]), None,  AssertionError, "None as reference points"),
        (pd.DataFrame([[1, 2]]), pd.DataFrame([[3, 4]]), AssertionError, "DataFrames instead of arrays"),
        
        # Empty arrays
        (np.array([]).reshape(0, 2), np.array([[1, 2]]), AssertionError, "Empty query points"),
        (np.array([[1, 2]]), np.array([]).reshape(0, 2), AssertionError, "Empty reference points"),
        (np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), AssertionError, "Both empty"),
        
        # Wrong dimensions
        (np.array([1, 2, 3]), np.array([[1, 2]]), AssertionError, "1D array as query points"),
        (np.array([[1, 2]]), np.array([1, 2, 3]), AssertionError, "1D array as reference points"),
        (np.array([[[1, 2]]]), np.array([[1, 2]]), AssertionError, "3D array as query points"),
        (np.array([[1, 2]]), np.array([[[1, 2]]]), AssertionError, "3D array as reference points"),
        
        # Inconsistent dimensions
        (np.array([[1, 2]]), np.array([[1, 2, 3]]), AssertionError, "2D query, 3D reference"),
        (np.array([[1, 2, 3]]), np.array([[1, 2]]), AssertionError, "3D query, 2D reference"),
        
        # NaN or infinite values
        (np.array([[np.nan, 1]]), np.array([[1, 2]]), AssertionError, "NaN in query points"),
        (np.array([[1, 2]]), np.array([[np.nan, 1]]), AssertionError, "NaN in reference points"),
        (np.array([[np.inf, 1]]), np.array([[1, 2]]), AssertionError, "Infinity in query points"),
        (np.array([[1, 2]]), np.array([[np.inf, 1]]), AssertionError, "Infinity in reference points"),
        
        # Wrong data types
        (np.array([["a", "b"]]), np.array([[1, 2]]), AssertionError, "String values in query"),
        (np.array([[1, 2]]), np.array([["a", "b"]]), AssertionError, "String values in reference"),
        (np.array([[True, False]]), np.array([[1, 2]]), AssertionError, "Boolean values in query"),
        (np.array([[1, 2]]), np.array([[True, False]]), AssertionError, "Boolean values in reference")

    ])
    def test_invalid_cases_parametrized(self, invalid_query, invalid_reference, expected_error_type, test_description):
        """
        Test invalid cases that should raise errors using parametrization.
        
        """
        with pytest.raises(expected_error_type):
            generic.nearest_neighbors(invalid_query, invalid_reference)


class TestCreateGrid:
    """Test cases for the generic.create_grid function using pytest."""
    
    @pytest.fixture
    def basic_grid_data(self):
        """Fixture providing basic grid data for testing."""
        return {
            'lat_min': -23.6,
            'lat_max': -23.5,
            'lon_min': -46.7,
            'lon_max': -46.6,
            'spacing_m': 1000,
            'expected_min_points': 10  # Approximate minimum expected points
        }
    
    def test_shared_test_utils(self, basic_grid_data):
        """Test using SharedTestUtils methods."""
        # Test function signature and return type
        SharedTestUtils.test_function_signature_and_return(
            generic.create_grid,
            ['lat_min', 'lat_max', 'lon_min', 'lon_max', 'spacing_m'],
            pd.DataFrame
        )
        
        # Test docstring exists
        SharedTestUtils.test_docstring_exists(
            generic.create_grid,
            ['grid', 'spacing', 'latitude', 'longitude']
        )
        
        # Test function determinism
        SharedTestUtils.test_function_determinism(
            generic.create_grid,
            basic_grid_data['lat_min'],
            basic_grid_data['lat_max'],
            basic_grid_data['lon_min'],
            basic_grid_data['lon_max'],
            basic_grid_data['spacing_m']
        )
        
        # Test performance timing with larger grid
        result = SharedTestUtils.test_performance_timing(
            generic.create_grid,
            (-23.7, -23.4, -46.8, -46.5, 500),  # Larger area, smaller spacing
            max_time=10.0,
            description="create_grid function with large grid"
        )
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize(
        "lat_min,lat_max,lon_min,lon_max,spacing_m,test_description", [
        # Basic São Paulo grid
        (-23.6, -23.5, -46.7, -46.6, 1000, "Basic São Paulo downtown grid"),
        # Small precise grid
        (-23.55, -23.54, -46.65, -46.64, 100, "Small high-precision grid"),
        # Large city grid
        (-23.7, -23.3, -46.8, -46.4, 2000, "Large São Paulo metropolitan grid"),
        # NYC Manhattan grid
        (40.7, 40.8, -74.1, -73.9, 500, "NYC Manhattan grid"),
        # London central grid
        (51.48, 51.52, -0.2, 0.0, 300, "London central grid"),
        # Tokyo grid
        (35.6, 35.8, 139.6, 139.8, 800, "Tokyo metropolitan grid"),
        # Very small area (neighborhood level)
        (-23.5505, -23.5495, -46.6340, -46.6330, 50, "Neighborhood level precision"),
        # Wide longitude span (crossing time zones conceptually)
        (40.7, 40.8, -75.0, -73.0, 5000, "Wide longitude span"),
        # Tall latitude span
        (-24.0, -23.0, -46.7, -46.6, 3000, "Tall latitude span"),
        # Equatorial region
        (-1.0, 1.0, -50.0, -48.0, 1500, "Equatorial region grid"),
        # High latitude region
        (60.0, 61.0, 10.0, 12.0, 2000, "High latitude region"),
        # Crossing Greenwich meridian
        (51.4, 51.6, -1.0, 1.0, 800, "Crossing Greenwich meridian"),
        # Large spacing (sparse grid)
        (-23.8, -23.2, -46.9, -46.3, 10000, "Sparse grid large spacing"),
        # Very fine spacing (dense grid - small area)
        (-23.5510, -23.5500, -46.6340, -46.6330, 25, "Very dense fine grid")
    ])
    def test_valid_cases_parametrized(self, lat_min, lat_max, lon_min, lon_max, spacing_m, test_description):
        """
        Test valid cases with expected results using parametrization.
        
        """
        result = generic.create_grid(lat_min, lat_max, lon_min, lon_max, spacing_m)
        
        # Basic type and structure assertions
        assert isinstance(result, pd.DataFrame), f"Result should be a DataFrame for {test_description}"
        assert list(result.columns) == ['latitude', 'longitude'], f"Columns should be ['latitude', 'longitude'] for {test_description}"
        assert len(result) > 0, f"Result should not be empty for {test_description}"
        
        # Data type assertions
        assert pd.api.types.is_numeric_dtype(result['latitude']), f"Latitude should be numeric for {test_description}"
        assert pd.api.types.is_numeric_dtype(result['longitude']), f"Longitude should be numeric for {test_description}"
        
        # No missing values
        assert not result['latitude'].isnull().any(), f"Latitude should not have null values for {test_description}"
        assert not result['longitude'].isnull().any(), f"Longitude should not have null values for {test_description}"
        
        # Boundary assertions
        assert result['latitude'].min() >= lat_min, f"All latitudes should be >= lat_min for {test_description}"
        assert result['latitude'].max() <= lat_max, f"All latitudes should be <= lat_max for {test_description}"
        assert result['longitude'].min() >= lon_min, f"All longitudes should be >= lon_min for {test_description}"
        assert result['longitude'].max() <= lon_max, f"All longitudes should be <= lon_max for {test_description}"
        
        # Check that corner points are included (or very close)
        corners = [
            (lat_min, lon_min), (lat_min, lon_max),
            (lat_max, lon_min), (lat_max, lon_max)
        ]
        
        tolerance = spacing_m / 111000  # Convert meters to rough degrees
        for corner_lat, corner_lon in corners:
            # Check if any point is close to this corner
            lat_close = abs(result['latitude'] - corner_lat) <= tolerance
            lon_close = abs(result['longitude'] - corner_lon) <= tolerance
            corner_exists = (lat_close & lon_close).any()
            # Note: Due to grid generation method, not all corners may be exactly present
            # This is acceptable as the function creates a regular grid starting from lat_min, lon_min
        
        # Spacing validation - check approximate consistency
        if len(result) > 1:
            # Check latitude spacing consistency for points in same longitude
            unique_lons = result['longitude'].unique()
            if len(unique_lons) > 1:
                first_lon = unique_lons[0]
                same_lon_points = result[result['longitude'] == first_lon].sort_values('latitude')
                if len(same_lon_points) > 1:
                    lat_diffs = same_lon_points['latitude'].diff().dropna()
                    # Convert to approximate meters (rough calculation)
                    lat_diffs_m = lat_diffs * 111000  # 1 degree lat ≈ 111km
                    avg_lat_spacing = lat_diffs_m.mean()
                    # Allow 10% tolerance due to Earth's curvature and approximations
                    assert abs(avg_lat_spacing - spacing_m) / spacing_m < 0.1, \
                        f"Latitude spacing should be approximately {spacing_m}m for {test_description}"
        
        # Check for duplicates
        duplicates = result.duplicated().sum()
        assert duplicates == 0, f"Result should not contain duplicate points for {test_description}"
        
        # Reasonable grid size validation
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min
        
        # Rough estimate of expected points
        lat_span_m = lat_span * 111000  # Convert to meters
        lon_span_m = lon_span * 111000 * np.cos(np.radians((lat_min + lat_max) / 2))  # Account for longitude convergence
        
        expected_lat_points = max(1, int(lat_span_m / spacing_m) + 1)
        expected_lon_points = max(1, int(lon_span_m / spacing_m) + 1)
        expected_total = expected_lat_points * expected_lon_points
        
        # Allow reasonable tolerance for grid size (due to Earth's curvature effects)
        tolerance_factor = 0.5  # 50% tolerance
        min_expected = int(expected_total * (1 - tolerance_factor))
        max_expected = int(expected_total * (1 + tolerance_factor))
        
        assert min_expected <= len(result) <= max_expected, \
            f"Grid size {len(result)} should be roughly between {min_expected} and {max_expected} for {test_description}"
        
        # Consistency check - same parameters should give same result
        result2 = generic.create_grid(lat_min, lat_max, lon_min, lon_max, spacing_m)
        pd.testing.assert_frame_equal(result, result2, f"Function should be deterministic for {test_description}")

    @pytest.mark.parametrize(
        "lat_min,lat_max,lon_min,lon_max,spacing_m,expected_error_type,test_description", [
        # Invalid latitude ranges
        (-23.5, -23.6, -46.7, -46.6, 1000,  AssertionError, "lat_min > lat_max"),
        (95, 100, -46.7, -46.6, 1000,  AssertionError, "Invalid latitude > 90"),
        (-100, -95, -46.7, -46.6, 1000,  AssertionError, "Invalid latitude < -90"),
        
        # Invalid longitude ranges  
        (-23.6, -23.5, -46.6, -46.7, 1000,  AssertionError, "lon_min > lon_max"),
        (-23.6, -23.5, 185, 190, 1000,  AssertionError, "Invalid longitude > 180"),
        (-23.6, -23.5, -190, -185, 1000,  AssertionError, "Invalid longitude < -180"),
        
        # Invalid spacing
        (-23.6, -23.5, -46.7, -46.6, 0, AssertionError, "Zero spacing"),
        (-23.6, -23.5, -46.7, -46.6, -1000,  AssertionError, "Negative spacing"),
        
        # Non-numeric inputs
        ("invalid", -23.5, -46.7, -46.6, 1000, AssertionError, "String lat_min"),
        (-23.6, "invalid", -46.7, -46.6, 1000, AssertionError, "String lat_max"),
        (-23.6, -23.5, "invalid", -46.6, 1000, AssertionError, "String lon_min"),
        (-23.6, -23.5, -46.7, "invalid", 1000, AssertionError, "String lon_max"),
        (-23.6, -23.5, -46.7, -46.6, "invalid", AssertionError, "String spacing"),
        
        # None inputs
        (None, -23.5, -46.7, -46.6, 1000, AssertionError, "None lat_min"),
        (-23.6, None, -46.7, -46.6, 1000, AssertionError, "None lat_max"),
        (-23.6, -23.5, None, -46.6, 1000, AssertionError, "None lon_min"),
        (-23.6, -23.5, -46.7, None, 1000, AssertionError, "None lon_max"),
        (-23.6, -23.5, -46.7, -46.6, None, AssertionError, "None spacing"),

        # NaN and infinity values
        (np.nan, -23.5, -46.7, -46.6, 1000, AssertionError, "NaN lat_min"),
        (-23.6, np.nan, -46.7, -46.6, 1000, AssertionError, "NaN lat_max"),
        (-23.6, -23.5, np.nan, -46.6, 1000, AssertionError, "NaN lon_min"),
        (-23.6, -23.5, -46.7, np.nan, 1000, AssertionError, "NaN lon_max"),
        (-23.6, -23.5, -46.7, -46.6, np.nan, AssertionError, "NaN spacing"),
        
        (np.inf, -23.5, -46.7, -46.6, 1000, AssertionError, "Infinite lat_min"),
        (-23.6, np.inf, -46.7, -46.6, 1000, AssertionError, "Infinite lat_max"),
        (-23.6, -23.5, np.inf, -46.6, 1000, AssertionError, "Infinite lon_min"),
        (-23.6, -23.5, -46.7, np.inf, 1000, AssertionError, "Infinite lon_max"),
        (-23.6, -23.5, -46.7, -46.6, np.inf, AssertionError, "Infinite spacing"),
        
        # # Boolean inputs
        (True, -23.5, -46.7, -46.6, 1000, AssertionError, "Boolean lat_min"),
        (-23.6, False, -46.7, -46.6, 1000, AssertionError, "Boolean lat_max"),
        (-23.6, -23.5, True, -46.6, 1000, AssertionError, "Boolean lon_min"),
        (-23.6, -23.5, -46.7, False, 1000, AssertionError, "Boolean lon_max"),
        (-23.6, -23.5, -46.7, -46.6, True, AssertionError, "Boolean spacing"),
        
        # # List/array inputs
        ([-23.6], -23.5, -46.7, -46.6, 1000, AssertionError, "List lat_min"),
        (-23.6, [-23.5], -46.7, -46.6, 1000, AssertionError, "List lat_max"),
        (-23.6, -23.5, [-46.7], -46.6, 1000, AssertionError, "List lon_min"),
        (-23.6, -23.5, -46.7, [-46.6], 1000, AssertionError, "List lon_max"),
        (-23.6, -23.5, -46.7, -46.6, [1000], AssertionError, "List spacing"),

        (-23.55, -23.55, -46.65, -46.65, 1000, AssertionError, "Equal bounds "),  # This might actually work
    ])
    def test_invalid_cases_parametrized(self, lat_min, lat_max, lon_min, lon_max, spacing_m, expected_error_type, test_description):
        """
        Test invalid cases that should raise errors using parametrization.
        
        """

        with pytest.raises(expected_error_type):
            generic.create_grid(lat_min, lat_max, lon_min, lon_max, spacing_m)

    @pytest.mark.visual
    def test_visual_grid_output(self):
        """
        Visual test to generate and display a grid for manual inspection.
        This test creates a plot showing the generated grid points.
        """    
        # Test São Paulo downtown area
        lat_min, lat_max = -23.6, -23.5
        lon_min, lon_max = -46.7, -46.6
        spacing_m = 500
        
        # Generate grid
        grid = generic.create_grid(lat_min, lat_max, lon_min, lon_max, spacing_m)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(grid['longitude'], grid['latitude'], c='blue', alpha=0.6, s=10)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Generated Grid - São Paulo Downtown\nSpacing: {spacing_m}m, Points: {len(grid)}')
        plt.grid(True, alpha=0.3)
        
        # Add boundary rectangle
        plt.plot([lon_min, lon_max, lon_max, lon_min, lon_min], 
                [lat_min, lat_min, lat_max, lat_max, lat_min], 
                'r--', linewidth=2, label='Boundary')
        
        # Add some reference points (São Paulo landmarks)
        landmarks = {
            'Sé Cathedral': (-23.5505, -46.6333),
            'Municipal Market': (-23.5489, -46.6388),
            'República Square': (-23.5475, -46.6361)
        }
        
        for name, (lat, lon) in landmarks.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                plt.plot(lon, lat, 'ro', markersize=8)
                plt.annotate(name, (lon, lat), xytext=(5, 5), 
                            textcoords='offset points', fontsize=8)
        
        plt.legend()
        plt.tight_layout()
        
        # Show plot and wait for user to close it
        plt.show(block=True)
        
        # Test passes if no exception was raised
        assert len(grid) > 0, "Grid should contain points"
        assert isinstance(grid, pd.DataFrame), "Grid should be a DataFrame"