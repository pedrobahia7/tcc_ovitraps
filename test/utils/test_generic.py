# Add the parent directory to the path to import the module
import sys
import pytest
import inspect
import time
import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook


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