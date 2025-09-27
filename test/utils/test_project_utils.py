import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import inspect
import time
from openpyxl import load_workbook
import typing


# Add the parent directory to the path to import the module
sys.path.append('utils')
import project_utils 

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


class TestGetDailyOvitraps:
    """Test cases for the project_utils.get_daily_ovitraps function using pytest."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample test data."""
        return pd.DataFrame({
            'dt_instal': [
                datetime(2023, 1, 1),
                datetime(2023, 1, 5),
                datetime(2023, 1, 10)
            ],
            'dt_col': [
                datetime(2023, 1, 3),  # 3 days (1st-3rd)
                datetime(2023, 1, 7),  # 3 days (5th-7th) 
                datetime(2023, 1, 12)  # 3 days (10th-12th)
            ],
            'narmad': ['A', 'B', 'A'],
            'novos': [6, 9, 12],  # Will be divided by number of days
            'days_expo': [2, 2, 2]  # Not used in function but realistic
            
        })
    
    @pytest.fixture
    def overlapping_data(self):
        """Fixture providing overlapping installation periods."""
        return pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'dt_col': [datetime(2023, 1, 3), datetime(2023, 1, 4)],
            'narmad': ['A', 'A'],
            'novos': [6, 4]
        })
    
    @pytest.fixture
    def empty_data(self):
        """Fixture providing empty DataFrame."""
        return pd.DataFrame(columns=['dt_instal', 'dt_col', 'narmad', 'novos'])
        
    def test_basic_functionality(self, sample_data):
        """Test basic functionality with sample data."""
        result = project_utils.get_daily_ovitraps(sample_data)
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that index is datetime
        assert pd.api.types.is_datetime64_any_dtype(result.index)
        
        # Check that columns correspond to unique narmad values
        expected_columns = ['A', 'B']
        assert sorted(result.columns.tolist()) == sorted(expected_columns)
        
    def test_egg_distribution_calculation(self, sample_data):
        """
        Test that eggs are correctly distributed across days.
        
        """
        result = project_utils.get_daily_ovitraps(sample_data)
        
        # For trap A on Jan 1-3: 6 eggs / 2 days = 3 eggs per day
        jan_1_trap_a = result.loc[datetime(2023, 1, 1), 'A']
        assert jan_1_trap_a == 3.0
        
        # For trap B on Jan 5-7: 9 eggs / 2 days = 4.5 eggs per day  
        jan_5_trap_b = result.loc[datetime(2023, 1, 5), 'B']
        assert jan_5_trap_b == 4.5
        
    def test_overlapping_periods_same_trap(self, overlapping_data):
        """
        Test behavior when same trap has overlapping installation periods.
        
        """
        result = project_utils.get_daily_ovitraps(overlapping_data)
        
        # On Jan 2-3, both installations should contribute
        # First: 6/2 = 3 per day, Second: 4/2 = 2 per day
        # On Jan 2: 3 + 2 = 5, On Jan 3: 2 = 2
        jan_1_trap_a = result.loc[datetime(2023, 1, 1), 'A']
        expected_jan_1 = 3.0 
        assert abs(jan_1_trap_a - expected_jan_1) < 0.01
        
        jan_2_trap_a = result.loc[datetime(2023, 1, 2), 'A']
        expected_jan_2 = 5.0
        assert abs(jan_2_trap_a - expected_jan_2) < 0.01

        jan_3_trap_a = result.loc[datetime(2023, 1, 3), 'A']
        expected_jan_3 = 2.0
        assert abs(jan_3_trap_a - expected_jan_3) < 0.01
        

        
    def test_missing_values_filled_with_nan(self, sample_data):
        """
        Test that missing values in the date range are filled with NaN.
        
        """
        result = project_utils.get_daily_ovitraps(sample_data)
        
        # Check that days without any trap data have NaN values
        # Jan 4 should have NaN for both traps since no trap was active
        jan_4_trap_a = result.loc[datetime(2023, 1, 4), 'A']
        jan_4_trap_b = result.loc[datetime(2023, 1, 4), 'B']
        
        assert pd.isna(jan_4_trap_a)
        assert pd.isna(jan_4_trap_b)
        
    def test_complete_date_range(self, sample_data):
        """
        Test that the result includes all dates in the range.
        
        """
        result = project_utils.get_daily_ovitraps(sample_data)
        
        # Check that date range is complete from min to max date
        expected_start = datetime(2023, 1, 1)
        expected_end = datetime(2023, 1, 12)
        expected_range = pd.date_range(expected_start, expected_end - pd.Timedelta(days=1), freq='D')
        
        assert result.index.equals(expected_range)
        
    def test_empty_dataframe_raises_error(self, empty_data):
        """Test behavior with empty input DataFrame."""
        with pytest.raises(AssertionError):
            project_utils.get_daily_ovitraps(empty_data)
            
    def test_data_types_and_structure(self, sample_data):
        """Test that the output has correct data types and structure."""
        result = project_utils.get_daily_ovitraps(sample_data)
        
        # Check that values are numeric (float)
        assert pd.api.types.is_numeric_dtype(result.dtypes['A'])
        assert pd.api.types.is_numeric_dtype(result.dtypes['B'])
        
        # Check that columns are sorted
        assert result.columns.tolist() == sorted(result.columns.tolist())
        
    def test_data_immutability(self, sample_data):
        """Test that the original DataFrame is not modified."""
        original_data = sample_data.copy()
        project_utils.get_daily_ovitraps(sample_data)
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(sample_data, original_data)
        
    @pytest.mark.parametrize("novos_values,expected_daily", [
        ([3], [1.5]),  # Single day, single value
        ([6], [3.0]),  # 6 eggs over 3 days = 2 per day
        ([0], [0.0]),  # Zero eggs
        ])
    def test_parametrized_egg_calculations(self, novos_values, expected_daily):
        """Parametrized test for different egg calculation scenarios."""
        test_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1)] * len(novos_values),
            'dt_col': [datetime(2023, 1, 3)] * len(novos_values),  # 3 days for non-single day tests
            'narmad': [f'Trap_{i}' for i in range(len(novos_values))],
            'novos': novos_values
        })
                    
        result = project_utils.get_daily_ovitraps(test_data)
        
        for i, expected in enumerate(expected_daily):
            trap_name = f'Trap_{i}'
            actual = result.loc[datetime(2023, 1, 1), trap_name]
            assert abs(actual - expected) < 0.01
            
    def test_multiple_traps_different_periods(self):
        """Test with multiple traps having different installation periods."""
        test_data = pd.DataFrame({
            'dt_instal': [
                datetime(2023, 1, 1),  # Trap A: 4 days
                datetime(2023, 1, 3),  # Trap B: 2 days  
                datetime(2023, 1, 6),  # Trap C: 1 day
            ],
            'dt_col': [
                datetime(2023, 1, 5),
                datetime(2023, 1, 5), 
                datetime(2023, 1, 7),
            ],
            'narmad': ['A', 'B', 'C'],
            'novos': [10, 6, 4]  # 2, 2, 2 eggs per day respectively
        })
        
        result = project_utils.get_daily_ovitraps(test_data)
        
        # Check specific dates
        assert result.loc[datetime(2023, 1, 1), 'A'] == 2.5  # 10/4 days
        assert result.loc[datetime(2023, 1, 3), 'B'] == 3.0  # 6/2 days
        assert result.loc[datetime(2023, 1, 6), 'C'] == 4.0  # 4/1 days
        
        # Check that traps have NaN when not active
        assert pd.isna(result.loc[datetime(2023, 1, 1), 'B'])  # B not active yet
        assert pd.isna(result.loc[datetime(2023, 1, 6), 'A'])  # A no longer active

    def test_zero_novos_values(self):
        """
        Test behavior with zero 'novos' values (should be valid).
        
        """
        
        zero_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1)],
            'dt_col': [datetime(2023, 1, 3)],
            'narmad': ['A'],
            'novos': [0]  # Zero eggs
        })
        
        # Should work and produce zero daily values
        result = project_utils.get_daily_ovitraps(zero_data)
        assert result.loc[datetime(2023, 1, 1), 'A'] == 0.0

    # Additional test functions for edge cases
    def test_large_date_range(self):
        """
        Test with a large date range to ensure performance is acceptable.
        
        """
        test_data = pd.DataFrame({
            'dt_instal': [datetime(2020, 1, 1)],
            'dt_col': [datetime(2021, 1, 1)],  # Full year
            'narmad': ['A'],
            'novos': [366]  # Leap year, so 366 days = 1 egg per day
        })
        
        result = project_utils.get_daily_ovitraps(test_data)
        
        # Check that we have the full year
        assert len(result)  == 366  # 2020 is a leap year
        
        # Check that daily average is correct
        assert result.loc[datetime(2020, 1, 1), 'A'] == 1.0

    def test_numeric_trap_names(self):
        """
        Test that numeric trap names are handled correctly.
        
        """
        test_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1), datetime(2023, 1, 1)],
            'dt_col': [datetime(2023, 1, 3), datetime(2023, 1, 3)],
            'narmad': [1, 2],  # Numeric trap names
            'novos': [6, 9]
        })
        
        result = project_utils.get_daily_ovitraps(test_data)
        
        # Check that numeric columns are preserved
        assert 1 in result.columns
        assert 2 in result.columns
        assert result.loc[datetime(2023, 1, 1), 1] == 3.0
        assert result.loc[datetime(2023, 1, 1), 2] == 4.5

    @pytest.mark.parametrize(
        "wrong_input", 
        [
        "not a dataframe", 
        None,
        [1,2,3],
        123, 
        12.5, 
        True, 
        {'key': 'value'}, 
        set([1, 2, 3]), 
        ],
        ids=["string", "None", "list", "integer", "float", "boolean", "dictionary", "set"]
     )
    def test_error_handling(self,wrong_input):
        """Test various error conditions and edge cases."""
        
        # Test with non-DataFrame input
        with pytest.raises(AssertionError, match="Input must be a DataFrame"):
            project_utils.get_daily_ovitraps(wrong_input)

    def test_missing_required_columns(self):
        """Test behavior when required columns are missing."""
        
        # Missing 'dt_instal' column
        incomplete_data = pd.DataFrame({
            'dt_col': [datetime(2023, 1, 3)],
            'narmad': ['A'],
            'novos': [5]
        })
        with pytest.raises(AssertionError, match="DataFrame must contain"):
            project_utils.get_daily_ovitraps(incomplete_data)
        
        # Missing 'dt_col' column
        incomplete_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1)],
            'narmad': ['A'],
            'novos': [5]
        })
        with pytest.raises(AssertionError, match="DataFrame must contain"):
            project_utils.get_daily_ovitraps(incomplete_data)
        
        # Missing 'narmad' column
        incomplete_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1)],
            'dt_col': [datetime(2023, 1, 3)],
            'novos': [5]
        })
        with pytest.raises(AssertionError, match="DataFrame must contain"):
            project_utils.get_daily_ovitraps(incomplete_data)
        
        # Missing 'novos' column
        incomplete_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1)],
            'dt_col': [datetime(2023, 1, 3)],
            'narmad': ['A']
        })
        with pytest.raises(AssertionError, match="DataFrame must contain"):
            project_utils.get_daily_ovitraps(incomplete_data)

    def test_invalid_datetime_columns(self):
        """Test behavior with invalid datetime columns."""
        
        # Non-datetime 'dt_instal' column
        invalid_data = pd.DataFrame({
            'dt_instal': ['2023-01-01', '2023-01-02'],  # String instead of datetime
            'dt_col': [datetime(2023, 1, 3), datetime(2023, 1, 4)],
            'narmad': ['A', 'B'],
            'novos': [5, 6]
        })
        with pytest.raises(AssertionError, match="'dt_instal' column must be of datetime type"):
            project_utils.get_daily_ovitraps(invalid_data)
        
        # Non-datetime 'dt_col' column
        invalid_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'dt_col': ['2023-01-03', '2023-01-04'],  # String instead of datetime
            'narmad': ['A', 'B'],
            'novos': [5, 6]
        })
        with pytest.raises(AssertionError, match="'dt_col' column must be of datetime type"):
            project_utils.get_daily_ovitraps(invalid_data)

    def test_invalid_numeric_columns(self):
        """Test behavior with invalid numeric columns."""
        
        # Non-numeric 'novos' column
        invalid_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1)],
            'dt_col': [datetime(2023, 1, 3)],
            'narmad': ['A'],
            'novos': ['not_a_number']  # String instead of number
        })
        with pytest.raises(AssertionError, match="'novos' column must be numeric"):
            project_utils.get_daily_ovitraps(invalid_data)

    def test_invalid_date_logic(self):
        """Test behavior when dt_col is before dt_instal."""
        
        # dt_col before dt_instal
        invalid_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 5)],
            'dt_col': [datetime(2023, 1, 3)],  # Collection before installation
            'narmad': ['A'],
            'novos': [5]
        })
        with pytest.raises(AssertionError, match="'dt_col' must be greater than or equal to 'dt_instal'"):
            project_utils.get_daily_ovitraps(invalid_data)

    def test_negative_novos_values(self):
        """Test behavior with negative 'novos' values."""
        
        invalid_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1)],
            'dt_col': [datetime(2023, 1, 3)],
            'narmad': ['A'],
            'novos': [-5]  # Negative eggs count
        })
        with pytest.raises(AssertionError, match="'novos' must be non-negative"):
            project_utils.get_daily_ovitraps(invalid_data)

    def test_nan_values_in_critical_columns(self):
        """Test behavior with NaN values in critical columns."""
        
        # NaN in dt_instal
        invalid_data = pd.DataFrame({
            'dt_instal': [pd.NaT],  # NaT (Not a Time)
            'dt_col': [datetime(2023, 1, 3)],
            'narmad': ['A'],
            'novos': [5]
        })
        # This should raise an error during date range creation
        with pytest.raises(AssertionError):
            project_utils.get_daily_ovitraps(invalid_data)
        
        # NaN in dt_col
        invalid_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1)],
            'dt_col': [pd.NaT],  # NaT (Not a Time)
            'narmad': ['A'],
            'novos': [5]
        })
        with pytest.raises(AssertionError):
            project_utils.get_daily_ovitraps(invalid_data)

    def test_nan_values_in_novos_column(self):
        """Test behavior with NaN values in 'novos' column."""
        
        # NaN values should be handled gracefully
        data_with_nan = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1), datetime(2023, 1, 5)],
            'dt_col': [datetime(2023, 1, 3), datetime(2023, 1, 7)],
            'narmad': ['A', 'B'],
            'novos': [5, np.nan]  # One valid, one NaN
        })
        
        # Should raise an error
        with pytest.raises(AssertionError, match="'novos' must not contain null values"):
            project_utils.get_daily_ovitraps(data_with_nan)

    def test_empty_narmad_values(self):
        """Test behavior with empty or None narmad values."""
        
        # Empty string in narmad
        data_with_empty = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1)],
            'dt_col': [datetime(2023, 1, 3)],
            'narmad': [''],  # Empty string
            'novos': [5]
        })
        
        # Should work but create a column with empty name
        result = project_utils.get_daily_ovitraps(data_with_empty)
        assert '' in result.columns

    def test_mixed_data_types_in_columns(self):
        """Test behavior with mixed data types that should cause errors."""
        
        # Mixed types in novos (some numeric, some string)
        mixed_data = pd.DataFrame({
            'dt_instal': [datetime(2023, 1, 1), datetime(2023, 1, 5)],
            'dt_col': [datetime(2023, 1, 3), datetime(2023, 1, 7)],
            'narmad': ['A', 'B'],
            'novos': [5, 'invalid']  # Mixed types
        })
        
        with pytest.raises(AssertionError, match="'novos' column must be numeric"):
            project_utils.get_daily_ovitraps(mixed_data)

    def test_future_dates(self):
        """Test behavior with future dates."""
        
        future_data = pd.DataFrame({
            'dt_instal': [datetime(2025, 1, 1)],
            'dt_col': [datetime(2025, 1, 3)],
            'narmad': ['A'],
            'novos': [5]
        })
        
        # Should work fine with future dates
        result = project_utils.get_daily_ovitraps(future_data)
        assert isinstance(result, pd.DataFrame)
        assert result.loc[datetime(2025, 1, 1), 'A'] == 5/2

class TestGetOverlappedSamples:
    """Test cases for the project_utils.get_overlapped_samples function using pytest."""
    
    @pytest.fixture
    def basic_overlapping_data(self):
        """Fixture providing basic overlapping data."""
        return pd.DataFrame({
            'narmad': [1, 1, 2, 2],
            'dtinstal': ['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-03'],
            'dtcol': ['2023-01-03', '2023-01-04', '2023-02-03', '2023-02-05'],
            'nplaca': ['A001', 'A002', 'B001', 'B002'],
            'novos': [5, 3, 8, 4]
        })
    
    @pytest.fixture
    def non_overlapping_data(self):
        """Fixture providing non-overlapping data."""
        return pd.DataFrame({
            'narmad': [1, 1, 2],
            'dtinstal': ['2023-01-01', '2023-01-05', '2023-02-01'],
            'dtcol': ['2023-01-03', '2023-01-07', '2023-02-03'],
            'nplaca': ['A001', 'A002', 'B001'],
            'novos': [5, 3, 8]
        })
    
    def test_shared_test_utils(self, basic_overlapping_data):
        """Test using SharedTestUtils methods."""
        # Test function signature and return type
        SharedTestUtils.test_function_signature_and_return(
            project_utils.get_overlapped_samples,
            ['ovitraps_data'],
            typing.List[typing.Tuple[str, str]]
        )
        
        # Test docstring exists
        SharedTestUtils.test_docstring_exists(
            project_utils.get_overlapped_samples,
            ['overlapping', 'nplaca', 'periods']
        )
        
        # Test data integrity (original data should not be modified)
        original_data = basic_overlapping_data.copy()
        project_utils.get_overlapped_samples(basic_overlapping_data)
        SharedTestUtils.test_data_integrity(original_data, basic_overlapping_data)
        
        # Test function determinism
        SharedTestUtils.test_function_determinism(
            project_utils.get_overlapped_samples, 
            basic_overlapping_data
        )
        
        # Test performance timing
        result = SharedTestUtils.test_performance_timing(
            project_utils.get_overlapped_samples,
            (basic_overlapping_data,),
            max_time=2.0,
            description="get_overlapped_samples function"
        )
        assert isinstance(result, list)

    @pytest.mark.parametrize(
        "test_case,expected_overlaps", [
        # Basic overlapping case - same trap, overlapping periods (≤1 day gap)
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': ['2023-01-01', '2023-01-02'],
                'dtcol': ['2023-01-03', '2023-01-04'],
                'nplaca': ['A001', 'A002']
            }),
            [('A001', 'A002')]
        ),
        # Same day collection and installation (0 day gap)
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': ['2023-01-01', '2023-01-03'],
                'dtcol': ['2023-01-03', '2023-01-05'],
                'nplaca': ['A001', 'A002']
            }),
            [('A001', 'A002')]
        ),
        # Exactly 1 day gap (should not be included)
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': ['2023-01-01', '2023-01-04'],
                'dtcol': ['2023-01-03', '2023-01-06'],
                'nplaca': ['A001', 'A002']
            }),
            []
        ),
        # Multiple traps with overlaps
        (
            pd.DataFrame({
                'narmad': [1, 1, 2, 2, 2],
                'dtinstal': ['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-02', '2023-03-01'],
                'dtcol': ['2023-01-03', '2023-01-04', '2023-02-03', '2023-02-04', '2023-03-03'],
                'nplaca': ['A001', 'A002', 'B001', 'B002', 'B003']
            }),
            [('A001', 'A002'), ('B001', 'B002')]
        ),
        # Multiple overlaps for same trap (3 consecutive periods)
        (
            pd.DataFrame({
                'narmad': [1, 1, 1],
                'dtinstal': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'dtcol': ['2023-01-02', '2023-01-03', '2023-01-04'],
                'nplaca': ['A001', 'A002', 'A003']
            }),
            [('A001', 'A002'), ('A002', 'A003')]
        ),
        # No overlaps - gap > 1 day
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': ['2023-01-01', '2023-01-05'],
                'dtcol': ['2023-01-03', '2023-01-07'],
                'nplaca': ['A001', 'A002']
            }),
            []
        ),
        # Single trap (no overlaps possible)
        (
            pd.DataFrame({
                'narmad': [1],
                'dtinstal': ['2023-01-01'],
                'dtcol': ['2023-01-03'],
                'nplaca': ['A001']
            }),
            []
        ),
        # Different traps (no overlaps)
        (
            pd.DataFrame({
                'narmad': [1, 2, 3],
                'dtinstal': ['2023-01-01', '2023-01-01', '2023-01-01'],
                'dtcol': ['2023-01-03', '2023-01-03', '2023-01-03'],
                'nplaca': ['A001', 'B001', 'C001']
            }),
            []
        ),
        # Same trap with non-overlapping periods interspersed
        (
            pd.DataFrame({
                'narmad': [1, 1, 1],
                'dtinstal': ['2023-01-01', '2023-01-10', '2023-01-05'],
                'dtcol': ['2023-01-03', '2023-01-12', '2023-01-07'],
                'nplaca': ['A001', 'A002', 'A003']
            }),
            []  # After sorting: A001 (1-3), A003 (5-7), A002 (10-12) - no overlaps
        ),
        # Numeric nplaca values
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': ['2023-01-01', '2023-01-02'],
                'dtcol': ['2023-01-03', '2023-01-04'],
                'nplaca': [1001, 1002]
            }),
            [(1001, 1002)]
        ),
        # Complex scenario with multiple traps and mixed overlaps
        (
            pd.DataFrame({
                'narmad': [1, 1, 1, 2, 2, 3],
                'dtinstal': ['2023-01-01', '2023-01-02', '2023-01-10', '2023-02-01', '2023-02-05', '2023-03-01'],
                'dtcol': ['2023-01-03', '2023-01-04', '2023-01-12', '2023-02-03', '2023-02-07', '2023-03-03'],
                'nplaca': ['A001', 'A002', 'A003', 'B001', 'B002', 'C001']
            }),
            [('A001', 'A002')]  # Only first pair overlaps, others don't
        )
    ])
    def test_valid_cases_parametrized(self, test_case, expected_overlaps):
        """
        Test valid cases with expected results using parametrization.
        
        """
        result = project_utils.get_overlapped_samples(test_case)
        
        # Basic type and structure assertions
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(item, tuple) for item in result), "All items should be tuples"
        assert all(len(item) == 2 for item in result), "All tuples should have length 2"
        
        # Content assertions
        assert len(result) == len(expected_overlaps), f"Expected {len(expected_overlaps)} overlaps, got {len(result)}"
        
        # Sort both lists to handle order differences
        result_sorted = sorted(result)
        expected_sorted = sorted(expected_overlaps)
        
        assert result_sorted == expected_sorted, f"Expected overlaps {expected_sorted}, got {result_sorted}"
        
        # Additional property assertions
        all_nplaca_in_result = [item for sublist in result for item in sublist]
        if not test_case.empty:
            available_nplaca = set(test_case['nplaca'].values)
            assert all(nplaca in available_nplaca for nplaca in all_nplaca_in_result), \
                "All nplaca values in result should exist in input data"
        
        # Verify no duplicate pairs
        assert len(result) == len(set(result)), "Result should not contain duplicate pairs"
        
        # If no overlaps expected, ensure empty result
        if len(expected_overlaps) == 0:
            assert result == [], "Should return empty list when no overlaps exist"

    @pytest.mark.parametrize(
        "invalid_input,expected_error_pattern", [
        # Non-DataFrame inputs
        ("not a dataframe", ".*"),
        (None, ".*"),
        ([1, 2, 3], ".*"),
        (123, ".*"),
        (12.5, ".*"),
        (True, ".*"),
        ({'key': 'value'}, ".*"),
        (set([1, 2, 3]), ".*"),
        # Missing required columns
        (
            pd.DataFrame({'narmad': [1], 'dtinstal': ['2023-01-01'], 'nplaca': ['A001']}), # Missing dtcol
            ".*"
        ),
        (
            pd.DataFrame({'dtinstal': ['2023-01-01'], 'dtcol': ['2023-01-03'], 'nplaca': ['A001']}), # Missing narmad
            ".*"
        ),
        (
            pd.DataFrame({'narmad': [1], 'dtcol': ['2023-01-03'], 'nplaca': ['A001']}), # Missing dtinstal
            ".*"
        ),
        (
            pd.DataFrame({'narmad': [1], 'dtinstal': ['2023-01-01'], 'dtcol': ['2023-01-03']}), # Missing nplaca
            ".*"
        ),
        # Invalid date formats that cause parsing errors
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': ['invalid-date', '2023-01-02'],
                'dtcol': ['2023-01-03', '2023-01-04'],
                'nplaca': ['A001', 'A002']
            }),
            ".*"
        ),
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': ['2023-01-01', '2023-01-02'],
                'dtcol': ['invalid-date', '2023-01-04'],
                'nplaca': ['A001', 'A002']
            }),
            ".*"
        ),
        # NaN/None values in critical columns
        (
            pd.DataFrame({
                'narmad': [1, None],
                'dtinstal': ['2023-01-01', '2023-01-02'],
                'dtcol': ['2023-01-03', '2023-01-04'],
                'nplaca': ['A001', 'A002']
            }),
            ".*"
        ),
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': [None, '2023-01-02'],
                'dtcol': ['2023-01-03', '2023-01-04'],
                'nplaca': ['A001', 'A002']
            }),
            ".*"
        ),
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': ['2023-01-01', '2023-01-02'],
                'dtcol': ['2023-01-03', None],
                'nplaca': ['A001', 'A002']
            }),
            ".*"
        ),
        (
            pd.DataFrame({
                'narmad': [1, 1],
                'dtinstal': ['2023-01-01', '2023-01-02'],
                'dtcol': ['2023-01-03', '2023-01-04'],
                'nplaca': ['A001', None]
            }),
            ".*"
        )
    ])
    def test_invalid_cases_parametrized(self, invalid_input, expected_error_pattern):
        """
        Test invalid cases that should raise errors using parametrization.
        
        """
        with pytest.raises((AssertionError, AttributeError, KeyError, ValueError, TypeError)):
            project_utils.get_overlapped_samples(invalid_input)