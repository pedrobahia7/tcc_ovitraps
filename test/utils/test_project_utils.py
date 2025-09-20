import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Add the parent directory to the path to import the module
sys.path.append('utils')
import project_utils 


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
            'novos': [6, 9, 12]  # Will be divided by number of days
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
        # On Jan 2: 3 + 2 = 5, On Jan 3: 3 + 2 = 5
        jan_1_trap_a = result.loc[datetime(2023, 1, 1), 'A']
        expected_jan_1 = 3.0 
        assert abs(jan_1_trap_a - expected_jan_1) < 0.01
        
        jan_2_trap_a = result.loc[datetime(2023, 1, 2), 'A']
        expected_jan_2 = 5.0
        assert abs(jan_2_trap_a - expected_jan_2) < 0.01

        jan_3_trap_a = result.loc[datetime(2023, 1, 3), 'A']
        expected_jan_3 = 5.0
        assert abs(jan_3_trap_a - expected_jan_3) < 0.01
        
        jan_4_trap_a = result.loc[datetime(2023, 1, 4), 'A']
        expected_jan_4 = 2.0
        assert abs(jan_4_trap_a - expected_jan_4) < 0.01

        
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
        expected_range = pd.date_range(expected_start, expected_end, freq='D')
        
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
        assert pd.isna(result.loc[datetime(2023, 1, 7), 'A'])  # A no longer active

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
        assert len(result) - 1 == 366  # 2020 is a leap year
        
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
