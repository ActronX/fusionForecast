
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.preprocess import prepare_forecast_input

class MockModel:
    def make_future_dataframe(self, df, periods, regressors_df, n_historic_predictions):
        # Mock implementation that returns a DataFrame with 'ds' and 'y'
        future = regressors_df.copy()
        future['y'] = 0.0
        return future

class TestForecastPrep(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.n_lags = 4
        self.n_forecasts = 4
        self.regressor_names = ['reg1']
        
        # Create dummy history
        dates = pd.date_range(start='2024-01-01', periods=10, freq='15min')
        self.history = pd.DataFrame({
            'ds': dates,
            'y': np.random.rand(10),
            'reg1': np.random.rand(10)
        })
        
    def test_padding_short_chunk(self):
        # Create a short chunk (length 2, n_forecasts=4)
        chunk_dates = pd.date_range(start='2024-01-01 02:30', periods=2, freq='15min')
        chunk = pd.DataFrame({
            'ds': chunk_dates,
            'reg1': [0.5, 0.6]
        })
        
        step_input, chunk_padded, actual_len = prepare_forecast_input(
            model=self.model,
            chunk=chunk,
            current_history=self.history,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            regressor_names=self.regressor_names
        )
        
        # Verify padding
        self.assertEqual(len(chunk_padded), self.n_forecasts)
        self.assertEqual(actual_len, 2)
        # Check if padded values are copies of last row
        self.assertEqual(chunk_padded.iloc[2]['reg1'], 0.6)
        self.assertEqual(chunk_padded.iloc[3]['reg1'], 0.6)
        # Check timestamps are continuous
        self.assertEqual(chunk_padded.iloc[2]['ds'], pd.Timestamp('2024-01-01 03:00'))
        
    def test_full_length_chunk(self):
        # Create a full length chunk
        chunk_dates = pd.date_range(start='2024-01-01 02:30', periods=4, freq='15min')
        chunk = pd.DataFrame({
            'ds': chunk_dates,
            'reg1': [0.5, 0.6, 0.7, 0.8]
        })
        
        step_input, chunk_padded, actual_len = prepare_forecast_input(
            model=self.model,
            chunk=chunk,
            current_history=self.history,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            regressor_names=self.regressor_names
        )
        
        self.assertEqual(len(chunk_padded), 4)
        self.assertEqual(actual_len, 4)
        # Verify no padding happened (values match original)
        pd.testing.assert_frame_equal(chunk, chunk_padded)

if __name__ == '__main__':
    unittest.main()
