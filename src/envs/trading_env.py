"""
Custom gym-anytrading environment backed by technical indicators.
"""

# Import typing
from typing import List, Optional, Tuple

# Import third-party libraries
import numpy as np
import pandas as pd
from gym_anytrading.envs import StocksEnv


# Define default feature columns
DEFAULT_FEATURES: List[str] = [
    "close_pct",
    "rsi",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_pct",
    "bb_width",
    "volume_norm",
    "high_low_pct",
]


# Define technical trading environment
class TechnicalStocksEnv(StocksEnv):
    # Initialize environment
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        frame_bound: Tuple[int, int],
        features: Optional[List[str]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        # Resolve requested features
        requested_features = features if features is not None else DEFAULT_FEATURES

        # Keep only columns that exist in the dataframe
        self.feature_columns: List[str] = [
            column_name
            for column_name in requested_features
            if column_name in df.columns
        ]

        # Validate feature columns
        if not self.feature_columns:
            raise ValueError(
                "None of the requested feature columns exist in the DataFrame. "
                f"Requested: {requested_features}. Available: {list(df.columns)}."
            )

        # Initialize parent environment
        super().__init__(
            df=df,
            window_size=window_size,
            frame_bound=frame_bound,
            render_mode=render_mode,
        )

    # Process full dataframe into prices and signal features
    def _process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # Extract prices used internally by the environment
        prices = self.df["Close"].to_numpy(dtype=np.float64)

        # Build normalized feature arrays
        feature_arrays = []

        for column_name in self.feature_columns:
            feature_array = self.df[column_name].to_numpy(dtype=np.float64)
            feature_array = zscore_clip(feature_array)
            feature_arrays.append(feature_array)

        # Stack features into a 2D array
        signal_features = np.column_stack(feature_arrays)

        return prices, signal_features


# Standardize and clip array values
def zscore_clip(array: np.ndarray, clip_value: float = 5.0) -> np.ndarray:
    # Replace invalid values
    clean_array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute standard deviation
    standard_deviation = clean_array.std()

    # Normalize when possible
    if standard_deviation > 1e-8:
        clean_array = (clean_array - clean_array.mean()) / standard_deviation

    # Clip outliers
    clean_array = np.clip(clean_array, -clip_value, clip_value)

    return clean_array


# Create environment factory for vectorized environments
def make_env(
    df: pd.DataFrame,
    window_size: int,
    frame_bound: Tuple[int, int],
    features: Optional[List[str]] = None,
):
    # Create environment initializer
    def initialize_environment() -> TechnicalStocksEnv:
        # Return new environment instance
        return TechnicalStocksEnv(
            df=df,
            window_size=window_size,
            frame_bound=frame_bound,
            features=features,
        )

    return initialize_environment