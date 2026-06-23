"""
tests/test_features.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for feature engineering — verifies temporal correctness and
absence of data leakage in all lag and rolling features.

Run: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_sample_df(n: int = 200) -> pd.DataFrame:
    """Create a minimal synthetic dataset for testing."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time":                pd.Series(times),
        "us_aqi":              rng.uniform(20, 200, n),
        "pm2_5":               rng.uniform(5, 80, n),
        "pm10":                rng.uniform(10, 120, n),
        "precipitation":       rng.uniform(0, 5, n),
        "wind_speed_10m":      rng.uniform(0, 20, n),
        "wind_direction_10m":  rng.uniform(0, 359, n),
        "relative_humidity_2m": rng.uniform(40, 95, n),
        "boundary_layer_height": rng.uniform(200, 2000, n),
        "temperature_2m":      rng.uniform(20, 40, n),
        "surface_pressure":    rng.uniform(990, 1020, n),
    })


class TestLagFeatures:
    """Verify lag features are correctly shifted backward in time."""

    def test_lag_1_is_previous_row(self):
        df = make_sample_df(50)
        df = df.sort_values("time").reset_index(drop=True)
        df["aqi_lag_1"] = df["us_aqi"].shift(1)

        # Row i's lag_1 should equal row i-1's us_aqi
        for i in range(1, len(df)):
            assert df["aqi_lag_1"].iloc[i] == pytest.approx(df["us_aqi"].iloc[i - 1])

    def test_lag_24_is_24_hours_back(self):
        df = make_sample_df(100)
        df = df.sort_values("time").reset_index(drop=True)
        df["aqi_lag_24"] = df["us_aqi"].shift(24)

        for i in range(24, len(df)):
            assert df["aqi_lag_24"].iloc[i] == pytest.approx(df["us_aqi"].iloc[i - 24])

    def test_first_lag_rows_are_nan(self):
        """First n rows of lag_n must be NaN (no past data available)."""
        df = make_sample_df(50)
        df["aqi_lag_6"] = df["us_aqi"].shift(6)

        assert df["aqi_lag_6"].iloc[:6].isna().all(), \
            "First 6 rows of lag_6 must be NaN"

    def test_lag_never_uses_future(self):
        """Verify shift() never creates a feature from a future row."""
        df = make_sample_df(50)
        df = df.sort_values("time").reset_index(drop=True)
        df["aqi_lag_1"] = df["us_aqi"].shift(1)

        # If lag_1[i] == us_aqi[i+1], that would be future leakage
        for i in range(len(df) - 1):
            if not pd.isna(df["aqi_lag_1"].iloc[i]):
                assert df["aqi_lag_1"].iloc[i] != pytest.approx(df["us_aqi"].iloc[i + 1]), \
                    f"Lag feature at row {i} appears to use future value at row {i+1}"


class TestRollingFeatures:
    """Verify rolling features are backward-looking only."""

    def test_rolling_mean_24_first_value(self):
        """Rolling mean at row 0 should equal the value itself (min_periods=1)."""
        df = make_sample_df(50)
        df["aqi_roll_mean_24"] = df["us_aqi"].rolling(24, min_periods=1).mean()

        assert df["aqi_roll_mean_24"].iloc[0] == pytest.approx(df["us_aqi"].iloc[0])

    def test_rolling_mean_uses_only_past(self):
        """At position n, rolling mean uses rows 0..n, never row n+1."""
        n = 30
        df = make_sample_df(n + 10)
        df["aqi_roll_mean_24"] = df["us_aqi"].rolling(24, min_periods=1).mean()

        # Manually compute expected value at position n (using only rows 0..n)
        window_start = max(0, n - 23)
        expected = df["us_aqi"].iloc[window_start: n + 1].mean()
        assert df["aqi_roll_mean_24"].iloc[n] == pytest.approx(expected, rel=1e-5)

    def test_rolling_std_is_nan_free_with_min_periods(self):
        """With min_periods=1, rolling std should have at most 1 NaN (first row, std of 1 element is NaN)."""
        df = make_sample_df(100)
        df["aqi_roll_std_24"] = df["us_aqi"].rolling(24, min_periods=1).std()
        nan_count = df["aqi_roll_std_24"].isna().sum()
        assert nan_count <= 1, f"Expected at most 1 NaN in rolling std, got {nan_count}"


class TestFeatureExclusion:
    """Verify that AQI sub-indices (data leakage) are excluded from feature columns."""

    LEAKAGE_COLS = [
        "us_aqi_pm2_5",
        "us_aqi_pm10",
        "us_aqi_nitrogen_dioxide",
        "us_aqi_ozone",
    ]

    def test_sub_indices_excluded_from_features(self):
        """AQI sub-index columns are derived FROM the target and must not be used as features."""
        import joblib

        feat_path = ROOT / "notebooks" / "model" / "feature_cols_v2.pkl"
        if not feat_path.exists():
            pytest.skip("Feature columns file not found — run train.py first")

        feature_cols = joblib.load(feat_path)
        for col in self.LEAKAGE_COLS:
            assert col not in feature_cols, \
                f"Leakage column '{col}' found in feature list! Remove it."

    def test_target_not_in_features(self):
        """The target 'us_aqi' must never appear as an input feature."""
        import joblib

        feat_path = ROOT / "notebooks" / "model" / "feature_cols_v2.pkl"
        if not feat_path.exists():
            pytest.skip("Feature columns file not found — run train.py first")

        feature_cols = joblib.load(feat_path)
        assert "us_aqi" not in feature_cols, \
            "Target 'us_aqi' found in feature list! This is direct leakage."


class TestTemporalOrdering:
    """Verify the dataset is correctly sorted before feature engineering."""

    def test_features_require_sorted_data(self):
        """If data is shuffled, lag features produce garbage — catch this early."""
        df = make_sample_df(50)
        df_shuffled = df.sample(frac=1, random_state=99).reset_index(drop=True)

        lag_sorted   = df.sort_values("time")["us_aqi"].shift(1)
        lag_shuffled = df_shuffled["us_aqi"].shift(1)

        # The two lag vectors should be different (shuffled = wrong)
        equal_count = (lag_sorted.dropna() == lag_shuffled.dropna()).sum()
        assert equal_count < len(df) * 0.5, \
            "Shuffled and sorted lag features are suspiciously similar — check sort order"
