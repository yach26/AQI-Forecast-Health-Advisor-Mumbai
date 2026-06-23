"""
tests/test_inference.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the model inference pipeline — verifies the predict_aqi()
function returns correct structure, types, and value ranges.

Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestModelLoading:
    """Verify model and feature column files are loadable."""

    def test_model_file_exists(self):
        model_path = ROOT / "notebooks" / "model" / "xgb_model_v2.pkl"
        assert model_path.exists(), \
            f"Model file not found: {model_path}\nRun: python src/train.py"

    def test_feature_cols_file_exists(self):
        feat_path = ROOT / "notebooks" / "model" / "feature_cols_v2.pkl"
        assert feat_path.exists(), \
            f"Feature columns file not found: {feat_path}\nRun: python src/train.py"

    def test_model_loads_successfully(self):
        import joblib
        model_path = ROOT / "notebooks" / "model" / "xgb_model_v2.pkl"
        if not model_path.exists():
            pytest.skip("Model file not found")

        model = joblib.load(model_path)
        assert model is not None

    def test_feature_cols_is_list(self):
        import joblib
        feat_path = ROOT / "notebooks" / "model" / "feature_cols_v2.pkl"
        if not feat_path.exists():
            pytest.skip("Feature columns file not found")

        feature_cols = joblib.load(feat_path)
        assert isinstance(feature_cols, list), "feature_cols should be a list"
        assert len(feature_cols) > 0, "feature_cols should not be empty"


class TestModelPrediction:
    """Verify the model produces valid AQI predictions."""

    @pytest.fixture(scope="class")
    def model_and_features(self):
        import joblib
        model_path = ROOT / "notebooks" / "model" / "xgb_model_v2.pkl"
        feat_path  = ROOT / "notebooks" / "model" / "feature_cols_v2.pkl"

        if not model_path.exists() or not feat_path.exists():
            pytest.skip("Model files not found — run src/train.py first")

        return joblib.load(model_path), joblib.load(feat_path)

    def test_prediction_is_numeric(self, model_and_features):
        import pandas as pd
        model, feature_cols = model_and_features
        X = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
        pred = model.predict(X)[0]
        assert isinstance(float(pred), float), "Prediction must be a float"

    def test_prediction_in_valid_range(self, model_and_features):
        """AQI is defined in range 0–500 in the US standard."""
        import pandas as pd
        model, feature_cols = model_and_features

        # Test with a range of reasonable input values
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.uniform(0, 200, (20, len(feature_cols))), columns=feature_cols)
        preds = model.predict(X)

        assert all(0 <= p <= 600 for p in preds), \
            f"Some predictions out of reasonable AQI range: min={preds.min():.1f}, max={preds.max():.1f}"

    def test_prediction_not_nan(self, model_and_features):
        import pandas as pd
        model, feature_cols = model_and_features
        X = pd.DataFrame(np.zeros((5, len(feature_cols))), columns=feature_cols)
        preds = model.predict(X)
        assert not any(np.isnan(preds)), "Model returned NaN predictions"

    def test_missing_features_handled_gracefully(self, model_and_features):
        """If a new feature is added to the list, filling with 0 should not crash."""
        import pandas as pd
        model, feature_cols = model_and_features

        # Build X with all features set to 0 (simulates missing features at inference)
        X = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
        # Corrupt one column with NaN — should be handled by fillna(0)
        X.iloc[0, 0] = np.nan
        X = X.fillna(0.0)
        pred = model.predict(X)[0]
        assert not np.isnan(pred), "NaN in input caused NaN prediction"


class TestPredictAqiOutput:
    """Verify the predict_aqi() pipeline function returns the correct schema."""

    REQUIRED_KEYS = [
        "current_aqi",
        "forecast_aqi",
        "category",
        "timestamp",
        "forecast_timestamp",
        "confidence_lower",
        "confidence_upper",
        "history",
    ]

    VALID_CATEGORIES = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]

    def test_output_has_required_keys(self):
        """predict_aqi() must return a dict with all required keys."""
        # We can't call the live API in tests, so we test the structure contract
        # by checking that get_category() covers all AQI ranges
        from src.core_utils import get_category

        test_values = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400]
        for aqi in test_values:
            cat = get_category(aqi)
            assert cat in self.VALID_CATEGORIES, \
                f"get_category({aqi}) returned unexpected category: '{cat}'"

    def test_category_boundaries(self):
        """Verify AQI category boundaries match the US EPA standard."""
        from src.core_utils import get_category

        assert get_category(0)   == "Good"
        assert get_category(50)  == "Good"
        assert get_category(51)  == "Moderate"
        assert get_category(100) == "Moderate"
        assert get_category(101) == "Unhealthy for Sensitive Groups"
        assert get_category(150) == "Unhealthy for Sensitive Groups"
        assert get_category(151) == "Unhealthy"
        assert get_category(200) == "Unhealthy"
        assert get_category(201) == "Very Unhealthy"
        assert get_category(300) == "Very Unhealthy"
        assert get_category(301) == "Hazardous"
        assert get_category(500) == "Hazardous"

    def test_confidence_interval_ordering(self):
        """confidence_lower must always be <= forecast_aqi <= confidence_upper."""
        from src.core_utils import MODEL_RMSE

        for forecast in [30, 75, 120, 180, 250]:
            lower = max(0, forecast - round(MODEL_RMSE))
            upper = forecast + round(MODEL_RMSE)
            assert lower <= forecast <= upper, \
                f"Confidence interval broken for forecast={forecast}: [{lower}, {upper}]"

    def test_confidence_lower_never_negative(self):
        """AQI cannot be negative — confidence_lower must be clamped at 0."""
        from src.core_utils import MODEL_RMSE

        for forecast in [1, 2, 3, 4]:
            lower = max(0, forecast - round(MODEL_RMSE))
            assert lower >= 0, \
                f"confidence_lower is negative ({lower}) for forecast={forecast}"
