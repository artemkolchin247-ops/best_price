import pandas as pd

from src.models.sales_forecast import SalesForecaster


def test_sales_forecaster_basic(tmp_path):
    df = pd.DataFrame(
        [
            {"date": f"2026-01-{i:02d}", "orders": 8 + (i % 5), "price_before_spp": 900.0 + i * 15, "price_after_spp": 850.0 + i * 12}
            for i in range(1, 16)
        ]
    )

    sf = SalesForecaster(feature_cols=["price_after_spp"], time_col="date")
    best = sf.fit(df, n_splits=2)
    assert best in ("loglog", "rf", "poisson")
    pred = sf.predict_sales(990)
    assert isinstance(pred, float)
    assert pred >= 0.0


def test_local_elasticity_excludes_invalid_points_and_reports_debug_metrics():
    df = pd.DataFrame(
        [
            {"date": f"2026-01-{i:02d}", "orders": 10 + (i % 3), "price_before_spp": 1000.0 + i, "price_after_spp": 900.0 + i}
            for i in range(1, 21)
        ]
    )

    sf = SalesForecaster(feature_cols=["price_after_spp"], time_col="date")

    def patched_predict(price, features_row=None):
        if 905 <= float(price) <= 910:
            return float("nan")
        return max(1.0, 1000.0 / float(price))

    sf.predict_sales = patched_predict
    elasticity = sf._calculate_numerical_elasticity(df)

    e_stats = elasticity["e_stats"]
    assert e_stats["excluded_invalid_points"] > 0
    assert e_stats["total_points"] == e_stats["valid_points"] + e_stats["excluded_invalid_points"]
    assert elasticity["local_elasticity_invalid_points"] == e_stats["excluded_invalid_points"]
    assert 0.0 <= e_stats["zero_share"] <= 1.0
