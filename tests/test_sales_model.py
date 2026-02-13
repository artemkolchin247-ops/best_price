import pandas as pd

from src.models.sales_forecast import SalesForecaster


def test_sales_forecaster_basic(tmp_path):
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "orders": 10, "price_before_spp": 1000.0, "price_after_spp": 950.0},
            {"date": "2026-01-02", "orders": 12, "price_before_spp": 980.0, "price_after_spp": 931.0},
            {"date": "2026-01-03", "orders": 9, "price_before_spp": 1020.0, "price_after_spp": 969.0},
            {"date": "2026-01-04", "orders": 11, "price_before_spp": 1005.0, "price_after_spp": 954.0},
            {"date": "2026-01-05", "orders": 8, "price_before_spp": 1050.0, "price_after_spp": 997.0},
        ]
    )

    sf = SalesForecaster(feature_cols=["price_after_spp"], time_col="date")
    best = sf.fit(df, n_splits=2)
    assert best in ("loglog", "rf")
    pred = sf.predict_sales(990)
    assert isinstance(pred, float)
    assert pred >= 0.0
