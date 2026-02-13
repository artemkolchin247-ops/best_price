import pandas as pd

from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price


def test_optimize_price_basic():
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
    sf.fit(df, n_splits=2)

    results, best = optimize_price(
        forecaster=sf,
        base_features={},
        price_min=900,
        price_max=1100,
        step=50,
        commission_rate=0.12,
        vat_rate=0.20,
        spp=0.05,
        cogs=600.0,
        logistics=50.0,
        storage=10.0,
    )

    # basic asserts
    assert "price_before" in results.columns
    assert "predicted_sales" in results.columns
    assert "profit" in results.columns
    assert best["best_price_before"] in results["price_before"].values
