import pandas as pd

from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price


def test_optimize_price_basic():
    df = pd.DataFrame(
        [
            {"date": f"2026-01-{i:02d}", "orders": 8 + (i % 4), "price_before_spp": 950.0 + i * 15, "price_after_spp": 900.0 + i * 10}
            for i in range(1, 16)
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
    assert "price_before_spp" in results.columns
    assert "predicted_sales" in results.columns
    assert "profit" in results.columns
    assert best["best_price_before_spp"] in results["price_before_spp"].values
    assert "is_boundary_search" in best
    assert "is_boundary_history" in best
    assert "boundary_meta" in best
    assert best["is_boundary"] == best["is_boundary_search"]


class DummyForecaster:
    def get_info(self):
        return {"stability_mode": "S1", "monotonicity_flag": "monotone", "protective_mode": None}

    def predict_sales(self, price, base_features):
        # убывающий спрос: максимум прибыли будет на нижней границе сетки
        return max(1.0, 2000.0 / float(price))

    def calibrate_curve(self, prices, preds):
        return preds


def test_optimize_price_two_level_boundary_flags():
    results, best = optimize_price(
        forecaster=DummyForecaster(),
        base_features={},
        price_min=1000,
        price_max=1400,
        step=100,
        commission_rate=0.1,
        vat_rate=0.05,
        spp=0.2,
        cogs=700.0,
        logistics=20.0,
        storage=10.0,
        hist_min_before=900.0,
        hist_max_before=1700.0,
    )

    assert not results.empty
    assert best["is_boundary_search"] is True
    assert best["is_boundary_history"] is False
    meta = best["boundary_meta"]
    assert meta["search_min"] == 1000.0
    assert meta["search_max"] == 1400.0
    assert meta["hist_min_before"] == 900.0
    assert meta["hist_max_before"] == 1700.0
