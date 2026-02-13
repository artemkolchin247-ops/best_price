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
    assert "price_before" in results.columns
    assert "price_before_spp" in results.columns
    assert "predicted_sales" in results.columns
    assert "profit" in results.columns
    assert (results["price_before"] == results["price_before_spp"]).all()
    assert best["best_price_before"] in results["price_before_spp"].values
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


class DummyForecasterPeak:
    def __init__(self, peak_price):
        self.peak_price = float(peak_price)

    def get_info(self):
        return {"stability_mode": "S1", "monotonicity_flag": "monotone", "protective_mode": None}

    def predict_sales(self, price, base_features):
        # Пик спроса около peak_price, чтобы управлять положением оптимума
        x = float(price)
        return max(1.0, 500.0 - ((x - self.peak_price) ** 2) / 5.0)

    def calibrate_curve(self, prices, preds):
        return preds


def test_optimize_price_boundary_history_true_search_false():
    results, best = optimize_price(
        forecaster=DummyForecasterPeak(peak_price=880.0),
        base_features={},
        price_min=1000,
        price_max=1200,
        step=100,
        commission_rate=0.01,
        vat_rate=0.01,
        spp=0.2,  # customer prices: 800, 880, 960 -> peak in middle
        cogs=0.0,
        logistics=0.0,
        storage=0.0,
        hist_min_before=1050.0,
        hist_max_before=1700.0,
    )

    assert not results.empty
    assert best["is_boundary_search"] is False
    assert best["is_boundary_history"] is True


def test_optimize_price_boundary_both_true():
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
        hist_min_before=600.0,
        hist_max_before=1450.0,
    )

    assert not results.empty
    assert best["is_boundary_search"] is True
    assert best["is_boundary_history"] is True


def test_optimize_price_boundary_both_false():
    results, best = optimize_price(
        forecaster=DummyForecasterPeak(peak_price=880.0),
        base_features={},
        price_min=1000,
        price_max=1200,
        step=100,
        commission_rate=0.01,
        vat_rate=0.01,
        spp=0.2,
        cogs=0.0,
        logistics=0.0,
        storage=0.0,
        hist_min_before=500.0,
        hist_max_before=1700.0,
    )

    assert not results.empty
    assert best["is_boundary_search"] is False
    assert best["is_boundary_history"] is False


def test_optimize_price_hist_before_fallback_from_after_prices():
    results, best = optimize_price(
        forecaster=DummyForecaster(),
        base_features={},
        price_min=1000,
        price_max=1200,
        step=100,
        commission_rate=0.1,
        vat_rate=0.05,
        spp=0.2,
        cogs=700.0,
        logistics=20.0,
        storage=10.0,
        hist_min=800.0,
        hist_max=1200.0,
    )

    assert not results.empty
    meta = best["boundary_meta"]
    assert meta["hist_min_before"] == 1000.0
    assert meta["hist_max_before"] == 1500.0
