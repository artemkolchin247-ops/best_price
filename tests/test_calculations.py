import pandas as pd
from src.fincore.calculations import (
    commission_per_unit,
    vat_per_unit,
    margin_per_unit,
    profit_total,
    apply_fin_calculations,
)


def test_commission_vat_margin_profit():
    p_before = pd.Series([1000.0, 500.0])
    p_after = pd.Series([950.0, 475.0])
    qty = pd.Series([10, 5])
    cogs = pd.Series([600.0, 200.0])
    logistics = pd.Series([50.0, 20.0])
    storage = pd.Series([10.0, 5.0])

    commission = commission_per_unit(p_before, 0.1)
    assert commission.iloc[0] == 100.0
    assert commission.iloc[1] == 50.0

    vat = vat_per_unit(p_after, 0.2)
    assert pytest_approx(vat.iloc[0], 190.0)

    margin = margin_per_unit(p_before, p_after, 0.1, 0.2, cogs, logistics, storage)
    # manual compute first row: 1000 - 100 - 190 - 600 - 50 -10 = 50
    assert pytest_approx(margin.iloc[0], 50.0)

    profit = profit_total(margin, qty)
    assert pytest_approx(profit.iloc[0], 500.0)


def pytest_approx(a, b, eps=1e-6):
    return abs(a - b) <= eps


def test_apply_fin_calculations_pipeline():
    df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "sku": "SKU1",
                "orders": 10,
                "price_before_spp": 1000.0,
                "price_after_spp": 950.0,
                "cogs": 600.0,
                "logistics": 50.0,
                "storage": 10.0,
            }
        ]
    )

    out = apply_fin_calculations(df, commission_rate=0.1, vat_rate=0.2)
    assert "commission_per_unit" in out.columns
    assert "vat_per_unit" in out.columns
    assert "margin_per_unit" in out.columns
    assert "profit" in out.columns
    assert pytest_approx(out.loc[0, "margin_per_unit"], 50.0)
    assert pytest_approx(out.loc[0, "profit"], 500.0)
