import pandas as pd
from src.fincore.calculations import apply_fin_calculations


def main():
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
            },
            {
                "date": "2026-01-02",
                "sku": "SKU1",
                "orders": 8,
                "price_before_spp": 980.0,
                "price_after_spp": 931.0,
                "cogs": 600.0,
                "logistics": 50.0,
                "storage": 10.0,
            },
        ]
    )

    commission_rate = 0.12
    vat_rate = 0.20

    out = apply_fin_calculations(df, commission_rate, vat_rate)
    print("Before:")
    print(df[["sku", "orders", "price_before_spp", "price_after_spp"]])
    print("After:")
    print(out[["sku", "orders", "price_before_spp", "commission_per_unit", "vat_per_unit", "margin_per_unit", "profit"]])


if __name__ == "__main__":
    main()
