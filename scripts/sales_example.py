import pandas as pd
from src.models.sales_forecast import SalesForecaster


def main():
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "orders": 10, "price_before_spp": 1000.0},
            {"date": "2026-01-02", "orders": 12, "price_before_spp": 980.0},
            {"date": "2026-01-03", "orders": 9, "price_before_spp": 1020.0},
            {"date": "2026-01-04", "orders": 11, "price_before_spp": 1005.0},
            {"date": "2026-01-05", "orders": 8, "price_before_spp": 1050.0},
        ]
    )

    sf = SalesForecaster(feature_cols=["price_before_spp"], time_col="date")
    best = sf.fit(df, n_splits=2)
    print("Best model:", best)
    print("Predict sales at price 990:", sf.predict_sales(990))


if __name__ == "__main__":
    main()
