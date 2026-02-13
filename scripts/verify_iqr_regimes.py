
import pandas as pd
import numpy as np
from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price

def test_regimes():
    # 1. Подготовка данных
    days = 100
    dates = pd.date_range("2024-01-01", periods=days)
    prices = np.random.uniform(1000, 2000, size=days)
    orders = 100 - 0.04 * prices + np.random.normal(0, 5, size=days)
    df = pd.DataFrame({
        "date": dates,
        "price_before_spp": prices,
        "price_after_spp": prices * 0.9,
        "orders": np.clip(orders, 1, 200).astype(int)
    })

    sf = SalesForecaster(feature_cols=["price_after_spp"])
    sf.fit(df)

    # Заменяем информацию об эластичности вручную для имитации разных IQR
    def run_with_iqr(iqr_val, current_p=1500):
        sf.elasticity_info["beta_iqr"] = iqr_val
        # Имитируем sku_df с текущей ценой
        test_sku_df = df.copy()
        test_sku_df.loc[test_sku_df.index[-1], "price_before_spp"] = current_p
        test_sku_df.loc[test_sku_df.index[-1], "price_after_spp"] = current_p * 0.9 # Текущая цена в тесте всегда с СПП 10%%
        
        res, best = optimize_price(
            forecaster=sf,
            base_features={},
            price_min=500,
            price_max=3000,
            step=10,
            commission_rate=0.1,
            vat_rate=0.2,
            spp=0.1,
            cogs=500,
            logistics=50,
            storage=10,
            hist_min=1000,
            hist_max=2000,
            sku_df=test_sku_df
        )
        print(f"\n--- IQR: {iqr_val} ---")
        print(f"Regime: {best['regime']}")
        print(f"Price Range in results: {res['price_before_spp'].min()} - {res['price_before_spp'].max()}")
        print(f"Best Price: {best['best_price_before_spp']}")
        
    # Test Regime 1 (Stable)
    run_with_iqr(0.2)
    
    # Test Regime 2 (Moderate)
    run_with_iqr(0.5)
    
    # Test Regime 3 (Unstable)
    run_with_iqr(0.8, current_p=1500)

if __name__ == "__main__":
    test_regimes()
