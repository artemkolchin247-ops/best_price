import pandas as pd
import numpy as np
from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price

def test_ad_spend_logic():
    # 1. Тест влияния на прогноз заказов
    # Увеличим объем данных для стабильного обучения моделей
    dates = pd.date_range("2024-01-01", periods=100)
    # Половина данных без рекламы (10 заказов), половина с рекламой (20 заказов)
    ad_vals = [0] * 50 + [1000] * 50
    order_vals = [10] * 50 + [20] * 50
    
    df = pd.DataFrame({
        "date": dates,
        "price_after_spp": [1000] * 100,
        "ad_internal": ad_vals,
        "ad_bloggers": [0] * 100,
        "ad_vk": [0] * 100,
        "orders": order_vals,
        "sku": "test_sku",
    })
    
    sf = SalesForecaster(feature_cols=["price_after_spp", "ad_internal", "ad_bloggers", "ad_vk"])
    sf.fit(df)
    
    print(f"Used model: {sf.get_info().get('model_name')}")
    
    q_no_ad = float(sf.predict_sales(1000, {"ad_internal": 0}))
    q_with_ad = float(sf.predict_sales(1000, {"ad_internal": 1000}))
    
    print(f"Orders (No Ad): {q_no_ad:.1f}")
    print(f"Orders (With Ad): {q_with_ad:.1f}")
    
    # 2. Тест вычета из прибыли
    res_df, best_info = optimize_price(
        forecaster=sf,
        base_features={"ad_internal": 1000, "ad_bloggers": 0, "ad_vk": 0},
        price_min=1000,
        price_max=1000,
        step=10,
        commission_rate=0.1,
        vat_rate=0,
        spp=0,
        cogs=500,
        logistics=0,
        storage=0
    )
    
    # price=1000, comm=100, cogs=500 -> margin=400. q=20 (из-за ad). profit = 20*400 - 1000 = 7000
    calculated_profit = best_info["best_profit"]
    print(f"Calculated Daily Profit: {calculated_profit:.1f}")
    
    if q_with_ad > q_no_ad and abs(calculated_profit - (400 * q_with_ad - 1000)) < 0.1:
        print("SUCCESS: Advertising logic verified.")
    else:
        print("FAILURE: Ad logic issues.")

if __name__ == "__main__":
    test_ad_spend_logic()
