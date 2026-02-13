import pandas as pd
import numpy as np
from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price

def test_spp_sensitivity():
    # 1. Синтетические данные с четкой эластичностью
    prices = np.array([1000, 1100, 1200, 1300, 1400, 1500])
    # Спрос падает при росте цены
    orders = np.array([100, 80, 60, 45, 30, 20])
    
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(prices)),
        "price_after_spp": prices,
        "price_before_spp": prices * 1.05,
        "orders": orders
    })
    
    sf = SalesForecaster(feature_cols=["price_after_spp"])
    sf.fit(df)
    
    common_params = {
        "forecaster": sf,
        "base_features": {},
        "price_min": 1000,
        "price_max": 3000, # Увеличили, чтобы LP могла расти
        "step": 10,
        "commission_rate": 0.1,
        "vat_rate": 0.2,
        "cogs": 500,
        "logistics": 50,
        "storage": 10,
        "hist_min": 1000,
        "hist_max": 1500
    }
    
    print("--- Тест чувствительности СПП ---")
    results = []
    for spp in [0.0, 0.1, 0.3, 0.5]:
        _, best = optimize_price(spp=spp, **common_params)
        lp = best['best_price_before']
        cp = lp * (1 - spp)
        print(f"SPP: {spp*100:>2.0f}% | Оптим. ListPrice: {lp:>7.1f} | CustomerPrice: {cp:>7.1f}")
        results.append(lp)
    
    # Проверка: ListPrice должна расти при росте SPP
    if results[-1] > results[0]:
        print("\nУСПЕХ: Оптимальная цена до скидки растет при росте СПП.")
    elif results[-1] == results[0]:
        print("\nПРЕДУПРЕЖДЕНИЕ: Оптимальная цена не изменилась (возможно, из-за шага или границ).")
    else:
        print("\nОШИБКА: Оптимальная цена упала при росте СПП!")

if __name__ == "__main__":
    test_spp_sensitivity()
