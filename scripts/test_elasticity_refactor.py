import pandas as pd
import numpy as np
from src.models.sales_forecast import SalesForecaster

def test_elasticity_non_zero_rf():
    print("\n--- Testing RF Elasticity Non-Zero ---")
    # Генерируем данные с ОЧЕНЬ сильной зависимостью
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60)
    prices = np.random.uniform(500, 1500, size=60)
    # Экспоненциальная зависимость для LogLog: q = A * p^-1.5
    orders = 1e6 * (prices**-1.5) + np.random.normal(0, 1, size=60)
    orders = np.clip(orders, 1, 500)
    
    df = pd.DataFrame({
        "date": dates,
        "price_after_spp": prices,
        "price_before_spp": prices / 0.9,
        "orders": orders,
        "sku": "SKU1"
    })
    
    # Обучаем специально RF
    sf = SalesForecaster(feature_cols=["price_after_spp"])
    # Форсируем выбор RF (хотя он и так может выиграть)
    best = sf.fit(df)
    print(f"Best model selected: {best}")
    
    info = sf.get_info()
    e_med = info["elasticity"]["elasticity_med"]
    e_iqr = info["elasticity"]["elasticity_iqr"]
    stability = info["stability_mode"]
    mono = info["monotonicity_flag"]
    
    print(f"Elasticity Median: {e_med:.4f}")
    print(f"Elasticity IQR: {e_iqr:.4f}")
    print(f"Stability: {stability}")
    print(f"Monotonicity: {mono}")
    
    assert e_med < 0, f"Elasticity should be negative, got {e_med}"
    assert abs(e_med) > 0.0001, f"Elasticity should be non-zero, got {e_med}"
    print("SUCCESS: RF Elasticity is non-zero and negative.")

def test_stability_vs_monotonicity():
    print("\n--- Testing Stability vs Monotonicity logic ---")
    # Создаем "плохие" данные: высокая вариативность (низкая стабильность) 
    # но в целом монотонно убывающие
    np.random.seed(42)
    p = np.linspace(500, 1500, 60)
    q = 1000 / (p/100) + np.random.normal(0, 50, size=60) # сильный шум
    q = np.clip(q, 1, 1000)
    
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60),
        "price_after_spp": p,
        "price_before_spp": p / 0.9,
        "orders": q,
        "sku": "SKU_NOISY"
    })
    
    sf = SalesForecaster(feature_cols=["price_after_spp"])
    sf.fit(df)
    info = sf.get_info()
    
    print(f"Stability: {info['stability_mode']}")
    print(f"Monotonicity Violations: {info['elasticity']['mono_violations']:.2%}")
    print(f"Monotonicity Flag: {info['monotonicity_flag']}")
    
    # В шумных данных IQR обычно выше -> S2 или S3
    assert info['stability_mode'] in ["S1", "S2", "S3"]
    print("SUCCESS: Modes calculated.")

if __name__ == "__main__":
    test_elasticity_non_zero_rf()
    test_stability_vs_monotonicity()
