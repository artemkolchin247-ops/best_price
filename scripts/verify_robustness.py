import pandas as pd
import numpy as np
from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price

def test_S1_insufficient_data():
    # Мало данных (10 дней) -> должен включиться S1
    np.random.seed(42)
    days = 10
    prices = np.random.uniform(1000, 2000, size=days)
    orders = np.random.poisson(5, size=days)
    
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=days),
        "sku": "SKU_S1_DATA",
        "price_after_spp": prices,
        "price_before_spp": prices,
        "orders": orders
    })
    
    sf = SalesForecaster(feature_cols=["price_after_spp"])
    sf.fit(df)
    info = sf.get_info()
    
    print("--- Тест S1 (Мало данных) ---")
    print(f"data_ok: {info['quality']['data_ok']}")
    print(f"protective_mode: {info['protective_mode']}")
    assert info['protective_mode'] == "S1"

def test_S1_useless_model():
    # Данных много, но они шумные -> S1
    np.random.seed(42)
    days = 100
    prices = np.random.uniform(1000, 2000, size=days)
    orders = np.random.poisson(10, size=days)
    
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=days),
        "sku": "SKU_S1_MODEL",
        "price_after_spp": prices,
        "price_before_spp": prices,
        "orders": orders
    })
    
    sf = SalesForecaster(feature_cols=["price_after_spp"])
    sf.fit(df)
    info = sf.get_info()
    
    print("\n--- Тест S1 (Бесполезная модель) ---")
    print(f"protective_mode: {info['protective_mode']}")
    assert info['protective_mode'] == "S1"

def test_S2_non_monotonic():
    # Данных много, но кривая "горбатая" (немонотонная) -> S2
    # Создаем искуственную немонотонность: рост спроса при росте цены
    np.random.seed(42)
    days = 100
    prices = np.linspace(1000, 2000, days)
    # Спрос РЕЗКО РАСТЕТ с ценой (аномалия)
    orders = 5 + 0.1 * prices + np.random.normal(0, 5, size=days)
    orders = np.clip(orders, 1, 1000).astype(int)
    
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=days),
        "sku": "SKU_S2",
        "price_after_spp": prices,
        "price_before_spp": prices,
        "orders": orders
    })
    
    sf = SalesForecaster(feature_cols=["price_after_spp"])
    sf.fit(df) # RandomForest легко выучит рост
    info = sf.get_info()
    
    print("\n--- Тест S2 (Немонотонная кривая) ---")
    print(f"data_ok: {info['quality']['data_ok']}")
    print(f"mono_violations: {info['policy'].get('mono_violations', 0):.2f}")
    print(f"protective_mode: {info['protective_mode']}")
    # S2 включится если data_ok=True и нарушения > 20%
    assert info['protective_mode'] == "S2"

if __name__ == "__main__":
    test_S1_insufficient_data()
    test_S1_useless_model()
    test_S2_non_monotonic()
