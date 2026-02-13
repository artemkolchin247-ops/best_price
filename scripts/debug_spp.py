import pandas as pd
import numpy as np
import sys
import os

# Добавляем корень проекта в пути
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price

# 1. Создаем синтетические данные
prices_before = np.linspace(1000, 2000, 20)
spps_hist = np.random.uniform(0.05, 0.20, 20)
prices_after = prices_before * (1 - spps_hist)
orders = 1000000 / (prices_after**1.5) + np.random.normal(0, 1, 20) # Эластичность -1.5

df = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=20),
    "price_after_spp": prices_after,
    "orders": orders,
    "sku": "TEST_SKU"
})

# 2. Обучаем модель
sf = SalesForecaster(feature_cols=["price_after_spp"], time_col="date")
sf.fit(df)

# 3. Тестируем оптимизацию при разных СПП
params = {
    "price_min": 500,
    "price_max": 5000,
    "step": 1.0,
    "commission_rate": 0.12,
    "vat_rate": 0.20,
    "cogs": 500,
    "logistics": 50,
    "storage": 10,
}

print(f"Model trained. Best model: {sf.best_model_name}")

for test_spp in [0.0, 0.1, 0.3, 0.5]:
    res, best = optimize_price(sf, {}, spp=test_spp, **params)
    print(f"SPP: {test_spp*100:.0f}% | Best Price Before: {best['best_price_before']:.2f} | Best Profit: {best['best_profit']:.2f}")
