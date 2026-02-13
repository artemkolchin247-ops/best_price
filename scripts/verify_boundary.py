import pandas as pd
import numpy as np
from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price

def test_boundary_optimum():
    # Создаем данные
    data = {
        "date": pd.date_range("2024-01-01", periods=10),
        "price_before_spp": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        "price_after_spp": [90, 99, 108, 117, 126, 135, 144, 153, 162, 171],
        "orders": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(data)
    
    sf = SalesForecaster()
    sf.fit(df)
    
    # Режим будет Режим 1 (Стабильный), так как данных мало
    # Чтобы получить граничный оптимум, выберем диапазон [200, 300], где спрос будет 0, но маржа растет
    # Или просто диапазон [50, 80], где спрос максимален на краю.
    
    # Вариант 1: Оптимум на левом краю (минимальная цена)
    results_left, best_left = optimize_price(
        forecaster=sf,
        base_features={},
        price_min=50,
        price_max=80,
        step=5,
        commission_rate=0.1,
        vat_rate=0.2,
        cogs=10,
        logistics=5,
        storage=1,
        spp=0.1,
        sku_df=df
    )
    print(f"Best Price Left: {best_left['best_price_before']}, Is Boundary: {best_left['is_boundary']}")

    # Вариант 2: Оптимум на правом краю (максимальная цена - если маржа перекрывает падение спроса)
    # Сделаем COGS очень маленьким, чтобы высокая цена была выгоднее
    results_right, best_right = optimize_price(
        forecaster=sf,
        base_features={},
        price_min=200,
        price_max=250,
        step=5,
        commission_rate=0.1,
        vat_rate=0.2,
        cogs=0,
        logistics=0,
        storage=0,
        spp=0.1,
        sku_df=df
    )
    print(f"Best Price Right: {best_right['best_price_before']}, Is Boundary: {best_right['is_boundary']}")

    if best_left['is_boundary'] and best_right['is_boundary']:
        print("SUCCESS: Boundary optimum detected on both sides.")
    else:
        print("FAILURE: Boundary optimum not detected correctly.")

if __name__ == "__main__":
    test_boundary_optimum()
