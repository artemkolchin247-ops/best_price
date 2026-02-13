import pandas as pd
import numpy as np
from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price

def test_s1_regime():
    # Создаем "плохие" данные (почти нет вариации цены)
    data = {
        "date": pd.date_range("2024-01-01", periods=40),
        "price_before_spp": [1000] * 40,
        "price_after_spp": [900] * 40,
        "orders": [10] * 40
    }
    df = pd.DataFrame(data)
    
    sf = SalesForecaster()
    # Обучаем. Так как цена константа, data_ok будет False -> Режим S1
    sf.fit(df)
    
    info = sf.get_info()
    print(f"Protective Mode: {info.get('protective_mode')}")
    
    # Запускаем оптимизацию
    results, best_info = optimize_price(
        forecaster=sf,
        base_features={},
        price_min=500,
        price_max=1500,
        step=10,
        commission_rate=0.1,
        vat_rate=0.2,
        cogs=500,
        logistics=50,
        storage=10,
        spp=0.1,
        sku_df=df
    )
    
    print(f"Final Regime: {best_info['regime']}")
    if "S1" in best_info["regime"]:
        print("SUCCESS: S1 regime detected correctly.")
    else:
        print("FAILURE: S1 regime not detected.")

if __name__ == "__main__":
    test_s1_regime()
