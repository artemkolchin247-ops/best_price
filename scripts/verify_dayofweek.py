import pandas as pd
import numpy as np
from src.models.sales_forecast import SalesForecaster

def test_day_of_week_influence():
    # Создаем данные с явной недельной сезонностью
    # Понедельники (0) - низкие продажи, Субботы (5) - высокие
    dates = pd.date_range("2024-01-01", periods=60)
    prices = [100] * 60
    orders = []
    for d in dates:
        if d.weekday() == 5: # Суббота
            orders.append(20)
        elif d.weekday() == 0: # Понедельник
            orders.append(5)
        else:
            orders.append(10)
            
    df = pd.DataFrame({
        "date": dates,
        "price_after_spp": prices,
        "orders": orders,
        "sku": "test_sku"
    })
    
    sf = SalesForecaster()
    sf.fit(df)
    
    # Проверяем прогноз для понедельника (0) и субботы (5) при одинаковой цене
    q_mon = sf.predict_sales(100, {"day_of_week": 0})
    q_sat = sf.predict_sales(100, {"day_of_week": 5})
    
    print(f"Predicted Mon: {q_mon:.1f}")
    print(f"Predicted Sat: {q_sat:.1f}")
    
    if q_sat > q_mon:
        print("SUCCESS: Day of week influence detected (Saturday > Monday).")
    else:
        print("FAILURE: Model did not capture day of week seasonality.")

if __name__ == "__main__":
    test_day_of_week_influence()
