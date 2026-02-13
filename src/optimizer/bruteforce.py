"""Brute-force price optimizer.

Iterates prices in range, predicts sales via provided forecaster, computes
unit economics and returns table with results and best price by profit.
"""
print("DEBUG: bruteforce.py module loading...")

from typing import Dict, Any, Tuple, Iterable, Optional
print("DEBUG: typing import successful")

import numpy as np
print("DEBUG: numpy import successful")

import pandas as pd
print("DEBUG: pandas import successful")

try:
    from src.models.sales_forecast import SalesForecaster
    print("DEBUG: SalesForecaster import successful")
except ImportError as e:
    print(f"ERROR: Cannot import SalesForecaster: {e}")
    raise


def optimize_price(
    forecaster: SalesForecaster,
    base_features: Dict[str, Any],
    price_min: float,
    price_max: float,
    step: float,
    commission_rate: float,
    vat_rate: float,
    spp: float,
    cogs: float,
    logistics: float,
    storage: float,
    hist_min: Optional[float] = None,
    hist_max: Optional[float] = None,
    hist_min_before: Optional[float] = None,
    hist_max_before: Optional[float] = None,
    sku_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Brute-force search best price.

    Args:
        forecaster: trained SalesForecaster (fit called).
        base_features: dict of additional features (category medians etc.).
        price_min/price_max/step: range for price_before_spp.
        commission_rate, vat_rate, spp: market params (fractions).
        cogs, logistics, storage: per-unit costs (numbers).
        hist_min/hist_max: historical price range for reference.
        hist_min_before/hist_max_before: historical price_before_spp range.
        sku_df: DataFrame with historical data for additional calculations.
    """
    # Валидация входных параметров
    print(f"DEBUG: optimize_price called with forecaster type: {type(forecaster)}")
    print(f"DEBUG: base_features type: {type(base_features)}")
    print(f"DEBUG: sku_df type: {type(sku_df)}")
    print(f"DEBUG: sku_df is None: {sku_df is None}")
    
    if sku_df is not None:
        print(f"DEBUG: sku_df empty: {sku_df.empty}")
        print(f"DEBUG: sku_df columns: {list(sku_df.columns) if hasattr(sku_df, 'columns') else 'No columns attr'}")
    
    if not isinstance(forecaster, SalesForecaster):
        raise TypeError(f"forecaster must be SalesForecaster, got {type(forecaster)}")
    
    if not isinstance(base_features, dict):
        raise TypeError(f"base_features must be dict, got {type(base_features)}")
    
    if sku_df is not None and not isinstance(sku_df, pd.DataFrame):
        raise TypeError(f"sku_df must be pandas.DataFrame or None, got {type(sku_df)}")
    
    # Проверка что forecaster обучен
    if forecaster.best_model_name is None:
        raise RuntimeError("Forecaster must be trained before optimization")
    
    print("DEBUG: Basic validation passed")
    
    # Получаем информацию о модели
    try:
        f_info = forecaster.get_info()
        print(f"DEBUG: Got forecaster info: {type(f_info)}")
    except Exception as e:
        print(f"DEBUG: Error getting forecaster info: {e}")
        raise
    
    stability = f_info.get("stability_mode", "S1")
    mono_flag = f_info.get("monotonicity_flag", "monotone")
    protective = f_info.get("protective_mode")
    
    print(f"DEBUG: stability={stability}, mono_flag={mono_flag}, protective={protective}")
    
    # Режим оптимизации по умолчанию
    regime = stability
    penalty_enabled = False
    
    # Определяем текущую цену ПОСЛЕ СПП и текущую ПРИБЫЛЬ для расчетов
    current_p_after = None
    current_profit_daily = 0.0
    
    # Квантили истории для режимов (по price_after_spp)
    p5, p10, p90, p95 = None, None, None, None
    
    print("DEBUG: Starting sku_df processing")
    
    if sku_df is not None and not sku_df.empty and "price_after_spp" in sku_df.columns:
        try:
            print("DEBUG: Processing sku_df for current price")
            last_row_sku = sku_df.sort_values("date").iloc[-1]
            print(f"DEBUG: Got last row, type: {type(last_row_sku)}")
            
            current_p_after = float(last_row_sku["price_after_spp"])
            print(f"DEBUG: Got current_p_after: {current_p_after}")
            
            # Считаем текущую прибыль (прогнозную) по ТЗ: UnitMargin * Orders
            p_last_before = float(last_row_sku["price_before_spp"])
            print(f"DEBUG: Got p_last_before: {p_last_before}")
            
            s_val = spp
            p_after_last = p_last_before * (1.0 - s_val)
            comm_last = p_last_before * commission_rate
            vat_last = p_after_last * vat_rate
            m_last = p_last_before - comm_last - vat_last - cogs - logistics - storage
            print(f"DEBUG: Calculated margin: {m_last}")
            
            # Прогноз для текущей цены
            print("DEBUG: Calling predict_sales")
            q_last = forecaster.predict_sales(current_p_after, base_features)
            print(f"DEBUG: Got q_last: {q_last}")
            
            current_profit_daily = m_last * q_last
            print(f"DEBUG: Calculated current_profit_daily: {current_profit_daily}")
            
            # Расчет квантилей
            print("DEBUG: Calculating quantiles")
            p_after_hist = sku_df["price_after_spp"]
            p5, p10, p90, p95 = p_after_hist.quantile([0.05, 0.1, 0.9, 0.95])
            print(f"DEBUG: Got quantiles: p5={p5}, p10={p10}, p90={p90}, p95={p95}")
            
        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(f"Warning: Error processing sku_df: {e}")
            # Устанавливаем значения по умолчанию
            current_p_after = None
            current_profit_daily = None
    else:
        print("DEBUG: sku_df is None, empty, or missing price_after_spp column")

    # --- Определение режима на основе стабильности и монотонности ---
    reg_min_a, reg_max_a = 0, float("inf")
    
    if protective == "S1":
        regime = "Защитный S1 (Критически мало данных)"
    elif stability == "S3":
        regime = "S3 (Нестабильный)"
        penalty_enabled = True
        # Ограничение диапазона: intersection([current * 0.8; current * 1.2], [p10; p90])
        low_a, high_a = 0, float("inf")
        if current_p_after is not None:
            low_a, high_a = current_p_after * 0.8, current_p_after * 1.2
        
        if p10 is not None: low_a = max(low_a, p10)
        if p90 is not None: high_a = min(high_a, p90)
        reg_min_a, reg_max_a = low_a, high_a
    elif stability == "S2":
        regime = "S2 (Умеренный)"
        # Оптимизация внутри [p10; p90]
        if p10 is not None: reg_min_a = max(reg_min_a, p10)
        if p90 is not None: reg_max_a = min(reg_max_a, p90)
    else:  # S1 stability - Stable
        regime = "S1 (Стабильный)"
        # Глобальная оптимизация внутри [p5; p95]
        if p5 is not None: reg_min_a = max(reg_min_a, p5)
        if p95 is not None: reg_max_a = min(reg_max_a, p95)

    # Применяем итоговые ограничения на сетку price_before_spp
    price_min = max(price_min, reg_min_a / (1.0 - spp))
    price_max = min(price_max, reg_max_a / (1.0 - spp))

    prices = np.arange(price_min, price_max + step, step)
    results = []

    # Исторические границы в цене ДО СПП (для второго уровня boundary-флага)
    hist_min_before_effective = hist_min_before
    hist_max_before_effective = hist_max_before
    spp_factor = (1.0 - spp)
    if spp_factor > 0:
        if hist_min_before_effective is None and hist_min is not None:
            hist_min_before_effective = float(hist_min) / spp_factor
        if hist_max_before_effective is None and hist_max is not None:
            hist_max_before_effective = float(hist_max) / spp_factor

    # Предварительный расчет спроса для всей сетки
    raw_preds = []
    customer_prices = []
    
    for p in prices:
        p_after = p * (1.0 - spp)
        q = forecaster.predict_sales(p_after, base_features)
        raw_preds.append(q)
        customer_prices.append(p_after)
    
    # Калибровка всей кривой разом при немонотонности (ТЗ 8)
    # Используем ту же логику, что и в _calculate_numerical_elasticity
    if mono_flag == "non_monotone":
        calibrated_preds = forecaster.calibrate_curve(np.array(customer_prices), np.array(raw_preds))
    else:
        calibrated_preds = np.array(raw_preds)
    
    if protective == "S1" and sku_df is not None and not sku_df.empty and "orders" in sku_df.columns:
        # Перетираем калиброванные прогнозы константой для режима S1
        try:
            last_orders = sku_df.sort_values("date").tail(14)["orders"].median()
            calibrated_preds = np.full_like(calibrated_preds, last_orders)
        except (KeyError, IndexError, ValueError) as e:
            print(f"Warning: Error processing S1 regime: {e}")
            # Оставляем calibrated_preds как есть

    for i, p in enumerate(prices):
        p_after = customer_prices[i]
        predicted_q = float(calibrated_preds[i])

        # Total Ad Spend (Daily)
        ad_internal = base_features.get("ad_internal", 0.0)
        ad_bloggers = base_features.get("ad_bloggers", 0.0)
        ad_vk = base_features.get("ad_vk", 0.0)
        total_ad_spend = ad_internal + ad_bloggers + ad_vk

        # Financials
        commission = p * commission_rate
        vat = p_after * vat_rate
        margin_unit = p - commission - vat - cogs - logistics - storage
        profit = (margin_unit * predicted_q) - total_ad_spend

        # Check extrapolation risk
        is_extra = False
        if hist_min is not None and p_after < hist_min:
            is_extra = True
        if hist_max is not None and p_after > hist_max:
            is_extra = True
            
        # Penalty for dangerous extrapolation if not in protective modes
        if is_extra and "Режим 1" in regime:
            profit = -1e9
        
        # Penalty for deviation in Regime 3
        actual_profit = profit # Сохраняем чистую прибыль для лога
        if penalty_enabled and current_p_after:
            # Штраф: k * profit_current * delta_pct (k=0.5)
            delta_pct = abs(p_after - current_p_after) / current_p_after
            penalty = 0.5 * current_profit_daily * delta_pct
            profit -= penalty # Теперь profit это AdjustedProfit

        results.append(
            {
                # Backward-compatible alias for external consumers/tests
                "price_before": p,
                "price_before_spp": p,
                "price_after_spp": p_after,
                "predicted_sales": predicted_q,
                "margin_unit": margin_unit,
                "profit": profit,
                "is_extrapolated": is_extra,
            }
        )

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res, {}

    best_idx = df_res["profit"].idxmax()
    best_row = df_res.loc[best_idx]

    is_boundary_search = bool(best_idx == 0 or best_idx == len(df_res) - 1)

    # Двухуровневый boundary-флаг: 2) близость к историческим границам
    is_boundary_history = False
    tol = float(step)
    if hist_min_before_effective is not None and hist_max_before_effective is not None:
        hist_span = float(hist_max_before_effective - hist_min_before_effective)
        if hist_span > 0:
            tol = max(float(step), hist_span * 0.02)

        p_opt_before = float(best_row["price_before_spp"])
        is_boundary_history = (
            abs(p_opt_before - float(hist_min_before_effective)) <= tol
            or abs(p_opt_before - float(hist_max_before_effective)) <= tol
        )

    best_info = {
        "best_price_before": best_row["price_before_spp"],
        "best_customer_price": best_row["price_after_spp"],
        "best_profit": best_row["profit"],
        "best_sales": best_row["predicted_sales"],
        "best_margin": best_row["margin_unit"],
        "is_extrapolated": best_row["is_extrapolated"],
        "is_boundary_search": is_boundary_search,
        "is_boundary_history": is_boundary_history,
        "is_boundary": is_boundary_search,  # backward compatibility alias
        "boundary_meta": {
            "search_min": float(df_res["price_before_spp"].min()),
            "search_max": float(df_res["price_before_spp"].max()),
            "hist_min_before": float(hist_min_before_effective) if hist_min_before_effective is not None else None,
            "hist_max_before": float(hist_max_before_effective) if hist_max_before_effective is not None else None,
            "tol": float(tol)
        },
        "regime": regime
    }

    return df_res, best_info


__all__ = ["optimize_price"]
