"""Brute-force price optimizer.

Iterates prices in range, predicts sales via provided forecaster, computes
unit economics and returns table with results and best price by profit.
"""
from typing import Dict, Any, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

from src.models.sales_forecast import SalesForecaster


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

    Returns:
        (results_df, best_info)
    """
    """Brute-force optimization over price range [price_min, price_max]."""
    f_info = forecaster.get_info()
    stability = f_info.get("stability_mode", "S1")
    mono_flag = f_info.get("monotonicity_flag", "monotone")
    protective = f_info.get("protective_mode")
    
    # Режим оптимизации по умолчанию
    regime = stability
    penalty_enabled = False
    
    # Определяем текущую цену ПОСЛЕ СПП и текущую ПРИБЫЛЬ для расчетов
    current_p_after = None
    current_profit_daily = 0.0
    
    # Квантили истории для режимов (по price_after_spp)
    p5, p10, p90, p95 = None, None, None, None
    
    if sku_df is not None and not sku_df.empty:
        last_row_sku = sku_df.sort_values("date").iloc[-1]
        current_p_after = float(last_row_sku["price_after_spp"])
        # Считаем текущую прибыль (прогнозную) по ТЗ: UnitMargin * Orders
        p_last_before = float(last_row_sku["price_before_spp"])
        comm_last = p_last_before * commission_rate
        vat_last = current_p_after * vat_rate
        m_last = p_last_before - comm_last - vat_last - cogs - logistics - storage
        q_last = float(forecaster.predict_sales(current_p_after, base_features))
        current_profit_daily = m_last * q_last
        
        # Расчет квантилей
        p_after_hist = sku_df["price_after_spp"]
        p5, p10, p90, p95 = p_after_hist.quantile([0.05, 0.1, 0.9, 0.95])

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
    
    if protective == "S1" and sku_df is not None:
        # Перетираем калиброванные прогнозы константой для режима S1
        last_orders = sku_df.sort_values("date").tail(14)["orders"].median()
        calibrated_preds = np.full_like(calibrated_preds, last_orders)

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
