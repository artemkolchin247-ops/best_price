"""Brute-force price optimizer.

Iterates prices in range, predicts sales via provided forecaster, computes
unit economics and returns table with results and best price by profit.
"""
import logging

logger = logging.getLogger(__name__)
logger.debug("bruteforce.py module loading...")

from typing import Dict, Any, Tuple, Iterable, Optional
logger.debug("typing import successful")

import numpy as np
logger.debug("numpy import successful")

import pandas as pd
logger.debug("pandas import successful")

try:
    from src.models.sales_forecast import SalesForecaster
    logger.debug("SalesForecaster import successful")
except ImportError as e:
    logger.error("Cannot import SalesForecaster: %s", e)
    raise


def optimize_price(
    forecaster,  # Убрали жесткую типизацию
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
    **kwargs  # Добавляем для совместимости
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Brute-force search best price.

    Args:
        forecaster: trained forecaster with predict method (fit called).
        base_features: dict of additional features (category medians etc.).
        price_min/price_max/step: range for price_before_spp.
        commission_rate, vat_rate, spp: market params (fractions).
        cogs, logistics, storage: per-unit costs (numbers).
        hist_min/hist_max: historical price range for reference.
        hist_min_before/hist_max_before: historical price_before_spp range.
        sku_df: DataFrame with historical data for additional calculations.
        **kwargs: дополнительные параметры для совместимости
    """
    logger.debug("optimize_price called with kwargs: %s", kwargs)
    
    # Валидация параметров сетки цен
    if price_min > price_max:
        raise ValueError(f"price_min ({price_min}) must be <= price_max ({price_max})")
    if step <= 0:
        raise ValueError(f"step ({step}) must be > 0")
    
    # Проверка что сетка не пустая
    price_points = np.arange(price_min, price_max + step, step)
    if len(price_points) == 0:
        raise ValueError(f"Price grid is empty: price_min={price_min}, price_max={price_max}, step={step}")
    
    logger.info("Price grid: %d points from %.2f to %.2f with step %.2f", 
                len(price_points), price_min, price_max, step)
    
    # Интерфейсные проверки forecaster
    has_get_info = callable(getattr(forecaster, "get_info", None))
    has_predict_sales = callable(getattr(forecaster, "predict_sales", None))
    has_predict = callable(getattr(forecaster, "predict", None))

    missing_methods = []
    if not has_get_info:
        missing_methods.append("get_info")
    if not (has_predict_sales or has_predict):
        missing_methods.append("predict_sales|predict")
    if missing_methods:
        raise ValueError(f"Forecaster missing required methods: {missing_methods}")

    def _predict_qty(price_after_spp: float, features: Dict[str, Any]) -> float:
        """Совместимый вызов прогноза для старого и нового интерфейсов."""
        if has_predict_sales:
            return float(forecaster.predict_sales(price_after_spp, features))

        # Legacy fallback: predict(price, features) or predict(price)
        try:
            pred = forecaster.predict(price_after_spp, features)
        except TypeError:
            pred = forecaster.predict(price_after_spp)

        if isinstance(pred, (list, tuple, np.ndarray, pd.Series)):
            return float(pred[0])
        return float(pred)
    
    # Проверка что forecaster обучен
    forecaster_info = getattr(forecaster, 'get_info', lambda: {})()
    if not forecaster_info:
        logger.warning("Forecaster get_info() returned empty or missing")
    
    logger.debug("Forecaster validation passed")
    logger.debug("optimize_price called with forecaster type: %s", type(forecaster))
    logger.debug("base_features type: %s", type(base_features))
    logger.debug("sku_df type: %s", type(sku_df))
    logger.debug("sku_df is None: %s", sku_df is None)
    
    if sku_df is not None:
        logger.debug("sku_df empty: %s", sku_df.empty)
        logger.debug("sku_df columns: %s", list(sku_df.columns) if hasattr(sku_df, 'columns') else 'No columns attr')
    
    # Валидация типов
    if not isinstance(base_features, dict):
        raise TypeError(f"base_features must be dict, got {type(base_features)}")
    
    if sku_df is not None and not isinstance(sku_df, pd.DataFrame):
        raise TypeError(f"sku_df must be pandas.DataFrame or None, got {type(sku_df)}")
    
    # Проверка что forecaster обучен (интерфейсная проверка)
    best_model_name = getattr(forecaster, 'best_model_name', None)
    if best_model_name is None:
        logger.warning("Forecaster best_model_name is None - may not be trained")
    
    logger.debug("Basic validation passed")
    
    # Получаем информацию о модели
    try:
        f_info = forecaster.get_info()
        logger.debug("Got forecaster info: %s", type(f_info))
    except Exception as e:
        logger.error("Error getting forecaster info: %s", e)
        raise
    
    # Нормализованные внутренние имена флагов
    stability_mode = f_info.get("stability_mode", "S1")
    monotonicity_flag = f_info.get("monotonicity_flag", "monotone")
    protective_mode = f_info.get("protective_mode")
    
    logger.info("Model flags - stability: %s, monotonicity: %s, protective: %s", 
                stability_mode, monotonicity_flag, protective_mode)
    
    # Режим оптимизации по умолчанию
    regime = stability_mode
    penalty_enabled = False
    
    # Определяем текущую цену ПОСЛЕ СПП и текущую ПРИБЫЛЬ для расчетов
    current_p_after = None
    current_profit_daily = 0.0
    
    # Квантили истории для режимов (по price_after_spp)
    p5, p10, p90, p95 = None, None, None, None
    
    logger.debug("Starting sku_df processing")
    
    if sku_df is not None and not sku_df.empty and "price_after_spp" in sku_df.columns:
        try:
            logger.debug("Processing sku_df for current price")
            last_row_sku = sku_df.sort_values("date").iloc[-1]
            logger.debug("Got last row, type: %s", type(last_row_sku))
            
            current_p_after = float(last_row_sku["price_after_spp"])
            logger.debug("Got current_p_after: %s", current_p_after)
            
            # Считаем текущую прибыль (прогнозную) по ТЗ: UnitMargin * Orders
            p_last_before = float(last_row_sku["price_before_spp"])
            logger.debug("Got p_last_before: %s", p_last_before)
            
            s_val = spp
            p_after_last = p_last_before * (1.0 - s_val)
            comm_last = p_last_before * commission_rate
            vat_last = p_after_last * vat_rate
            m_last = p_last_before - comm_last - vat_last - cogs - logistics - storage
            logger.debug("Calculated margin: %s", m_last)
            
            # Прогноз для текущей цены
            logger.debug("Calling predict_sales")
            q_last = _predict_qty(current_p_after, base_features)
            logger.debug("Got q_last: %s", q_last)
            
            current_profit_daily = m_last * q_last
            logger.debug("Calculated current_profit_daily: %s", current_profit_daily)
            
            # Расчет квантилей
            logger.debug("Calculating quantiles")
            p_after_hist = sku_df["price_after_spp"]
            p5, p10, p90, p95 = p_after_hist.quantile([0.05, 0.1, 0.9, 0.95])
            logger.debug("Got quantiles: p5=%s, p10=%s, p90=%s, p95=%s", p5, p10, p90, p95)
            
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.warning("Error processing sku_df: %s", e)
            # Устанавливаем значения по умолчанию
            current_p_after = None
            current_profit_daily = None
    else:
        logger.debug("sku_df is None, empty, or missing price_after_spp column")

    # --- Определение режима на основе стабильности и монотонности ---
    reg_min_a, reg_max_a = 0, float("inf")
    
    if protective_mode == "S1":
        regime = "Защитный S1 (Критически мало данных)"
        logger.info("Using protective S1 regime - critically low data")
    elif stability_mode == "S3":
        regime = "S3 (Нестабильный)"
        penalty_enabled = True
        logger.info("Using S3 regime - unstable model with penalty")
        # Ограничение диапазона: intersection([current * 0.8; current * 1.2], [p10; p90])
        low_a, high_a = 0, float("inf")
        if current_p_after is not None:
            low_a, high_a = current_p_after * 0.8, current_p_after * 1.2
        
        if p10 is not None: low_a = max(low_a, p10)
        if p90 is not None: high_a = min(high_a, p90)
        reg_min_a, reg_max_a = low_a, high_a
    elif stability_mode == "S2":
        regime = "S2 (Умеренный)"
        logger.info("Using S2 regime - moderate stability")
        # Оптимизация внутри [p10; p90]
        if p10 is not None: reg_min_a = max(reg_min_a, p10)
        if p90 is not None: reg_max_a = min(reg_max_a, p90)
    else:  # S1 stability - Stable
        regime = "S1 (Стабильный)"
        logger.info("Using S1 regime - stable model")
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
        q = _predict_qty(p_after, base_features)
        raw_preds.append(q)
        customer_prices.append(p_after)
    
    # Калибровка всей кривой разом при немонотонности (ТЗ 8)
    # Используем ту же логику, что и в _calculate_numerical_elasticity
    if monotonicity_flag == "non_monotone":
        logger.debug("Applying non-monotone calibration")
        calibrated_preds = forecaster.calibrate_curve(np.array(customer_prices), np.array(raw_preds))
    else:
        logger.debug("Using monotone predictions without calibration")
        calibrated_preds = np.array(raw_preds)
    
    if protective_mode == "S1" and sku_df is not None and not sku_df.empty and "orders" in sku_df.columns:
        # Перетираем калиброванные прогнозы константой для режима S1
        logger.info("Applying S1 protective mode - using constant predictions")
        try:
            last_orders = sku_df.sort_values("date").tail(14)["orders"].median()
            calibrated_preds = np.full_like(calibrated_preds, last_orders)
            logger.debug("S1 mode: using median orders from last 14 days: %s", last_orders)
        except (KeyError, IndexError, ValueError) as e:
            logger.warning("Error processing S1 regime: %s", e)
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
    best_price = float(best_row["price_before_spp"])
    best_profit = float(best_row["profit"])

    # Boundary detection и логирование причин выбора цены
    is_boundary_search = bool(best_idx == 0 or best_idx == len(df_res) - 1)
    
    if is_boundary_search:
        if best_idx == 0:
            boundary_reason = "Optimum at minimum price boundary - profit decreases with higher prices"
        else:
            boundary_reason = "Optimum at maximum price boundary - profit decreases with lower prices"
        logger.warning("Boundary search detected: %s", boundary_reason)
    else:
        logger.info("Internal optimum found - price is within search range boundaries")

    # Двухуровневый boundary-флаг: 2) близость к историческим границам
    is_boundary_history = False
    boundary_history_reason = None
    tol = float(step)
    if hist_min_before_effective is not None and hist_max_before_effective is not None:
        hist_span = float(hist_max_before_effective - hist_min_before_effective)
        if hist_span > 0:
            tol = max(float(step), hist_span * 0.02)

        p_opt_before = best_price
        is_boundary_history = (
            abs(p_opt_before - float(hist_min_before_effective)) <= tol
            or abs(p_opt_before - float(hist_max_before_effective)) <= tol
        )
        
        if is_boundary_history:
            if abs(p_opt_before - float(hist_min_before_effective)) <= tol:
                boundary_history_reason = f"Optimum near historical minimum ({hist_min_before_effective:.2f})"
            else:
                boundary_history_reason = f"Optimum near historical maximum ({hist_max_before_effective:.2f})"
            logger.warning("Historical boundary detected: %s", boundary_history_reason)

    # Логирование итогового выбора цены
    logger.info("Price optimization completed:")
    logger.info("  - Best price: %.2f (before SPP)", best_price)
    logger.info("  - Best profit: %.2f", best_profit)
    logger.info("  - Boundary search: %s", is_boundary_search)
    logger.info("  - Boundary history: %s", is_boundary_history)
    logger.info("  - Regime: %s", regime)

    best_info = {
        "best_price_before_spp": best_row["price_before_spp"],
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
