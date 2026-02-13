"""Financial calculations for unit economics.

Functions compute commission, VAT, margin per unit and profit for rows in a DataFrame.
"""
from typing import Iterable

import pandas as pd


def commission_per_unit(price_before: pd.Series, commission_rate: float) -> pd.Series:
    """Комиссия маркетплейса за единицу (в тех же единицах, что и цена).

    Args:
        price_before: Series цен до СПП.
        commission_rate: доля (например, 0.15).
    Returns:
        Series с комиссией на единицу.
    """
    return price_before * commission_rate


def vat_per_unit(price_after: pd.Series, vat_rate: float) -> pd.Series:
    """НДС на единицу по цене после СПП.

    Args:
        price_after: Series цен после СПП.
        vat_rate: доля (например, 0.20).
    """
    return price_after * vat_rate


def revenue_per_row(price_before: pd.Series, qty: pd.Series) -> pd.Series:
    return price_before * qty


def margin_per_unit(
    price_before: pd.Series,
    price_after: pd.Series,
    commission_rate: float,
    vat_rate: float,
    cogs: pd.Series,
    logistics: pd.Series,
    storage: pd.Series,
) -> pd.Series:
    """Маржа на единицу после вычета переменных затрат и налогов/комиссий.

    Формула: margin_unit = price_before - commission - vat - cogs - logistics - storage
    (VAT считается от price_after)
    """
    commission = commission_per_unit(price_before, commission_rate)
    vat = vat_per_unit(price_after, vat_rate)
    return price_before - commission - vat - cogs - logistics - storage


def profit_total(
    margin_unit: pd.Series,
    qty: pd.Series,
) -> pd.Series:
    """Общая прибыль для строки: margin_unit * qty."""
    return margin_unit * qty


def apply_fin_calculations(
    df: pd.DataFrame,
    commission_rate: float,
    vat_rate: float,
    price_before_col: str = "price_before_spp",
    price_after_col: str = "price_after_spp",
    qty_col: str = "orders",
    cogs_col: str = "cogs",
    logistics_col: str = "logistics",
    storage_col: str = "storage",
) -> pd.DataFrame:
    """Добавляет в датафрейм колонки с расчётами.

    Добавляет:
        - `commission_per_unit`
        - `vat_per_unit`
        - `margin_per_unit`
        - `revenue`
        - `profit`

    Возвращает новый DataFrame (копия).
    """
    df = df.copy()

    p_before = df[price_before_col]
    p_after = df[price_after_col]
    qty = df[qty_col]
    cogs = df[cogs_col]
    logistics = df[logistics_col]
    storage = df[storage_col]

    df["commission_per_unit"] = commission_per_unit(p_before, commission_rate)
    df["vat_per_unit"] = vat_per_unit(p_after, vat_rate)
    df["margin_per_unit"] = margin_per_unit(
        p_before, p_after, commission_rate, vat_rate, cogs, logistics, storage
    )
    df["revenue"] = revenue_per_row(p_before, qty)
    df["profit"] = profit_total(df["margin_per_unit"], qty)

    return df


__all__ = [
    "commission_per_unit",
    "vat_per_unit",
    "margin_per_unit",
    "profit_total",
    "apply_fin_calculations",
]
