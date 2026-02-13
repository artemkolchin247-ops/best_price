"""Sales forecasting module.

Provides two models: Log-Log linear regression and RandomForestRegressor.
Includes cross-validation and model selection. Exposes `SalesForecaster` with
`fit(df)` and `predict_sales(price, features_row)` methods.
"""
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import time
import uuid
import logging
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class BaseModel:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_elasticity(self) -> float:
        return 0.0


class LogLogModel(BaseModel):
    """Log-log linear model: ln(y+1) ~ ln(p) + other features"""

    def __init__(self):
        self.model = LinearRegression()

    def _transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = X.copy()
        for col in Xt.columns:
            # Логарифмируем только числовые колонки
            if pd.api.types.is_numeric_dtype(Xt[col]):
                Xt[col] = np.log(Xt[col].astype(float).clip(lower=1.0))
        return Xt

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_t = np.log1p(y)
        Xt = self._transform_X(X)
        self.model.fit(Xt, y_t)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self._transform_X(X)
        pred_ln = self.model.predict(Xt)
        return np.expm1(pred_ln)

    def get_elasticity(self, feature_names: List[str]) -> float:
        """Возвращает коэффициент при цене (эластичность)."""
        if hasattr(self.model, "coef_"):
            feature_names = feature_names or []
            try:
                # В LogLogModel коэффициенты соответствуют логарифмированным признакам
                if "price_after_spp" in feature_names:
                    idx = feature_names.index("price_after_spp")
                    return float(self.model.coef_[idx])
                return 0.0
            except (ValueError, IndexError):
                return 0.0
        return 0.0


class RFModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class PoissonModel(BaseModel):
    """Poisson regression for count data."""
    def __init__(self):
        # alpha=0.0 is traditional Poisson (no penalty), 
        # but alpha > 0 adds regularization.
        self.model = PoissonRegressor(alpha=0.1, max_iter=300)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


# Конфигурация рекламных признаков (ТЗ 6.2)
AD_FEATURES = ["ad_internal", "ad_bloggers", "ad_vk"]

class SalesForecaster:
    """Train and select best model for sales forecasting.

    Usage:
        sf = SalesForecaster(feature_cols=[...])
        sf.fit(df)  # df must contain 'orders' and 'price_after_spp' and feature_cols
        q = sf.predict_sales(price=1200, features_row=row_dict)
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, time_col: str = "date"):
        # Исключаем целевые колонки, утечки и запрещенные признаки из признаков (ТЗ 2.2)
        leaks = {"orders", "revenue", "profit", "margin", "margin_unit", "conversion"}
        forbidden_features = {"price_before_spp", "spp", "cogs", "logistics", "storage"}
        raw_features = feature_cols or ["price_after_spp", "day_of_week", "ad_internal", "ad_bloggers", "ad_vk"]
        self.feature_cols = [c for c in raw_features 
                           if c.lower() not in leaks and 
                           c not in forbidden_features and
                           not c.startswith('Unnamed')]
        self.time_col = time_col
        self.models: Dict[str, BaseModel] = {
            "loglog": LogLogModel(),
            "rf": RFModel(),
            "poisson": PoissonModel()
        }
        
        # Diagnostics & Quality
        self.quality_info: Dict[str, Any] = {}
        self.elasticity_info: Dict[str, Any] = {}
        self.performance_info: Dict[str, Any] = {}
        
        self.best_model_name: Optional[str] = None
        self.stability_mode: str = "S1"
        self.monotonicity_flag: str = "monotone"
        self.protective_mode: Optional[str] = None
        
        # Структурированные логи пайплайна (ТЗ 2) - канонический атрибут
        self.pipeline_log = {
            "run_id": f"run_{int(time.time())}",
            "steps": []
        }
        
        # Логи пайплайна для отображения в UI (legacy, derived view)
        self.pipeline_logs: List[str] = []
        
        # Состояние данных для gating UI (ТЗ)
        self.data_state: str = "OK"  # OK, EMPTY, TOO_SMALL, NO_PRICE_VARIATION, FAILED
        
        # Детальная информация об ошибках (ТЗ 1)
        self.error = {
            "code": "",
            "message": "",
            "failed_step": "",
            "exception_type": "",
            "traceback_id": None
        }

    def _reset_state(self):
        """Полная очистка состояния модели (убрать кэш прошлых запусков)."""
        self.best_model_name = None
        self.stability_mode = "S1"
        self.monotonicity_flag = "monotone"
        self.protective_mode = None
        self.quality_info = {}
        self.elasticity_info = {}
        self.performance_info = {}
        self.model_result = {}
        self.pipeline_logs = []
        self.data_state = "OK"
        self.error = {
            "code": "",
            "message": "",
            "failed_step": "",
            "exception_type": "",
            "traceback_id": None
        }
        # Создаем новый run_id для логов
        self.pipeline_log = {
            "run_id": f"run_{int(time.time())}",
            "steps": []
        }
        # pipeline_logs будет обновляться автоматически через _add_pipeline_step

    def _add_pipeline_step(self, name: str, data: pd.DataFrame, status: str = "ok", notes: str = None):
        """Добавляет шаг в структурированный лог пайплайна."""
        key_fields = ['date', 'orders', 'price_after_spp', 'price_before_spp', 'spp']
        nan_counts = {}
        for field in key_fields:
            if field in data.columns:
                nan_counts[field] = int(data[field].isna().sum())
            else:
                nan_counts[field] = 0
        
        # Безопасное вычисление date_min/date_max (ТЗ)
        date_min = None
        date_max = None
        if self.time_col in data.columns and len(data) > 0:
            try:
                dt = pd.to_datetime(data[self.time_col], errors="coerce")
                if dt.notna().any():  # Проверяем что есть хотя бы одна валидная дата
                    date_min = dt.min()
                    date_max = dt.max()
                    date_min = date_min.isoformat() if pd.notna(date_min) else None
                    date_max = date_max.isoformat() if pd.notna(date_max) else None
            except (ValueError, TypeError):
                # Если что-то пошло не так, оставляем None
                date_min = None
                date_max = None
        
        step_info = {
            "name": name,
            "status": status,
            "rows": len(data),
            "cols": len(data.columns),
            "nan_counts": nan_counts,
            "date_min": date_min,
            "date_max": date_max,
            "notes": notes
        }
        
        self.pipeline_log["steps"].append(step_info)
        
        # Также сохраняем в старый формат для совместимости
        log_message = f"📊 {name}: rows={len(data)}, NaN counts: {nan_counts}"
        if status == "failed":
            log_message = f"🚨 {name}: FAILED - {notes}"
        self.pipeline_logs.append(log_message)

    def _set_error(self, code: str, message: str, failed_step: str, exception: Exception = None):
        """Устанавливает детальную информацию об ошибке с декларативными кодами."""
        
        # Определяем рекомендации для каждого кода ошибки
        error_recommendations = {
            "E_NO_FILES": [
                "Загрузите файлы с данными через интерфейс",
                "Убедитесь что файлы содержат данные о продажах"
            ],
            "E_MISSING_COLUMNS": [
                "Добавьте обязательные колонки: date, orders, price_after_spp",
                "Проверьте названия колонок в исходном файле"
            ],
            "E_DATE_PARSE_FAILED": [
                "Проверьте формат дат в исходном файле (рекомендуемый: YYYY-MM-DD)",
                "Убедитесь что все строки имеют корректные даты"
            ],
            "E_NUMERIC_CAST_FAILED": [
                "Проверьте что price и orders содержат только числа",
                "Удалите текстовые значения или исправьте формат данных"
            ],
            "E_FILTER_EXCLUDED_ALL": [
                "Расширьте временной период для анализа",
                "Проверьте что даты в файле соответствуют выбранному периоду"
            ],
            "E_DROPNAS_REMOVED_ALL": [
                "Заполните пропущенные значения в ключевых колонках",
                "Проверьте что нет пустых строк в данных"
            ],
            "E_PIPELINE_EXCEPTION": [
                "Проверьте структуру и формат исходных данных",
                "Обратитесь к разработчику с деталями ошибки"
            ]
        }
        
        self.data_state = "FAILED"
        self.error = {
            "code": code,
            "message": message,
            "failed_step": failed_step,
            "exception_type": type(exception).__name__ if exception else "",
            "traceback_id": str(uuid.uuid4())[:8] if exception else None,
            "recommendations": error_recommendations.get(code, ["Проверьте данные и попробуйте снова"])
        }

    def _validate_input_files(self, df: pd.DataFrame) -> bool:
        """Проверка наличия входных файлов."""
        if df is None or df.empty:
            self._set_error("E_NO_FILES", "Входные файлы не загружены или пусты", "input_validation")
            return False
        return True

    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        """Проверка наличия обязательных колонок."""
        required_cols = [self.time_col, "orders", "price_after_spp"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self._set_error(
                "E_MISSING_COLUMNS", 
                f"Отсутствуют обязательные колонки: {', '.join(missing_cols)}", 
                "column_validation"
            )
            return False
        return True

    def _validate_date_parsing(self, df: pd.DataFrame) -> bool:
        """Проверка успешности парсинга дат."""
        if self.time_col in df.columns:
            nat_ratio = df[self.time_col].isna().sum() / len(df)
            if nat_ratio > 0.5:  # Более 50% дат не распарсились
                self._set_error(
                    "E_DATE_PARSE_FAILED",
                    f"Не удалось распарсить {nat_ratio:.1%} дат",
                    "date_validation"
                )
                return False
        return True

    def _validate_numeric_cast(self, df: pd.DataFrame) -> bool:
        """Проверка успешности приведения к числам."""
        numeric_cols = ["orders", "price_after_spp"]
        for col in numeric_cols:
            if col in df.columns:
                non_numeric_ratio = pd.to_numeric(df[col], errors='coerce').isna().sum() / len(df)
                if non_numeric_ratio > 0.5:  # Более 50% не приводятся к числам
                    self._set_error(
                        "E_NUMERIC_CAST_FAILED",
                        f"Не удалось привести {non_numeric_ratio:.1%} значений в колонке {col} к числам",
                        "numeric_validation"
                    )
                    return False
        return True

    def _validate_filter_results(self, df_before: pd.DataFrame, df_after: pd.DataFrame, step_name: str) -> bool:
        """Проверка что фильтрация не удалила все данные."""
        if len(df_after) == 0 and len(df_before) > 0:
            if step_name == "filter_period":
                self._set_error("E_FILTER_EXCLUDED_ALL", "Фильтрация по периоду удалила все строки", step_name)
            elif step_name == "drop_invalid_rows":
                self._set_error("E_DROPNAS_REMOVED_ALL", "Очистка от NaN удалила все строки", step_name)
            else:
                self._set_error("E_PIPELINE_EXCEPTION", f"Шаг {step_name} удалил все данные", step_name)
            return False
        return True

    def _log_data_step(self, step_name: str, data: pd.DataFrame):
        """Логирует размер данных и NaN по ключевым полям."""
        rows_count = len(data)
        key_fields = ['date', 'orders', 'price_after_spp']
        nan_counts = {}
        for field in key_fields:
            if field in data.columns:
                nan_counts[field] = data[field].isna().sum()
            else:
                nan_counts[field] = 'N/A'
        
        log_message = f"📊 {step_name}: rows={rows_count}, NaN counts: {nan_counts}"
        self.pipeline_logs.append(log_message)
        
        # Критическое предупреждение если строки закончились
        if rows_count == 0:
            critical_message = f"🚨 CRITICAL: Data became empty at step: {step_name}"
            self.pipeline_logs.append(critical_message)
            logger.warning(critical_message)

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка данных для модели с обязательными шагами пайплайна и валидацией."""
        df2 = df.copy()
        
        # 1. load_input - загрузка входных данных
        if not self._validate_input_files(df2):
            return pd.DataFrame(), pd.Series([])
        self._add_pipeline_step("load_input", df2)
        
        # 2. normalize_columns - удаление Unnamed и приведение к нижнему регистру
        # Удаляем колонки с 'Unnamed:' в названии
        unnamed_cols = [col for col in df2.columns if col.startswith('Unnamed:')]
        if unnamed_cols:
            df2 = df2.drop(columns=unnamed_cols)
            self._add_pipeline_step("normalize_columns", df2, notes=f"Dropped {len(unnamed_cols)} Unnamed columns")
        else:
            self._add_pipeline_step("normalize_columns", df2)
        
        # 3. Проверка обязательных колонок
        if not self._validate_required_columns(df2):
            return pd.DataFrame(), pd.Series([])
        
        # 4. parse_dates - парсинг дат
        if self.time_col in df2.columns:
            try:
                df2[self.time_col] = pd.to_datetime(df2[self.time_col], errors="coerce")
                self._add_pipeline_step("parse_dates", df2)
                
                # Валидация парсинга дат
                if not self._validate_date_parsing(df2):
                    return pd.DataFrame(), pd.Series([])
                    
            except Exception as e:
                self._set_error("E_DATE_PARSE_FAILED", f"Ошибка парсинга дат: {str(e)}", "parse_dates", e)
                self._add_pipeline_step("parse_dates", df2, "failed", f"Exception: {str(e)}")
                return pd.DataFrame(), pd.Series([])
        else:
            self._add_pipeline_step("parse_dates", df2, notes="No date column found")
        
        # 5. cast_numeric - приведение типов для ключевых колонок
        try:
            # Приводим orders к float
            if "orders" in df2.columns:
                df2["orders"] = pd.to_numeric(df2["orders"], errors="coerce")
            
            # Приводим price_after_spp к float
            if "price_after_spp" in df2.columns:
                df2["price_after_spp"] = pd.to_numeric(df2["price_after_spp"], errors="coerce")
            
            # Приводим другие колонки к numeric где возможно
            for col in df2.columns:
                if col not in [self.time_col, "orders", "price_after_spp"]:
                    if df2[col].dtype == 'object':
                        try:
                            df2[col] = pd.to_numeric(df2[col], errors="coerce")
                        except (ValueError, TypeError):
                            pass  # Оставляем как есть если не получается
            
            self._add_pipeline_step("cast_numeric", df2)
            
            # Валидация приведения к числам
            if not self._validate_numeric_cast(df2):
                return pd.DataFrame(), pd.Series([])
                
        except Exception as e:
            self._set_error("E_NUMERIC_CAST_FAILED", f"Ошибка приведения типов: {str(e)}", "cast_numeric", e)
            self._add_pipeline_step("cast_numeric", df2, "failed", f"Exception: {str(e)}")
            return pd.DataFrame(), pd.Series([])
        
        # 6. filter_period - фильтрация периода (если есть)
        if self.time_col in df2.columns:
            try:
                # Удаляем строки с невалидными датами
                valid_dates = df2[self.time_col].notna()
                df2_before = df2.copy()
                df2 = df2[valid_dates]
                
                # Валидация результатов фильтрации
                if not self._validate_filter_results(df2_before, df2, "filter_period"):
                    return pd.DataFrame(), pd.Series([])
                    
                self._add_pipeline_step("filter_period", df2, notes=f"Filtered {len(df2_before) - len(df2)} rows with invalid dates")
            except Exception as e:
                self._set_error("E_PIPELINE_EXCEPTION", f"Ошибка фильтрации периода: {str(e)}", "filter_period", e)
                self._add_pipeline_step("filter_period", df2, "failed", f"Exception: {str(e)}")
                return pd.DataFrame(), pd.Series([])
        else:
            self._add_pipeline_step("filter_period", df2, notes="No date column for period filtering")
        
        # 7. drop_invalid_rows - dropna по ключевым колонкам
        try:
            # Определяем ключевые колонки для проверки
            key_cols = [self.time_col, "orders", "price_after_spp"]
            key_cols = [col for col in key_cols if col in df2.columns]
            
            if key_cols:
                df2_before = df2.copy()
                df2 = df2.dropna(subset=key_cols)
                rows_dropped = len(df2_before) - len(df2)
                
                # Валидация результатов dropna
                if not self._validate_filter_results(df2_before, df2, "drop_invalid_rows"):
                    return pd.DataFrame(), pd.Series([])
                    
                self._add_pipeline_step("drop_invalid_rows", df2, notes=f"Dropped {rows_dropped} rows with NaN in key columns")
            else:
                self._add_pipeline_step("drop_invalid_rows", df2, notes="No key columns for NaN check")
        except Exception as e:
            self._set_error("E_DROPNAS_REMOVED_ALL", f"Ошибка удаления невалидных строк: {str(e)}", "drop_invalid_rows", e)
            self._add_pipeline_step("drop_invalid_rows", df2, "failed", f"Exception: {str(e)}")
            return pd.DataFrame(), pd.Series([])
        
        # 8. Удаляем запрещенные признаки для demand model (ТЗ 2.2)
        try:
            forbidden_features = {"price_before_spp", "spp", "cogs", "logistics", "storage"}
            cols_to_drop = []
            for col in df2.columns:
                if col in forbidden_features:
                    cols_to_drop.append(col)
            if cols_to_drop:
                df2 = df2.drop(columns=cols_to_drop)
                self._add_pipeline_step("remove_forbidden_features", df2, notes=f"Dropped {len(cols_to_drop)} forbidden features")
        except Exception as e:
            self._set_error("E_PIPELINE_EXCEPTION", f"Ошибка удаления запрещенных признаков: {str(e)}", "remove_forbidden_features", e)
            self._add_pipeline_step("remove_forbidden_features", df2, "failed", f"Exception: {str(e)}")
            return pd.DataFrame(), pd.Series([])
        
        # 9. Обработка price колонки
        try:
            if "price" in df2.columns:
                if "price_after_spp" not in df2.columns:
                    df2["price_after_spp"] = df2["price"]
                # Удаляем 'price', чтобы не было дублей-подсказок для модели
                df2 = df2.drop(columns=["price"])
                self._add_pipeline_step("handle_price_column", df2, notes="Converted price to price_after_spp")
        except Exception as e:
            self._set_error("E_PIPELINE_EXCEPTION", f"Ошибка обработки колонки цены: {str(e)}", "handle_price_column", e)
            self._add_pipeline_step("handle_price_column", df2, "failed", f"Exception: {str(e)}")
            return pd.DataFrame(), pd.Series([])
        
        # 10. feature_engineering - создание новых признаков
        try:
            if self.time_col in df2.columns:
                df2 = df2.sort_values(self.time_col)
                # Извлекаем день недели (0=Mon, 6=Sun)
                df2["day_of_week"] = df2[self.time_col].dt.dayofweek
                self._add_pipeline_step("feature_engineering", df2, notes="Added day_of_week feature")
            else:
                self._add_pipeline_step("feature_engineering", df2, notes="No date column for feature engineering")
        except Exception as e:
            self._set_error("E_PIPELINE_EXCEPTION", f"Ошибка feature engineering: {str(e)}", "feature_engineering", e)
            self._add_pipeline_step("feature_engineering", df2, "failed", f"Exception: {str(e)}")
            return pd.DataFrame(), pd.Series([])
        
        # 11. final_dataset - финальная подготовка данных
        try:
            # Фиксируем список колонок ДЛЯ МОДЕЛЕЙ (без orders и date)
            leaks = {"orders", "revenue", "profit", "margin", "margin_unit", "conversion", self.time_col}
            forbidden_features = {"price_before_spp", "spp", "cogs", "logistics", "storage"}
            
            # Ensure 'price_after_spp' is always in feature_cols if it exists in df2
            if "price_after_spp" in df2.columns and "price_after_spp" not in self.feature_cols:
                self.feature_cols.append("price_after_spp")

            # Add any other relevant columns from df2 that are not leaks and not already in feature_cols
            for col in df2.columns:
                if (col not in leaks and 
                    col not in self.feature_cols and 
                    col not in forbidden_features and
                    not col.startswith('Unnamed')):
                    self.feature_cols.append(col)
            
            # Обработка SKU для множества SKU (one-hot encoding)
            if "sku" in df2.columns and "sku" in self.feature_cols:
                unique_skus = df2["sku"].nunique()
                if unique_skus > 1:
                    # One-hot encoding для множества SKU
                    sku_dummies = pd.get_dummies(df2["sku"], prefix="sku")
                    df2 = pd.concat([df2, sku_dummies], axis=1)
                    # Удаляем исходную колонку sku
                    df2 = df2.drop(columns=["sku"])
                    # Обновляем feature_cols - убираем sku, добавляем one-hot колонки
                    self.feature_cols = [col for col in self.feature_cols if col != "sku"]
                    self.feature_cols.extend(sku_dummies.columns.tolist())
                    self._add_pipeline_step("sku_encoding", df2, notes=f"One-hot encoded {unique_skus} SKUs: {len(sku_dummies.columns)} features")
            
            # Filter X to only include columns that are actually present in df2
            # and are in self.feature_cols, and ARE NUMERIC
            X_cols_to_use = [col for col in self.feature_cols if col in df2.columns]
            X = df2[X_cols_to_use].copy()
            
            # Select only numeric columns for models
            X = X.select_dtypes(include=[np.number])
            self.feature_cols = X.columns.tolist()
            
            self._add_pipeline_step("final_dataset", X, notes=f"Final dataset: {len(X.columns)} numeric columns")
            
            y = df2["orders"].astype(float)
            # fill NA
            X = X.fillna(0)
            
            return X, y
            
        except Exception as e:
            self._set_error("E_PIPELINE_EXCEPTION", f"Ошибка создания финального датасета: {str(e)}", "final_dataset", e)
            self._add_pipeline_step("final_dataset", pd.DataFrame(), "failed", f"Exception: {str(e)}")
            return pd.DataFrame(), pd.Series([])

    def _calculate_quality_metrics(self, df: pd.DataFrame):
        """Расчет базовых метрик качества данных и определение data_state."""
        try:
            if df.empty:
                self.data_state = "EMPTY"
                self._set_error("EMPTY_DATA", "Входной DataFrame пуст", "quality_check")
                self._add_pipeline_step("quality_check", df, "failed", "DataFrame is empty")
                return
            
            # Число дней
            n_days = len(df[self.time_col].unique()) if self.time_col in df.columns else len(df)
            
            # Уникальность цен
            if "price_after_spp" in df.columns:
                n_price_unique = df["price_after_spp"].nunique()
                price_cv = df["price_after_spp"].std() / df["price_after_spp"].mean() if df["price_after_spp"].mean() > 0 else 0
            else:
                n_price_unique = 0
                price_cv = 0
            
            # Доля нулей
            zero_share = (df["orders"] == 0).sum() / len(df) if "orders" in df.columns else 1.0
            
            # Корреляция
            if "orders" in df.columns and "price_after_spp" in df.columns:
                corr = df["orders"].corr(df["price_after_spp"])
            else:
                corr = 0
            
            # Определяем состояние данных (ТЗ)
            if n_days < 7 or n_price_unique < 3:
                self.data_state = "TOO_SMALL"
                notes = f"n_days={n_days}, n_price_unique={n_price_unique}"
                self._add_pipeline_step("quality_check", df, "failed", f"Data too small: {notes}")
            elif zero_share > 0.8 or price_cv < 0.01:
                self.data_state = "NO_PRICE_VARIATION"
                notes = f"zero_share={zero_share:.2f}, price_cv={price_cv:.4f}"
                self._add_pipeline_step("quality_check", df, "failed", f"No price variation: {notes}")
            else:
                self.data_state = "OK"
                self._add_pipeline_step("quality_check", df, "ok")
            
            self.quality_info = {
                "n_days": int(n_days),
                "n_price_unique": int(n_price_unique),
                "price_cv": float(price_cv),
                "zero_share": float(zero_share),
                "corr": float(corr),
                "data_ok": (n_days >= 30 and n_price_unique >= 6 and price_cv >= 0.03),
                "data_state": self.data_state
            }
            
        except Exception as e:
            self._set_error("QUALITY_ERROR", f"Ошибка расчета качества: {str(e)}", "quality_check", e)
            self._add_pipeline_step("quality_check", df, "failed", f"Exception: {str(e)}")
            raise

    def _detect_ad_features(self, df: pd.DataFrame) -> List[str]:
        """Авто-детект рекламных признаков в датасете (ТЗ 6.3)."""
        available_ad_features = []
        for col in AD_FEATURES:
            if col in df.columns:
                # Проверяем что колонка имеет числовые данные и достаточное наблюдений
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    non_null_count = df[col].notna().sum()
                    non_zero_count = (df[col] != 0).sum()
                    if non_null_count >= 30 and non_zero_count >= 10:
                        available_ad_features.append(col)
                        logger.debug("Found ad feature %s with %s observations", col, non_null_count)
                    else:
                        logger.debug("Ad feature %s has insufficient data (%s obs, %s non-zero)", col, non_null_count, non_zero_count)
                else:
                    logger.debug("Ad feature %s has non-numeric dtype: %s", col, df[col].dtype)
            else:
                logger.debug("Ad feature %s not found in dataset", col)
        
        return available_ad_features

    def _build_ad_profiles(self, df: pd.DataFrame, ad_features: List[str]) -> Dict[str, Dict[str, float]]:
        """Построение профилей рекламы Low/Med/High (ТЗ 6.4)."""
        profiles = {"low": {}, "med": {}, "high": {}}
        
        for col in ad_features:
            # Очистка данных
            clean_values = df[col].dropna().drop_duplicates()
            clean_values = clean_values[clean_values != 0]  # Убираем нули
            
            if len(clean_values) < 10:
                logger.debug("Insufficient non-zero values for %s: %s", col, len(clean_values))
                # Заполняем нулями если недостаточно данных
                profiles["low"][col] = 0.0
                profiles["med"][col] = 0.0
                profiles["high"][col] = 0.0
                continue
            
            # Winsorize p1-p99 для устойчивости
            p1, p99 = clean_values.quantile([0.01, 0.99])
            clean_values = clean_values.clip(p1, p99)
            
            # Квантили по колонке (универсально)
            p25, p50, p75 = clean_values.quantile([0.25, 0.5, 0.75])
            
            profiles["low"][col] = float(p25)
            profiles["med"][col] = float(p50)
            profiles["high"][col] = float(p75)
            
            logger.debug("%s profiles - low: %.2f, med: %.2f, high: %.2f", col, p25, p50, p75)
        
        return profiles

    def _get_base_features(self, df: pd.DataFrame, method: str = "last_day") -> Dict[str, float]:
        """Фиксация неценовых признаков в базовом состоянии (ТЗ 6.5)."""
        X, _ = self._prepare_xy(df)
        base_features = {}
        
        if method == "last_day":
            # Вариант A: значения последнего дня
            last_row = X.iloc[-1].to_dict()
        elif method == "median":
            # Вариант B: медианные значения по истории
            last_row = X.median().to_dict()
        elif method == "typical_day":
            # Вариант C: типичный день недели + медианы
            # Находим самый частый день недели
            if "day_of_week" in X.columns:
                typical_day = X["day_of_week"].mode().iloc[0]
                typical_data = X[X["day_of_week"] == typical_day]
                last_row = typical_data.median().to_dict()
            else:
                last_row = X.median().to_dict()
        else:
            last_row = X.iloc[-1].to_dict()
        
        # Исключаем ценовые признаки
        price_cols = ["price_after_spp", "price_before_spp", "spp", "price"]
        for col in price_cols:
            if col in last_row:
                del last_row[col]
        
        # Убедимся что все значения float
        for col, val in last_row.items():
            try:
                base_features[col] = float(val)
            except (ValueError, TypeError):
                base_features[col] = 0.0
        
        return base_features

    def _calculate_ad_profiles(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Расчет профилей рекламы (универсальный метод)."""
        # Авто-детект рекламных признаков
        ad_features = self._detect_ad_features(df)
        
        if not ad_features:
            logger.debug("No valid ad features found")
            return None  # Возвращаем None если нет признаков (ТЗ 3.1)
        
        # Построение профилей
        profiles = self._build_ad_profiles(df, ad_features)
        profiles["method"] = "last_day"  # Метод фиксации признаков
        profiles["available_features"] = ad_features
        
        return profiles

    def _calculate_numerical_elasticity(self, df: pd.DataFrame, ad_profile: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Расчет численной эластичности по сетке для текущей модели."""
        X, _ = self._prepare_xy(df)
        if "price_after_spp" not in X.columns:
            return {"elasticity_med": 0.0, "elasticity_iqr": 0.0, "mono_violations": 0.0, "grid_info": []}

        p_min, p_max = X["price_after_spp"].min(), X["price_after_spp"].max()
        if p_max <= p_min:
            return {"elasticity_med": 0.0, "elasticity_iqr": 0.0, "mono_violations": 0.0, "grid_info": []}

        # Определяем профиль рекламы и базовые признаки (ТЗ 6.5)
        if ad_profile is None:
            # Пытаемся использовать med профиль по умолчанию
            ad_profiles = self._calculate_ad_profiles(df)
            if ad_profiles is not None and ad_profiles.get("available_features"):
                ad_profile = {k: v for k, v in ad_profiles["med"].items() if k in ad_profiles["available_features"]}
            else:
                ad_profile = {}  # Пустой профиль если нет рекламных признаков
        
        # Фиксируем базовые признаки
        base_features = self._get_base_features(df, method="last_day")
        
        # Строим сетку (20 точек)
        p_grid = np.linspace(p_min, p_max, 20)
        
        preds = []
        for p in p_grid:
            # Формируем features_row с ценой и профилем рекламы
            features_row = {"price_after_spp": p}
            features_row.update(base_features)  # Базовые признаки
            features_row.update(ad_profile) if ad_profile else None  # Рекламный профиль
            
            q = max(float(self.predict_sales(p, features_row=features_row)), 1e-6)
            preds.append(q)
        
        preds = np.array(preds)
        q_grid_raw = preds.copy()  # Сохраняем оригинальные предсказания
        
        # Считаем монотонность по сырой кривой (ТЗ 4.2)
        violations_raw = 0
        for i in range(len(preds) - 1):
            if preds[i+1] > preds[i] * 1.03:  # threshold = 3%
                violations_raw += 1
        mono_v_raw = violations_raw / (len(preds) - 1)
        
        # Правило калибровки (ТЗ 4.2)
        if mono_v_raw > 0.2:
            # Применяем isotonic regression (не возрастающая)
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
            preds_calibrated = ir.fit_transform(p_grid, preds)
            q_grid_used = preds_calibrated
        else:
            preds_calibrated = preds
            q_grid_used = preds
        
        # Считаем монотонность по калиброванной кривой
        violations_used = 0
        for i in range(len(q_grid_used) - 1):
            if q_grid_used[i+1] > q_grid_used[i] * 1.03:  # threshold = 3%
                violations_used += 1
        mono_v_used = violations_used / (len(q_grid_used) - 1)
        
        # Считаем эластичность по q_grid_used (ТЗ 4.3)
        ln_p = np.log(p_grid)
        ln_q = np.log(q_grid_used)  # Используем калиброванную кривую
        
        # Удаляем возможные NaN/Inf значения
        valid_mask = np.isfinite(ln_p) & np.isfinite(ln_q)
        ln_p_valid = ln_p[valid_mask]
        ln_q_valid = ln_q[valid_mask]
        
        if len(ln_p_valid) < 2:
            return {"elasticity_med": 0.0, "elasticity_iqr": 0.0, "mono_violations": mono_v_used, "grid_info": []}
        
        # Глобальная эластичность через OLS регрессию ln(q) = a + b*ln(p) (ТЗ 4.1)
        from sklearn.linear_model import LinearRegression
        
        # Формируем X = ln(p_grid), Y = ln(q_grid_used)
        ln_p_grid = np.log(p_grid)
        ln_q_grid = np.log(np.clip(q_grid_used, 1e-6, None))
        
        # Удаляем NaN/Inf значения
        valid_mask = np.isfinite(ln_p_grid) & np.isfinite(ln_q_grid)
        ln_p_valid = ln_p_grid[valid_mask]
        ln_q_valid = ln_q_grid[valid_mask]
        
        n_points = len(ln_p_valid)
        if n_points < 3:
            # Недостаточно точек для регрессии
            e_global = None
            r2 = None
        else:
            # Проверяем дисперсию Y (ТЗ 4.2)
            y_var = np.var(ln_q_valid)
            if y_var < 1e-8:
                # Дисперсия слишком мала, регрессия не информативна
                e_global = None
                r2 = None
            else:
                # OLS регрессия ln(q) = a + b*ln(p)
                X_reg = ln_p_valid.reshape(-1, 1)
                Y_reg = ln_q_valid
                
                reg = LinearRegression()
                reg.fit(X_reg, Y_reg)
                e_global = float(reg.coef_[0])  # Наклон b = глобальная эластичность
                r2 = float(reg.score(X_reg, Y_reg))
        
        # Расчет локальной эластичности по центральной разности (ТЗ 2.1)
        e_grid_local = []
        eps = 1e-6
        
        # Edge-case: недостаточно точек для расчета локальной эластичности (ТЗ 2.3)
        if len(p_grid) < 5:
            return {
                "elasticity_med": None,  # Используем None для edge-cases
                "elasticity_iqr": None,
                "beta_median": None,
                "beta_iqr": None,
                "mono_violations": mono_v_used,
                "mono_violations_raw": mono_v_raw,
                "e_grid": None,
                "q_grid": q_grid_used.tolist(),
                "q_grid_raw": q_grid_raw.tolist(),
                "q_grid_calibrated": preds_calibrated.tolist() if mono_v_raw > 0.2 else None,
                "r_squared": r2,
                "e_stats": {"min": 0, "median": 0, "max": 0, "std": 0, "len": 0},
                "global_regression": {
                    "global_elasticity": e_global,
                    "r_squared": r2,
                    "n_points": n_points
                },
                "calibrated": mono_v_raw > 0.2,
                "insufficient_data": True
            }
        
        # Клиппируем q чтобы избежать log(0) и отрицательных значений
        q_clipped = np.clip(q_grid_used, eps, None)  # Используем q_grid_used (ТЗ 2.2)
        
        # Локальная эластичность по центральной разности для внутренних точек
        for i in range(1, len(p_grid) - 1):
            ln_q_i_minus_1 = np.log(q_clipped[i-1])
            ln_q_i_plus_1 = np.log(q_clipped[i+1])
            ln_p_i_minus_1 = np.log(p_grid[i-1])
            ln_p_i_plus_1 = np.log(p_grid[i+1])
            
            # Проверка деления на ноль
            denominator = ln_p_i_plus_1 - ln_p_i_minus_1
            if abs(denominator) > 1e-10:
                e_i = (ln_q_i_plus_1 - ln_q_i_minus_1) / denominator  # Формула из ТЗ 2.1
                e_grid_local.append(e_i)  # Не округляем (ТЗ 2.2)
            else:
                e_grid_local.append(0.0)
        
        # e_grid содержит только внутренние точки (края не считаем)
        e_grid_with_nan = [np.nan] + e_grid_local + [np.nan]  # Для согласованности длины
        
        # Статистика для отладки
        if len(e_grid_local) > 0:
            e_stats = {
                "min": float(np.min(e_grid_local)),
                "median": float(np.median(e_grid_local)),
                "max": float(np.max(e_grid_local)),
                "std": float(np.std(e_grid_local)),
                "len": len(e_grid_local),
                "valid_points": len(e_grid_local),  # Для UI
                "total_points": len(e_grid_local)  # Для UI
            }
        else:
            e_stats = {
                "min": 0, "median": 0, "max": 0, "std": 0, "len": 0,
                "valid_points": 0,  # Для UI
                "total_points": 0  # Для UI
            }
        # IQR считаем через бутстрап по точкам сетки для всех моделей
        bootstrap_elasticities = []
        n_bootstrap = 50
        np.random.seed(42)  # Для воспроизводимости
        
        for _ in range(n_bootstrap):
            # Сэмплируем точки с возвращением
            indices = np.random.choice(len(ln_p_valid), size=len(ln_p_valid), replace=True)
            if len(np.unique(indices)) < 2:
                continue
                
            p_boot = ln_p_valid[indices]
            q_boot = ln_q_valid[indices]
            
            reg_boot = LinearRegression()
            reg_boot.fit(p_boot.reshape(-1, 1), q_boot)
            bootstrap_elasticities.append(float(reg_boot.coef_[0]))
        
        if bootstrap_elasticities:
            q75, q25 = np.percentile(bootstrap_elasticities, [75, 25])
            e_iqr = float(q75 - q25)
        else:
            e_iqr = 0.0
        
        return {
            "elasticity_med": e_global,  # Глобальная эластичность из OLS
            "elasticity_iqr": e_iqr,
            "beta_median": e_global,  # Для совместимости с UI
            "beta_iqr": e_iqr,    # Для совместимости с UI
            "mono_violations": mono_v_used,  # Монотонность по используемой кривой
            "mono_violations_raw": mono_v_raw,  # Монотонность по сырой кривой
            "e_grid": e_grid_with_nan,  # Локальная эластичность по сетке
            "q_grid": q_grid_used.tolist(),  # Используемая кривая (калиброванная или сырая)
            "q_grid_raw": q_grid_raw.tolist(),    # Сырая кривая для отладки
            "q_grid_calibrated": preds_calibrated.tolist() if mono_v_raw > 0.2 else None,  # Калиброванная кривая
            "r_squared": r2,  # Качество регрессии
            "e_stats": e_stats,  # Статистика локальной эластичности
            "global_regression": {  # Информация о глобальной регрессии
                "global_elasticity": e_global,
                "r_squared": r2,
                "n_points": n_points
            },
            "calibrated": mono_v_raw > 0.2  # Была ли применена калибровка
        }

    def cross_validate(self, df: pd.DataFrame, n_splits: int = 3) -> Dict[str, float]:
        X, y = self._prepare_xy(df)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores: Dict[str, List[float]] = {name: [] for name in self.models.keys()}
        
        # Для LogLog замеряем эластичность по фолдам (ТЗ 2.1)
        loglog_betas = []
        
        # Для baseline calculation
        baseline_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Baseline: rolling mean prediction (ТЗ 7.1)
            if len(y_train) >= 7:
                baseline_pred = [y_train.tail(7).mean()] * len(y_val)
            else:
                baseline_pred = [y_train.mean()] * len(y_val)
            baseline_rmse = float(np.sqrt(mean_squared_error(y_val, baseline_pred)))
            baseline_scores.append(baseline_rmse)
            
            for name, model in self.models.items():
                if name == "loglog":
                    m = LogLogModel()
                elif name == "rf":
                    m = RFModel()
                else:
                    m = PoissonModel()
                
                try:
                    m.fit(X_train, y_train)
                    pred = m.predict(X_val)
                    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
                    if name == "loglog":
                        loglog_betas.append(m.get_elasticity(self.feature_cols))
                except (ValueError, RuntimeError):
                    rmse = float("inf")
                scores[name].append(rmse)

        avg_scores = {name: float(np.mean(vals)) for name, vals in scores.items()}
        
        # Сохраняем baseline_rmse в performance_info (ТЗ 7.2)
        if baseline_scores:
            self.performance_info["baseline_rmse"] = float(np.mean(baseline_scores))
        else:
            self.performance_info["baseline_rmse"] = 0.0
            
        # Находим лучший RMSE, исключая бесконечные значения
        valid_scores = [score for score in avg_scores.values() if score != float("inf")]
        if valid_scores:
            self.performance_info["best_rmse"] = float(min(valid_scores))
        else:
            self.performance_info["best_rmse"] = float("inf")
            
        # Расчитываем improvement vs baseline
        if (self.performance_info["baseline_rmse"] > 0 and 
            self.performance_info["best_rmse"] != float("inf")):
            improvement = 1 - self.performance_info["best_rmse"] / self.performance_info["baseline_rmse"]
            self.performance_info["improvement_vs_baseline"] = float(improvement)
        else:
            self.performance_info["improvement_vs_baseline"] = 0.0
        
        # Качество по LogLog фолдам
        if loglog_betas:
            self.elasticity_info["loglog_cv_med"] = float(np.median(loglog_betas))
            q75, q25 = np.percentile(loglog_betas, [75, 25])
            self.elasticity_info["loglog_cv_iqr"] = float(q75 - q25)

        return avg_scores

    def fit(self, df: pd.DataFrame, n_splits: int = 3) -> str:
        """Train models with CV and select best by RMSE. Returns best model name."""
        
        # Полная очистка состояния (убрать кэш прошлых запусков)
        self._reset_state()
        
        # Сохраняем результат fit для информативных ошибок
        self._fit_return_value = None
        
        try:
            # Логирование всегда пишется, независимо от обучения (ТЗ)
            self._add_pipeline_step("raw_rows", df)
            
            # Обработка SKU в признаках в зависимости от количества SKU
            if "sku" in df.columns:
                unique_skus = df["sku"].nunique()
                # Сохраняем информацию о SKU для debug
                self.elasticity_info["unique_sku"] = unique_skus
                
                if unique_skus == 1:
                    # Если один SKU - удаляем sku из признаков
                    self.feature_cols = [col for col in self.feature_cols if col != "sku"]
                    self._add_pipeline_step("sku_processing", df, "ok", f"Single SKU detected: removed 'sku' from features")
                else:
                    # Если множество SKU - оставляем для one-hot/target encoding
                    self._add_pipeline_step("sku_processing", df, "ok", f"Multiple SKUs detected: {unique_skus} SKUs, keeping 'sku' for encoding")
            else:
                # Если нет колонки SKU, считаем что один SKU
                self.elasticity_info["unique_sku"] = 1
            
            # Расчет качества данных и определение состояния
            self._calculate_quality_metrics(df)
            
            # Gating: если данные недостаточны, прерываем обучение
            if self.data_state in ["TOO_SMALL", "NO_PRICE_VARIATION", "EMPTY"]:
                self._add_pipeline_step("training_stopped", pd.DataFrame(), "failed", f"Data state: {self.data_state}")
                
                # Устанавливаем best_model_name для предотвращения "Model not trained"
                self.best_model_name = "NO_MODEL"  # Явное значение для gating
                
                # Сохраняем результат fit
                self._fit_return_value = "NO_MODEL"
                
                # Формируем model_result перед выходом (ТЗ)
                self.model_result = {
                    "elasticity": {},
                    "quality": self.quality_info,
                    "stability_mode": "S2",         # или вычислить по правилам, но не UNKNOWN
                    "monotonicity_flag": "N/A",
                    "improvement_vs_baseline": 0.0,
                    "protective_mode": "scenario",
                    "performance": {},
                    "model_name": None,  # Явно указываем что модели нет
                    "features_used": self.feature_cols,
                    "data_state": self.data_state,
                    "error": self.error,
                    "pipeline_log": self.pipeline_log
                }
                
                return "NO_MODEL"
            
            self._add_pipeline_step("data_quality_ok", df)
            scores = self.cross_validate(df, n_splits=n_splits)
            self.best_model_name = min(scores.items(), key=lambda x: x[1])[0]
            
            # Сохраняем информацию о выборе модели
            best_score = scores[self.best_model_name]
            model_comparison = {name: score for name, score in scores.items()}
            self.performance_info["model_selection"] = {
                "chosen_model": self.best_model_name,
                "chosen_score": best_score,
                "all_scores": model_comparison,
                "selection_reason": self._get_model_selection_reason(scores, self.best_model_name)
            }

            # 2. ОБУЧАЕМ выбранную модель на полных данных
            X, y = self._prepare_xy(df)
            
            # 3. Логирование финальных данных для модели
            combined_data = X.copy()
            combined_data['orders'] = y
            self._add_pipeline_step("final_rows_for_model_rows", combined_data)
            
            if self.best_model_name == "loglog":
                self.models["loglog"].fit(X, y)
            elif self.best_model_name == "rf":
                self.models["rf"].fit(X, y)
            else:
                self.models["poisson"].fit(X, y)

            # 2. Расчет профилей рекламы и эластичности для каждого профиля (ТЗ 6.6)
            ad_profiles = self._calculate_ad_profiles(df)
            self.elasticity_info["ad_profiles"] = ad_profiles
            
            # Расчет эластичности для каждого профиля рекламы (универсально)
            if ad_profiles is not None:
                profile_names = ["low", "med", "high"]
                for profile_name in profile_names:
                    if profile_name in ad_profiles and ad_profiles.get("available_features"):
                        profile_data = {k: v for k, v in ad_profiles[profile_name].items() 
                                      if k in ad_profiles["available_features"]}
                        num_e = self._calculate_numerical_elasticity(df, ad_profile=profile_data)
                        
                        # Сохраняем с префиксом профиля
                        for key, value in num_e.items():
                            if key not in ["ad_profiles"]:  # Избегаем рекурсии
                                self.elasticity_info[f"{profile_name}_{key}"] = value
                    else:
                        # Если профиль недоступен, заполняем None
                        for key in ["elasticity_med", "elasticity_iqr", "mono_violations"]:
                            self.elasticity_info[f"{profile_name}_{key}"] = None
            
            # Основная эластичность - по med профилю
            if "med" in ad_profiles and ad_profiles.get("available_features"):
                med_profile = {k: v for k, v in ad_profiles["med"].items() 
                              if k in ad_profiles["available_features"]}
                med_e = self._calculate_numerical_elasticity(df, ad_profile=med_profile)
            else:
                med_e = self._calculate_numerical_elasticity(df)
            
            self.elasticity_info.update(med_e)
            
            # 3. Определение режимов (ТЗ 5)
            iqr = med_e.get("elasticity_iqr", 0)
            if iqr is None:
                iqr = 0.0  # Default если None
                
            if iqr <= 0.3:
                self.stability_mode = "S1"
            elif iqr <= 0.7:
                self.stability_mode = "S2"
            else:
                self.stability_mode = "S3"
                
            # Монотонность (используем mono_violations из q_grid_used)
            if med_e["mono_violations"] > 0.2:
                self.monotonicity_flag = "non_monotone"
            else:
                self.monotonicity_flag = "monotone"

            # 4. Protective Mode - уточненная логика с приоритетами (ТЗ 7.2)
            improvement = self.performance_info.get("improvement_vs_baseline", 0)
            data_ok = self.quality_info.get("data_ok", False)
            mono_v = med_e["mono_violations"]
            stability = self.stability_mode
            mono_flag = "non_monotone" if mono_v > 0.2 else "monotone"
            
            # Сохраняем логику для отладки
            self.elasticity_info["protective_logic"] = {
                "improvement": improvement,
                "data_ok": data_ok,
                "stability_mode": stability,
                "monotonicity_flag": mono_flag,
                "mono_violations": mono_v
            }
            
            # Правило приоритета с порогами доверия (обновлено)
            if not data_ok or improvement < 0.05:
                # Приоритет 1: плохие данные или improvement < 5%
                self.protective_mode = "scenario"
                reason = f"scenario (improvement={improvement:.3f} < 0.05 or data_ok={data_ok})"
            elif improvement < 0.10:
                # Приоритет 2: improvement 5-10% → conservative
                self.protective_mode = "conservative"
                reason = f"conservative (improvement={improvement:.3f} in 0.05-0.10 range)"
            elif stability == "S3" or mono_flag == "non_monotone":
                # Приоритет 3: S3 или немонотонность (только если improvement >= 10%)
                self.protective_mode = "conservative"
                reason = f"conservative (stability={stability} or monotonicity={mono_flag}, improvement={improvement:.3f} >= 0.10)"
            elif stability == "S1" and mono_flag == "monotone":
                # Приоритет 4: хорошие условия + improvement >= 10%
                self.protective_mode = None
                reason = f"normal (S1 + monotone + improvement={improvement:.3f} >= 0.10)"
            else:
                # S2 + монотонность - консервативный режим (только если improvement >= 10%)
                self.protective_mode = "conservative"
                reason = f"conservative (S2 + monotone, improvement={improvement:.3f} >= 0.10)"
            
            self.elasticity_info["protective_logic"]["reason"] = reason

            # Sanity Check (ТЗ 4.1)
            corr = self.quality_info.get("corr", 0)
            if corr < -0.2 and abs(med_e["elasticity_med"]) < 0.05:
                self.quality_info["sanity_warning"] = True

            # 5. Формируем единый объект результата (ТЗ 1.1)
            self.model_result = {
                "elasticity": self.elasticity_info,
                "quality": self.quality_info,
                "stability_mode": self.stability_mode,
                "monotonicity_flag": self.monotonicity_flag,
                "improvement_vs_baseline": self.performance_info.get("improvement_vs_baseline", 0),
                "protective_mode": self.protective_mode,
                "performance": self.performance_info,
                "model_name": self.best_model_name,
                "features_used": self.feature_cols,
                "data_state": self.data_state,  # Состояние данных для gating UI
                "error": self.error,  # Детальная информация об ошибках
                "pipeline_log": self.pipeline_log,  # Структурированные логи
                "unique_sku": self.elasticity_info.get("unique_sku", 1)  # Информация о SKU
            }

            # Сохраняем успешный результат fit
            self._fit_return_value = self.best_model_name

            return self.best_model_name
            
        except Exception as e:
            # Гарантируем запись лога об ошибке
            self._set_error("FIT_ERROR", f"Ошибка обучения модели: {str(e)}", "model_training", e)
            self._add_pipeline_step("model_training", pd.DataFrame(), "failed", f"Exception: {str(e)}")
            
            # Формируем результат с ошибкой
            self.model_result = {
                "elasticity": {},
                "quality": self.quality_info,
                "stability_mode": "S2",  # Не UNKNOWN, а осмысленное значение
                "monotonicity_flag": "monotone",
                "improvement_vs_baseline": 0,
                "protective_mode": "scenario",
                "performance": {},
                "model_name": None,
                "features_used": [],
                "data_state": self.data_state,
                "error": self.error,
                "pipeline_log": self.pipeline_log,
                "unique_sku": self.elasticity_info.get("unique_sku", 1)  # Информация о SKU
            }
            
            # Сохраняем результат fit с ошибкой
            self._fit_return_value = "FAILED"
            
            return "FAILED"  # Не raise, а return для сохранения UX/диагностики

    def calibrate_curve(self, prices: np.ndarray, preds: np.ndarray) -> np.ndarray:
        """Принудительная калибровка монотонности кривой спроса."""
        if len(prices) < 2:
            return preds
        
        # The policy_info and allow_pos logic is removed as per the new fit method.
        # The calibration logic should be updated based on the new stability/monotonicity flags if needed.
        # For now, keeping the original logic but noting the change in context.
        
        # If monotonicity_flag is "non_monotone" and stability_mode is not "S1" (data is good enough)
        # then we might want to enforce monotonicity.
        # For simplicity, let's assume if we need to calibrate, we enforce decreasing.
        
        # Жёсткая монотонность: спрос не должен расти при росте цены
        # Используем cummin (начиная с конца) или IsotonicRegression
        ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
        preds_calibrated = ir.fit_transform(prices, preds)
        return preds_calibrated

    def _get_model_selection_reason(self, scores: Dict[str, float], chosen_model: str) -> str:
        """Получить понятную причину выбора модели."""
        if not scores:
            return "Нет данных для сравнения моделей"
        
        # Сортируем модели по RMSE (чем меньше, тем лучше)
        sorted_models = sorted(scores.items(), key=lambda x: x[1])
        chosen_score = scores[chosen_model]
        
        # Если выбранная модель значительно лучше других
        if len(sorted_models) >= 2:
            second_best = sorted_models[1]
            score_diff = second_best[1] - chosen_score
            
            if score_diff > 0.1:  # Значимое преимущество
                return f"Наименьшая ошибка (RMSE={chosen_score:.3f}) - лучше {second_best[0]} на {score_diff:.3f}"
            elif score_diff > 0.01:  # Небольшое преимущество
                return f"Немного лучшая точность (RMSE={chosen_score:.3f}) - лучше {second_best[0]} на {score_diff:.3f}"
            else:  # Очень близкие результаты
                return f"Лучшая точность (RMSE={chosen_score:.3f}) - модели очень близки"
        
        return f"Наименьшая ошибка (RMSE={chosen_score:.3f})"

    def get_model_result(self) -> Dict[str, Any]:
        """Получить единый объект результата модели (ТЗ 1.1)."""
        if not hasattr(self, 'model_result'):
            return {}
        result = self.model_result.copy()
        result["pipeline_log"] = self.pipeline_log.copy()  # Канонический атрибут
        # pipeline_logs можно добавить для совместимости если нужно
        return result

    def get_pipeline_logs(self) -> List[str]:
        """Получить логи пайплайна обработки данных (legacy view)."""
        # Возвращаем derived view из канонического pipeline_log
        logs = []
        for step in self.pipeline_log.get("steps", []):
            status_emoji = "✅" if step["status"] == "ok" else "❌"
            log_message = f"{status_emoji} {step['name']}: rows={step['rows']}, cols={step['cols']}"
            if step.get("notes"):
                log_message += f" - {step['notes']}"
            logs.append(log_message)
        return logs

    def get_info(self) -> Dict[str, Any]:
        """Информация о текущей лучшей модели и качестве."""
        if self.best_model_name is None:
            return {}
        info = {
            "model_name": self.best_model_name,
            "quality": self.quality_info,
            "elasticity": self.elasticity_info,
            "stability_mode": self.stability_mode,
            "monotonicity_flag": self.monotonicity_flag,
            "protective_mode": self.protective_mode,
            "performance": self.performance_info
        }
        if self.best_model_name == "loglog":
            info["actual_elasticity"] = self.models["loglog"].get_elasticity(self.feature_cols)
        
        # Добавляем причину выбора модели
        if "model_selection" in self.performance_info:
            info["model_selection_reason"] = self.performance_info["model_selection"]["selection_reason"]
        
        return info

    def predict_sales(self, price: float, features_row: Optional[Dict[str, Any]] = None) -> float:
        """Predict sales quantity for a given price and optional other features."""
        # Debug логирование перед первой попыткой прогноза (ТЗ)
        logger.debug("model_name: %s", self.best_model_name)
        logger.debug("data_state: %s", getattr(self, "data_state", "UNKNOWN"))
        logger.debug("fit_return: %s", getattr(self, "_fit_return_value", "UNKNOWN"))
        
        if self.best_model_name is None:
            # Собираем информативный контекст о состоянии модели
            context_parts = []
            
            # best_model_name
            context_parts.append(f"best_model_name={self.best_model_name}")
            
            # data_state
            data_state = getattr(self, 'data_state', 'UNKNOWN')
            context_parts.append(f"data_state={data_state}")
            
            # error информация
            error = getattr(self, 'error', {})
            if error.get('code'):
                context_parts.append(f"error.code={error['code']}")
            if error.get('message'):
                context_parts.append(f"error.message={error['message']}")
            if error.get('failed_step'):
                context_parts.append(f"failed_step={error['failed_step']}")
            
            # fit_return_value (если есть)
            fit_return = getattr(self, '_fit_return_value', None)
            if fit_return:
                context_parts.append(f"fit_return={fit_return}")
            
            context = ", ".join(context_parts)
            raise RuntimeError(f"Model not trained: {context}")
        
        if self.best_model_name == "NO_MODEL":
            # Прогноз недоступен из-за плохих данных
            data_state = getattr(self, 'data_state', 'UNKNOWN')
            raise RuntimeError(f"Прогноз недоступен: состояние данных - {data_state}. Улучшите качество данных для обучения модели.")

        # build feature vector
        row = {c: 0 for c in self.feature_cols}
        # set price to the primary feature column
        found_price = False
        for price_col in ["price_after_spp", "price_before_spp"]:
            if price_col in self.feature_cols:
                row[price_col] = price
        if features_row:
            for k, v in features_row.items():
                if k in row:
                    row[k] = v
        
        # 2. Только ПОТОМ заполняем недостающее значениями по умолчанию
        # Извлекаем признаки
        # df_feats = df.copy() # This line is not needed here, as we are building a single row
        # If 'price' is provided in features_row, ensure it's mapped to 'price_after_spp'
        if features_row and "price" in features_row and "price_after_spp" not in row:
            row["price_after_spp"] = features_row["price"]
        
        # day_of_week
        if "day_of_week" in self.feature_cols and row.get("day_of_week") is None:
            import datetime
            row["day_of_week"] = datetime.datetime.now().weekday()
        
        # set default ad spend to 0 if not provided
        for ad_col in ["ad_internal", "ad_bloggers", "ad_vk"]:
            if ad_col in self.feature_cols and row.get(ad_col) is None:
                row[ad_col] = 0.0

        # Гарантируем наличие всех колонок и их порядок
        for col in self.feature_cols:
            if col not in row:
                row[col] = 0.0
        
        Xnew = pd.DataFrame([row])[self.feature_cols]
        model = self.models[self.best_model_name]
        # Важно: predict у модельного обертки (напр. LogLogModel) 
        # должен сам вызывать трансформацию.
        pred = model.predict(Xnew)[0]
        return float(max(0.0, pred))

    def predict_on_df(self, df: pd.DataFrame) -> pd.Series:
        """Получить прогнозы для всего датафрейма (для диагностики)."""
        if self.best_model_name is None:
            return pd.Series([0.0] * len(df))
        
        if self.best_model_name == "NO_MODEL":
            # Прогноз недоступен из-за плохих данных
            return pd.Series([0.0] * len(df))
        
        X, _ = self._prepare_xy(df)
        model = self.models[self.best_model_name]
        return pd.Series(model.predict(X), index=df.index)


__all__ = ["SalesForecaster", "LogLogModel", "RFModel", "PoissonModel"]
