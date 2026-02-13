import io
from typing import List
import os
import sys
import logging

# ensure project root is on sys.path so `from src...` imports work when
# Streamlit runs the app from the `ui/` folder
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import plotly.express as px
from src.models.sales_forecast import SalesForecaster
from src.optimizer.bruteforce import optimize_price
from src.ingestion.excel_parser import ExcelIngestor, DEFAULT_COLUMNS, ValidationError
import json

logger = logging.getLogger(__name__)

# --- Уровень отладки ---
DEBUG_LEVELS = ["off", "summary", "full"]
DEFAULT_DEBUG_LEVEL = "summary"

def get_debug_level():
    """Получить текущий уровень отладки из session state или использовать default"""
    return st.session_state.get("debug_level", DEFAULT_DEBUG_LEVEL)

def set_debug_level(level):
    """Установить уровень отладки в session state"""
    st.session_state["debug_level"] = level

def create_debug_summary(model_result, sf):
    """Создать краткую выжимку для debug_summary"""
    if not model_result:
        return None
    
    # Базовая информация
    pipeline_log = model_result.get("pipeline_log", {})
    run_id = pipeline_log.get("run_id", "N/A")
    
    # Статус
    data_state = model_result.get("data_state", "UNKNOWN")
    status = "OK" if data_state == "OK" else data_state
    
    # Ошибка
    error = model_result.get("error", {})
    failed_step = error.get("failed_step", "")
    error_code = error.get("code", "")
    error_msg = error.get("message", "")
    
    # Данные
    final_step = None
    if pipeline_log.get("steps"):
        final_step = pipeline_log["steps"][-1]
    rows_final = final_step.get("rows", 0) if final_step else 0
    
    # SKU информация
    unique_sku = model_result.get("unique_sku", 1)  # Будет заполнено в SalesForecaster
    sku_mode = "single" if unique_sku == 1 else "multi"
    
    # Модель
    model_name = model_result.get("model_name", "None")
    features = model_result.get("features_used", [])
    features_str = ", ".join(features) if features else "None"
    
    # Метрики
    improvement = model_result.get("improvement_vs_baseline", 0)
    improvement_str = f"+{improvement:.3f}%" if improvement > 0 else f"{improvement:.3f}%"
    
    # Эластичность
    elasticity = model_result.get("elasticity", {})
    elastic_med = elasticity.get("elasticity_med", 0)
    elastic_iqr = elasticity.get("elasticity_iqr", 0)
    elastic_str = f"{elastic_med:.2f} (IQR {elastic_iqr:.2f})" if elastic_med != 0 else "N/A"
    
    # Монотонность
    mono_violations = elasticity.get("mono_violations", 0)
    mono_str = f"{mono_violations:.1f}%" if mono_violations > 0 else "0%"
    
    # Protective mode
    protective_mode = model_result.get("protective_mode", "None") or "None"
    reason = ""
    if protective_mode == "scenario":
        reason = " (reason: training failed)"
    elif protective_mode == "protective":
        reason = " (reason: unstable zone)"
    
    # Формируем summary строку
    summary_parts = []
    summary_parts.append(f"run_id={run_id}")
    summary_parts.append(f"status={status}")
    
    if failed_step:
        summary_parts.append(f"step={failed_step}")
    
    if error_code:
        summary_parts.append(f"error={error_code} ({error_msg[:50]}...)" if len(error_msg) > 50 else f"error={error_code} ({error_msg})")
    
    summary_parts.append(f"data_state={data_state}")
    summary_parts.append(f"rows_final={rows_final}")
    summary_parts.append(f"unique_sku={unique_sku}")
    summary_parts.append(f"sku_mode={sku_mode}")
    summary_parts.append(f"features={features_str}")
    summary_parts.append(f"model={model_name}")
    summary_parts.append(f"improvement={improvement_str}")
    summary_parts.append(f"elasticity={elastic_str}")
    summary_parts.append(f"mono={mono_str}")
    summary_parts.append(f"mode={protective_mode}{reason}")
    
    return {
        "summary": " | ".join(summary_parts),
        "run_id": run_id,
        "status": status,
        "failed_step": failed_step,
        "error_code": error_code,
        "error_message": error_msg,
        "data_state": data_state,
        "rows_final": rows_final,
        "unique_sku": unique_sku,
        "sku_mode": sku_mode,
        "features": features_str,
        "model": model_name,
        "improvement": improvement_str,
        "improvement_value": improvement,  # Для gating
        "elasticity": elastic_str,
        "monotonicity": mono_str,
        "protective_mode": protective_mode + reason
    }

def create_debug_full(model_result, sf):
    """Создать полную отладочную информацию"""
    if not model_result:
        return None
    
    return {
        "model_result": model_result,
        "debug_info": {
            "best_model_name": getattr(sf, 'best_model_name', 'NOT_FOUND'),
            "data_state": getattr(sf, 'data_state', 'UNKNOWN'),
            "fit_return_value": getattr(sf, '_fit_return_value', 'UNKNOWN'),
            "error": getattr(sf, 'error', {}),
            "quality_info": getattr(sf, 'quality_info', {}),
            "elasticity_info": getattr(sf, 'elasticity_info', {}),
            "performance_info": getattr(sf, 'performance_info', {}),
            "feature_cols": getattr(sf, 'feature_cols', []),
            "models": list(getattr(sf, 'models', {}).keys())
        }
    }

st.set_page_config(page_title="Best Price Optimizer", layout="wide")


def read_uploaded_files(uploaded_files: List) -> List:
    # return list of file-like objects accepted by pandas
    return [io.BytesIO(f.read()) for f in uploaded_files]


def main():
    st.title("Best Price — оптимизация цены")

    st.sidebar.header("1. Загрузите данные")
    uploaded = st.sidebar.file_uploader(
        "Excel файлы (несколько)", accept_multiple_files=True, type=["xlsx", "xls"]
    )

    ing = ExcelIngestor(strict=False)
    df = None
    report = None
    if uploaded:
        file_objs = read_uploaded_files(uploaded)
        try:
            df, report = ing.load_files(file_objs)
        except ValidationError as e:
            st.sidebar.error(f"Ошибка валидации данных: {e}")
            return

        if report.get("errors"):
            st.sidebar.error("; ".join(report.get("errors")))
        if report.get("warnings"):
            st.sidebar.warning("; ".join(report.get("warnings")))
        st.sidebar.success(f"Загружено строк: {len(df)}")

    st.sidebar.header("2. Уровень отладки")
    debug_level = st.sidebar.selectbox(
        "Детализация логов",
        options=DEBUG_LEVELS,
        index=DEBUG_LEVELS.index(get_debug_level()),
        format_func=lambda x: {
            "off": "🚫 Выкл (без логов)",
            "summary": "📋 Кратко (основное)",
            "full": "🔍 Подробно (всё)"
        }.get(x, x)
    )
    set_debug_level(debug_level)

    st.sidebar.header("3. Параметры рынка")
    commission_pct = st.sidebar.number_input("Комиссия (%)", value=36.0, step=0.5)
    vat_pct = st.sidebar.number_input("НДС (%)", value=5.0, step=0.5)
    spp_pct = st.sidebar.number_input("СПП (%)", value=32.0, step=0.1)

    st.sidebar.header("4. Реклама (дневной бюджет)")
    ad_internal = st.sidebar.number_input("Реклама внут., ₽", value=0.0, step=100.0)
    ad_bloggers = st.sidebar.number_input("Реклама блогеры, ₽", value=0.0, step=100.0)
    ad_vk = st.sidebar.number_input("Реклама ВК, ₽", value=0.0, step=100.0)
    total_ad_spend = ad_internal + ad_bloggers + ad_vk

    if df is None:
        st.info("Загрузите хотя бы один Excel-файл в боковой панели.")
        st.markdown("**Ожидаемые колонки (пример):**")
        st.write(DEFAULT_COLUMNS)
        return

    # SKU selection
    skus = sorted(df["sku"].astype(str).unique())
    selected_sku = st.sidebar.selectbox("Выберите SKU", skus)

    sku_df = df[df["sku"].astype(str) == str(selected_sku)].copy()
    st.markdown(f"**Анализ SKU:** {selected_sku} — строк: {len(sku_df)}")

    st.header("Исходные данные (преобразованные)")
    st.dataframe(sku_df)

    st.sidebar.header("3. Настройки оптимизации")
    price_min = st.sidebar.number_input(
        "Мин. цена (до СПП)", value=float(sku_df["price_before_spp"].min()), format="%.2f"
    )
    price_max = st.sidebar.number_input(
        "Макс. цена (до СПП)", value=float(sku_df["price_before_spp"].max()), format="%.2f"
    )
    step = st.sidebar.number_input("Шаг перебора", value=10.0, format="%.2f")

    st.sidebar.subheader("Переменные расходы (если пусто — усреднить по данным)")
    use_mean_costs = st.sidebar.checkbox("Использовать средние из данных", value=True)
    if use_mean_costs:
        cogs = float(sku_df["cogs"].mean()) if "cogs" in sku_df.columns else 0.0
        logistics = float(sku_df["logistics"].mean()) if "logistics" in sku_df.columns else 0.0
        storage = float(sku_df["storage"].mean()) if "storage" in sku_df.columns else 0.0
    else:
        cogs = st.sidebar.number_input("COGS per unit", value=0.0, format="%.2f")
        logistics = st.sidebar.number_input("Logistics per unit", value=0.0, format="%.2f")
        storage = st.sidebar.number_input("Storage per unit", value=0.0, format="%.2f")

    if st.sidebar.button("Запустить оптимизацию"):
        with st.spinner("Тренируем модель спроса и перебираем цены..."):
            # train forecaster on sku data
            sf = SalesForecaster(
                feature_cols=["price_after_spp", "day_of_week", "ad_internal", "ad_bloggers", "ad_vk"], 
                time_col="date"
            )
            try:
                sf.fit(sku_df, n_splits=3)
            except (ValueError, RuntimeError):
                # fallback: fit without time col
                sf.fit(sku_df, n_splits=2)

            # run optimizer
            try:
                # Debug логирование перед оптимизацией (ТЗ)
                logger.debug("model_name: %s", sf.best_model_name)
                logger.debug("data_state: %s", getattr(sf, "data_state", "UNKNOWN"))
                logger.debug("fit_return: %s", getattr(sf, "_fit_return_value", "UNKNOWN"))
                
                # Проверка параметров перед вызовом optimize_price
                base_features = {
                    "ad_internal": ad_internal,
                    "ad_bloggers": ad_bloggers,
                    "ad_vk": ad_vk
                }
                
                # Детальная отладка прямо в UI
                print("=== UI DEBUG: Before optimize_price ===")
                print(f"sf type: {type(sf)}")
                print(f"sf.best_model_name: {getattr(sf, 'best_model_name', 'None')}")
                print(f"base_features: {base_features}")
                print(f"price_min: {price_min}, price_max: {price_max}, step: {step}")
                print(f"commission_rate: {commission_pct / 100.0}, vat_rate: {vat_pct / 100.0}, spp: {spp_pct / 100.0}")
                print(f"cogs: {cogs}, logistics: {logistics}, storage: {storage}")
                print(f"sku_df type: {type(sku_df)}")
                print(f"sku_df empty: {sku_df.empty}")
                print(f"sku_df columns: {list(sku_df.columns)}")
                
                # Проверка импорта optimize_price
                try:
                    from src.optimizer.bruteforce import optimize_price
                    print("DEBUG: optimize_price import successful")
                except ImportError as e:
                    print(f"ERROR: Cannot import optimize_price: {e}")
                    st.error(f"❌ Ошибка импорта optimize_price: {e}")
                    return
                
                logger.debug("Checking optimize_price parameters...")
                logger.debug("sf type: %s", type(sf))
                logger.debug("base_features: %s", base_features)
                logger.debug("price_min: %s, price_max: %s, step: %s", price_min, price_max, step)
                logger.debug("commission_rate: %s, vat_rate: %s, spp: %s", commission_pct / 100.0, vat_pct / 100.0, spp_pct / 100.0)
                logger.debug("cogs: %s, logistics: %s, storage: %s", cogs, logistics, storage)
                logger.debug("sku_df type: %s, empty: %s", type(sku_df), sku_df.empty if hasattr(sku_df, 'empty') else 'N/A')
                if hasattr(sku_df, 'columns'):
                    logger.debug("sku_df columns: %s", list(sku_df.columns))
                
                # Проверка наличия необходимых колонок
                required_cols = ["price_after_spp", "price_before_spp"]
                missing_cols = [col for col in required_cols if col not in sku_df.columns]
                if missing_cols:
                    st.error(f"❌ Отсутствуют необходимые колонки в данных: {missing_cols}")
                    return
                
                # Проверка что sku_df не пустой
                if sku_df.empty:
                    st.error("❌ DataFrame с данными пуст")
                    return
                
                print("=== UI DEBUG: Calling optimize_price ===")
                try:
                    results, best_info = optimize_price(
                        forecaster=sf,
                        base_features=base_features,
                        price_min=price_min,
                        price_max=price_max,
                        step=step,
                        commission_rate=commission_pct / 100.0,
                        vat_rate=vat_pct / 100.0,
                        spp=spp_pct / 100.0,
                        cogs=cogs,
                        logistics=logistics,
                        storage=storage,
                        hist_min=sku_df["price_after_spp"].min(),
                        hist_max=sku_df["price_after_spp"].max(),
                        sku_df=sku_df  # Передаем sku_df для расчета режимов и текущей прибыли
                    )
                    print("=== UI DEBUG: optimize_price completed successfully ===")
                except Exception as e:
                    print(f"=== UI DEBUG: optimize_price failed with error ===")
                    print(f"Error type: {type(e)}")
                    print(f"Error message: {str(e)}")
                    print(f"Error args: {e.args}")
                    import traceback
                    print(f"Full traceback:")
                    traceback.print_exc()
                    st.error(f"❌ Ошибка при оптимизации: {type(e).__name__}: {str(e)}")
                    return
            except RuntimeError as e:
                # Показываем информативное сообщение об ошибке
                st.error(f"🚫 **Ошибка оптимизации:** {str(e)}")
                
                # Показываем рекомендации если есть model_result
                if hasattr(sf, 'model_result'):
                    model_result = sf.get_model_result()
                    error = model_result.get("error", {})
                    if error.get("recommendations"):
                        st.markdown("### 💡 Что сделать:")
                        for i, rec in enumerate(error["recommendations"], 1):
                            st.write(f"{i}. {rec}")
                return
            
            # Save to session state
            st.session_state["sf"] = sf
            st.session_state["results"] = results
            st.session_state["best_info"] = best_info
            st.session_state["current_sku"] = selected_sku

    # Check if we have results in session state for the selected SKU
    if "results" in st.session_state and st.session_state.get("current_sku") == selected_sku:
        sf = st.session_state["sf"]
        results = st.session_state["results"]
        best_info = st.session_state["best_info"]

        st.success("Оптимизация завершена")

        # --- Расширенная диагностика данных и модели ---
        with st.expander("🛠 Расширенная диагностика качества"):
            # Получаем данные из модели
            if hasattr(sf, 'model_result'):
                model_result = sf.get_model_result()
                if not model_result:  # Пустой результат - настоящая ошибка
                    # Пытаемся получить информативное сообщение об ошибке
                    try:
                        # Попробуем вызвать predict_sales чтобы получить информативную ошибку
                        sf.predict_sales(100.0)
                    except RuntimeError as e:
                        st.error(f"🚫 **Модель не обучена:** {str(e)}")
                        return
                    except (ValueError, TypeError):
                        st.warning("Модель не обучена: неизвестная ошибка")
                        return
            else:
                st.warning("Модель не обучена: объект модели не найден")
                return
            
            # 1. Структурированные логи пайплайна (ТЗ 2) - МАКСИМАЛЬНО ПОДРОБНО
            st.markdown("### 🔍 Полные логи пайплайна обработки данных")
            pipeline_log = model_result.get("pipeline_log", {})
            
            if pipeline_log and pipeline_log.get("steps"):
                st.write(f"**Run ID:** `{pipeline_log.get('run_id', 'N/A')}`")
                
                # Создаем таблицу с детальной информацией
                log_data = []
                for i, step in enumerate(pipeline_log["steps"]):
                    status_emoji = "✅" if step["status"] == "ok" else "❌"
                    
                    # NaN counts
                    nan_counts = step.get("nan_counts", {})
                    nan_text = ", ".join([f"{k}:{v}" for k, v in nan_counts.items() if v > 0]) or "нет"
                    
                    # Период данных
                    period_text = "N/A"
                    if step.get("date_min") and step.get("date_max"):
                        period_text = f"{step['date_min']} → {step['date_max']}"
                    
                    # Добавляем в таблицу
                    log_data.append({
                        "№": i + 1,
                        "Шаг": f"{status_emoji} {step['name']}",
                        "Статус": step["status"],
                        "Строк": step["rows"],
                        "Колонки": step["cols"],
                        "NaN": nan_text,
                        "Период": period_text,
                        "Заметки": step.get("notes", "нет")
                    })
                
                # Отображаем таблицу
                st.dataframe(pd.DataFrame(log_data), use_container_width=True)
                
                # Статистика по статусам
                status_counts = {}
                for step in pipeline_log["steps"]:
                    status = step["status"]
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                st.markdown("#### 📊 Статистика по статусам")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Успешных шагов", status_counts.get("ok", 0))
                with col2:
                    st.metric("❌ Проваленных шагов", status_counts.get("failed", 0))
                with col3:
                    total_steps = len(pipeline_log["steps"])
                    success_rate = (status_counts.get("ok", 0) / total_steps * 100) if total_steps > 0 else 0
                    st.metric("📈 Успешных (%)", f"{success_rate:.1f}%")
                
                # Детальная информация по каждому шагу
                st.markdown("#### 🔍 Детальная информация по шагам")
                for i, step in enumerate(pipeline_log["steps"]):
                    with st.expander(f"Шаг {i+1}: {step['name']} ({step['status']})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Строк", step["rows"])
                            st.metric("Колонки", step["cols"])
                            st.metric("Статус", step["status"])
                        with col2:
                            # NaN counts детально
                            nan_counts = step.get("nan_counts", {})
                            if nan_counts:
                                st.write("**NaN по полям:**")
                                for field, count in nan_counts.items():
                                    if count > 0:
                                        st.write(f"  • {field}: {count}")
                            else:
                                st.write("**NaN:** нет")
                        
                        if step.get("date_min") and step.get("date_max"):
                            st.write(f"**Период:** {step['date_min']} → {step['date_max']}")
                        
                        if step.get("notes"):
                            st.info(f"📝 **Заметки:** {step['notes']}")
            else:
                st.warning("Логи пайплайна недоступны")
            
            # 2. Детальная информация об ошибках - МАКСИМАЛЬНО ПОДРОБНО
            st.markdown("### 🚨 Детальная информация об ошибках")
            error = model_result.get("error", {})
            
            if error.get("code"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Код ошибки", error.get("code", ""))
                    st.metric("Шаг", error.get("failed_step", ""))
                with col2:
                    st.metric("Тип исключения", error.get("exception_type", ""))
                    st.metric("Traceback ID", error.get("traceback_id", ""))
                with col3:
                    st.metric("Data State", model_result.get("data_state", "UNKNOWN"))
                    st.metric("Fit Return", getattr(sf, '_fit_return_value', 'UNKNOWN'))
                with col4:
                    st.metric("Best Model", model_result.get("model_name", "None"))
                    st.metric("Protective Mode", model_result.get("protective_mode", "None"))
                
                if error.get("message"):
                    st.error(f"**Сообщение:** {error['message']}")
                
                # Декларативные рекомендации
                recommendations = error.get("recommendations", [])
                if recommendations:
                    st.markdown("### 💡 Что сделать:")
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.info("Рекомендации недоступны")
            else:
                st.success("✅ Ошибок не обнаружено")
            
            # 3. Состояние модели и защитные режимы - МАКСИМАЛЬНО ПОДРОБНО
            st.markdown("### 🤖 Состояние модели и защитные режимы")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data State", model_result.get("data_state", "UNKNOWN"))
                st.metric("Best Model", model_result.get("model_name", "None"))
                st.metric("Protective Mode", model_result.get("protective_mode", "None"))
            with col2:
                st.metric("Stability Mode", model_result.get("stability_mode", "UNKNOWN"))
                st.metric("Monotonicity", model_result.get("monotonicity_flag", "UNKNOWN"))
                st.metric("Improvement", f"{model_result.get('improvement_vs_baseline', 0):.3f}")
            with col3:
                st.metric("Fit Return", getattr(sf, '_fit_return_value', 'UNKNOWN'))
                st.metric("Features Used", len(model_result.get("features_used", [])))
                st.metric("Elasticity Med", f"{model_result.get('elasticity', {}).get('elasticity_med', 0):.3f}")
            
            # Детальная информация о признаках
            features_used = model_result.get("features_used", [])
            if features_used:
                st.markdown("#### 🔧 Используемые признаки")
                st.write(", ".join(features_used))
            
            # Детальная информация об эластичности
            elasticity = model_result.get("elasticity", {})
            if elasticity:
                st.markdown("#### 📈 Детальная информация об эластичности")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Elasticity Med", f"{elasticity.get('elasticity_med', 0):.3f}")
                    st.metric("Elasticity IQR", f"{elasticity.get('elasticity_iqr', 0):.3f}")
                with col2:
                    st.metric("Beta Median", f"{elasticity.get('beta_median', 0):.3f}")
                    st.metric("Beta IQR", f"{elasticity.get('beta_iqr', 0):.3f}")
                with col3:
                    st.metric("Mono Violations", elasticity.get('mono_violations', 0))
                    st.metric("R Squared", f"{elasticity.get('r_squared', 0):.3f}")
                
                # Статистика локальной эластичности
                e_stats = elasticity.get('e_stats', {})
                if e_stats:
                    st.markdown("##### 📊 Статистика локальной эластичности")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("E Mean", f"{e_stats.get('mean', 0):.3f}")
                        st.metric("E Std", f"{e_stats.get('std', 0):.3f}")
                    with col2:
                        st.metric("E Min", f"{e_stats.get('min', 0):.3f}")
                        st.metric("E Max", f"{e_stats.get('max', 0):.3f}")
                    with col3:
                        st.metric("E 25%", f"{e_stats.get('q25', 0):.3f}")
                        st.metric("E 75%", f"{e_stats.get('q75', 0):.3f}")
                    with col4:
                        st.metric("Valid Points", e_stats.get('valid_points', 0))
                        st.metric("Total Points", e_stats.get('total_points', 0))
            
            # 4. Метрики качества - только при data_state == "OK" (ТЗ)
            data_state = model_result.get("data_state", "UNKNOWN")
            if data_state == "OK":
                st.markdown("### 📊 Метрики качества данных")
                q = model_result.get("quality", {})
                if q:
                    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                    col_q1.metric("Дней с данными", q.get("n_days", 0))
                    col_q2.metric("Уникальных цен", q.get("n_price_unique", 0))
                    col_q3.metric("Вариация цены (CV)", f"{q.get('price_cv', 0)*100:.1f}%")
                    col_q4.metric("Доля нулей", f"{q.get('zero_share', 0)*100:.0f}%")
                    
                    if q.get("data_ok"):
                        st.success("✅ Данных достаточно для обучения эластичности.")
                    else:
                        st.warning("⚠️ Данных критически мало или цена не менялась.")
                else:
                    st.info("Метрики качества недоступны")
            else:
                st.markdown("### ⚠️ Метрики качества недоступны")
                st.info(f"Метрики качества не доступны из-за состояния данных: {data_state}")
            
            # 5. Техническая информация (Debug) - МАКСИМАЛЬНО ПОДРОБНО
            st.markdown("### 🔍 Техническая информация (Debug)")
            debug_info = {
                "model_result": model_result,  # Единый источник данных
                "features_used": model_result.get("features_used", []),
                "pipeline_log": model_result.get("pipeline_log", []),  # Канонический атрибут
                "debug_info": {
                    "best_model_name": sf.best_model_name,
                    "data_state": getattr(sf, 'data_state', 'UNKNOWN'),
                    "fit_return_value": getattr(sf, '_fit_return_value', 'UNKNOWN'),
                    "error": getattr(sf, 'error', {}),
                    "quality_info": getattr(sf, 'quality_info', {}),
                    "elasticity_info": getattr(sf, 'elasticity_info', {}),
                    "performance_info": getattr(sf, 'performance_info', {}),
                    "feature_cols": getattr(sf, 'feature_cols', []),
                    "models": list(getattr(sf, 'models', {}).keys())
                }
            }
            st.json(debug_info)

            # 2. Модель и эластичность - ЖЕСТКИЙ GATING по data_state (ТЗ)
        if data_state != "OK":
            st.error(f"🚫 **Анализ недоступен:** состояние данных - {data_state}")
            
            # Показываем декларативные рекомендации если есть ошибка
            if error.get("code") and error.get("recommendations"):
                st.markdown("### 💡 Что сделать:")
                for i, rec in enumerate(error["recommendations"], 1):
                    st.write(f"{i}. {rec}")
            else:
                # Общие рекомендации для состояний без кода ошибки
                st.info("💡 **Общие рекомендации:**")
                if data_state == "TOO_SMALL":
                    st.write("• Увеличьте период наблюдений (минимум 7 дней)")
                    st.write("• Добавьте больше уникальных цен (минимум 3)")
                elif data_state == "NO_PRICE_VARIATION":
                    st.write("• Проверьте корректность данных о заказах")
                    st.write("• Убедитесь что цена варьируется (CV > 1%)")
                    st.write("• Снизьте долю нулевых заказов (< 80%)")
                elif data_state == "EMPTY":
                    st.write("• Проверьте наличие и формат входных данных")
                else:
                    st.write("• Проверьте наличие и качество входных данных")
            return


        # Diagnostic info
        info = sf.get_info()
        st.info(f"Используемая модель: **{info.get('model_name')}**")
        
        # Причина выбора модели
        if info.get("model_selection_reason"):
            st.caption(f"🎯 **Причина выбора:** {info['model_selection_reason']}")
        
        st.write(f"Используемые признаки: `{sf.feature_cols}`")
        if "elasticity" in info:
            e_data = info["elasticity"]
            e_val = e_data.get("elasticity_med", 0)  # Исправляем на новый ключ
            e_iqr = e_data.get("elasticity_iqr", 0)  # Исправляем на новый ключ
            
            # Категоризация эластичности (с допуском tol=0.05)
            tol = 0.05
            if e_val < -1.0 - tol:
                e_cat = "эластичный"
                e_color = "green"
            elif abs(e_val + 1.0) <= tol:
                e_cat = "пограничный (около -1)"
                e_color = "orange"
            else:
                e_cat = "неэластичный"
                e_color = "blue"
            
            st.markdown(f"**Характер спроса:** :{e_color}[{e_cat}]")
            st.write(f"Эластичность (med): **{e_val:.3f}** (IQR: {e_iqr:.2f})")
            
            if e_iqr > 0.4:
                st.warning("⚠️ **Внимание: Оценка эластичности нестабильна (высокий IQR).** Рекомендуется полагаться на анализ прибыли по сетке, а не на коэффициент.")


        # Historical context
        p_min_hist = sku_df["price_before_spp"].min()
        p_max_hist = sku_df["price_before_spp"].max()
        st.write(f"Исторический диапазон цен (до СПП): **{p_min_hist:.0f} — {p_max_hist:.0f} RUB**")

        # --- Отладочная информация с уровнями ---
        debug_level = get_debug_level()
        
        if debug_level != "off":
            # Получаем данные модели
            if "results" in st.session_state and st.session_state.get("current_sku") == selected_sku:
                sf = st.session_state["sf"]
                model_result = sf.get_model_result()
                
                if model_result:
                    # Создаем debug данные
                    debug_summary = create_debug_summary(model_result, sf)
                    debug_full = create_debug_full(model_result, sf)
                    
                    # Заголовок блока
                    if debug_level == "summary":
                        title = "📋 Краткая диагностика"
                    else:
                        title = "🔍 Подробная диагностика"
                    
                    with st.expander(title):
                        if debug_level == "summary":
                            # Summary режим
                            st.markdown("### 📋 Debug Summary")
                            if debug_summary:
                                st.code(debug_summary["summary"], language="text")
                                
                                # Дополнительная информация в виде метрик
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Статус", debug_summary["status"])
                                    st.metric("Run ID", debug_summary["run_id"][:8] + "...")
                                with col2:
                                    st.metric("Data State", debug_summary["data_state"])
                                    st.metric("Model", debug_summary["model"])
                                with col3:
                                    st.metric("Improvement", debug_summary["improvement"])
                                    st.metric("Elasticity", debug_summary["elasticity"])
                                
                                # Ошибка если есть
                                if debug_summary["error_code"]:
                                    st.error(f"**{debug_summary['error_code']}:** {debug_summary['error_message']}")
                            else:
                                st.warning("Debug summary недоступен")
                        
                        elif debug_level == "full":
                            # Full режим
                            st.markdown("### 📋 Debug Summary")
                            if debug_summary:
                                st.code(debug_summary["summary"], language="text")
                            
                            st.markdown("### 🔍 Полная отладочная информация")
                            if debug_full:
                                st.json(debug_full)
                            else:
                                st.warning("Полная отладочная информация недоступна")
                else:
                    st.warning("Модель не обучена - отладочная информация недоступна")
            else:
                st.info("Оптимизация еще не запускалась - отладочная информация недоступна")

        st.subheader("1. Анализ спроса и эластичности")
        
        # ⚠️ ВАЖНО: Запрещено пересчитывать метрики в UI! (ТЗ 1.2)
        # Использовать только данные из model_result
        # ❌ recalculate_improvement()
        # ❌ recalculate_stability() 
        # ✅ value = model_result["..."]
        
        # Sanity-check перед выводом UI (ТЗ 1.3)
        def sanity_check(model_result, ui_values):
            """Проверка согласованности данных модели и UI."""
            errors = []

            # Проверка improvement между блоками model/performance/UI
            model_improvement = model_result.get("improvement_vs_baseline", 0)
            performance_improvement = model_result.get("performance", {}).get("improvement_vs_baseline", 0)
            ui_improvement = ui_values.get("improvement", 0)
            if abs(model_improvement - performance_improvement) > 1e-6:
                errors.append(
                    f"Improvement mismatch: model={model_improvement:.6f}, performance={performance_improvement:.6f}"
                )
            if abs(model_improvement - ui_improvement) > 1e-6:
                errors.append(f"Improvement mismatch: model={model_improvement:.6f}, ui={ui_improvement:.6f}")

            # Проверка stability_mode
            model_stability = model_result.get("stability_mode", "")
            ui_stability = ui_values.get("stability", "")
            logic_stability = model_result.get("elasticity", {}).get("protective_logic", {}).get("stability_mode", model_stability)
            if model_stability != ui_stability:
                errors.append(f"Stability mismatch: model={model_stability}, ui={ui_stability}")
            if model_stability != logic_stability:
                errors.append(f"Stability mismatch: model={model_stability}, protective_logic={logic_stability}")

            # Проверка protective_mode
            model_protective = model_result.get("protective_mode", "")
            ui_protective = ui_values.get("protective", "")
            if model_protective != ui_protective:
                errors.append(f"Protective mode mismatch: model={model_protective}, ui={ui_protective}")

            # Проверка elasticity_med (включая None значения)
            elasticity_info = model_result.get("elasticity", {})
            model_elasticity = elasticity_info.get("elasticity_med", 0)
            ui_elasticity = ui_values.get("elasticity", 0)
            beta_median = elasticity_info.get("beta_median", model_elasticity)

            # Special case: оба None - это OK
            if model_elasticity is None and ui_elasticity is None:
                pass  # Несогласованности нет
            elif model_elasticity is None or ui_elasticity is None:
                errors.append(f"Elasticity mismatch: model={model_elasticity}, ui={ui_elasticity}")
            elif abs(model_elasticity - ui_elasticity) > 1e-6:
                errors.append(f"Elasticity mismatch: model={model_elasticity:.6f}, ui={ui_elasticity:.6f}")

            if model_elasticity is None and beta_median is not None:
                errors.append(f"Elasticity mismatch: elasticity_med={model_elasticity}, beta_median={beta_median}")
            elif model_elasticity is not None and beta_median is None:
                errors.append(f"Elasticity mismatch: elasticity_med={model_elasticity}, beta_median={beta_median}")
            elif model_elasticity is not None and abs(model_elasticity - beta_median) > 1e-6:
                errors.append(f"Elasticity mismatch: elasticity_med={model_elasticity:.6f}, beta_median={beta_median:.6f}")

            # Проверка monotonicity_flag
            model_monotonicity = model_result.get("monotonicity_flag", "")
            ui_monotonicity = ui_values.get("monotonicity", "")
            if model_monotonicity != ui_monotonicity:
                errors.append(f"Monotonicity mismatch: model={model_monotonicity}, ui={ui_monotonicity}")

            # Проверка валидных точек локальной эластичности
            e_grid = elasticity_info.get("e_grid", []) or []
            e_stats = elasticity_info.get("e_stats", {})
            valid_from_grid = sum(1 for e in e_grid if e is not None and not pd.isna(e))
            valid_reported = int(e_stats.get("valid_points", 0))
            total_reported = int(e_stats.get("total_points", 0))
            excluded_reported = int(e_stats.get("excluded_invalid_points", 0))
            if valid_from_grid != valid_reported:
                errors.append(f"Valid points mismatch: e_grid={valid_from_grid}, e_stats={valid_reported}")
            if total_reported != valid_reported + excluded_reported:
                errors.append(
                    f"Point accounting mismatch: total={total_reported}, valid={valid_reported}, excluded={excluded_reported}"
                )

            return errors
        
        # Извлекаем данные из единого результата (ТЗ 6.1)
        e_info = model_result.get("elasticity", {})
        q_info = model_result.get("quality", {})
        
        # Собираем UI значения для sanity-check
        ui_values = {
            "improvement": model_result.get("improvement_vs_baseline", 0),
            "stability": model_result.get("stability_mode", ""),
            "protective": model_result.get("protective_mode", ""),
            "elasticity": e_info.get("elasticity_med", 0),
            "monotonicity": model_result.get("monotonicity_flag", "")
        }
        
        # Выполняем sanity-check
        sanity_errors = sanity_check(model_result, ui_values)
        if sanity_errors:
            st.error("🚨 **Обнаружены несогласованности данных:**")
            for error in sanity_errors:
                st.error(f"• {error}")
            st.error("Пожалуйста, проверьте логику расчета метрик!")
                
            # Адаптивное форматирование (ТЗ 3.1)
            def fmt_e(val):
                if val is None: return "Н/Д"
                abs_v = abs(val)
                if abs_v < 0.1:
                    return f"{val:.4f}"
                else:
                    return f"{val:.2f}"

            e_med = e_info.get("elasticity_med", 0.0)
            e_iqr = e_info.get("elasticity_iqr", 0.0)
            
            # Обработка None значений
            if e_med is None:
                e_med_display = "Н/Д"
                e_iqr_display = "Н/Д"
            else:
                e_med_display = fmt_e(e_med)
                e_iqr_display = fmt_e(e_iqr)
            
            col_e1, col_e2, col_e3 = st.columns(3)
            col_e1.metric("Эластичность (median)", e_med_display)
            col_e2.metric("Разброс (IQR)", e_iqr_display)
            
            # Эластичность для разных профилей рекламы (ТЗ 6.6)
            ad_profiles = e_info.get("ad_profiles", {})
            available_features = ad_profiles.get("available_features", [])
            
            # Правильная проверка наличия профилей (ТЗ 3.1)
            if ad_profiles is None:
                col_e3.metric("Профили рекламы", "Нет данных")
            elif not available_features:
                col_e3.metric("Профили рекламы", "Нет данных")
            else:
                low_e = e_info.get("low_elasticity_med", 0)
                med_e = e_info.get("med_elasticity_med", 0)
                high_e = e_info.get("high_elasticity_med", 0)
                
                # Проверяем что профили построены корректно
                if low_e == 0 and med_e == 0 and high_e == 0:
                    col_e3.metric("Профили рекламы", "Недостаточно данных")
                else:
                    col_e3.metric("Разброс эластичности (ads)", f"{fmt_e(low_e)} - {fmt_e(high_e)}")
                    
                    # Детальная информация о профилях
                    with st.expander("📊 Профили рекламы"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Low эластичность", fmt_e(low_e))
                            st.write("**Low профиль:**")
                            for feature in available_features:
                                val = ad_profiles.get("low", {}).get(feature, 0)
                                st.write(f"{feature}: {val:.1f}")
                        
                        with col2:
                            st.metric("Med эластичность", fmt_e(med_e))
                            st.write("**Med профиль:**")
                            for feature in available_features:
                                val = ad_profiles.get("med", {}).get(feature, 0)
                                st.write(f"{feature}: {val:.1f}")
                        
                        with col3:
                            st.metric("High эластичность", fmt_e(high_e))
                            st.write("**High профиль:**")
                            for feature in available_features:
                                val = ad_profiles.get("high", {}).get(feature, 0)
                                st.write(f"{feature}: {val:.1f}")
                        
                        st.write(f"**Метод фиксации признаков:** {ad_profiles.get('method', 'N/A')}")
                        st.write(f"**Доступные признаки:** {', '.join(available_features)}")
            
            # Режимы стабильности и монотонность по ТЗ 5
            stability = model_result.get("stability_mode", "S1")
            st.markdown(f"**Режим стабильности:** `{stability}`")
            
            # Монотонность по используемой кривой (ТЗ 4.4)
            mono_v = e_info.get("mono_violations", 0.0) * 100
            mono_v_raw = e_info.get("mono_violations_raw", 0.0) * 100
            calibrated = e_info.get("calibrated", False)
            
            if mono_v <= 20:
                mono_text = "Почти монотонен"
                mono_emoji = "✅"
            else:
                mono_text = "Немонотонен"
                mono_emoji = "⚠️"
            
            st.markdown(f"**Монотонность спроса:** {mono_emoji} {mono_text} (нарушения: {mono_v:.1f}%)")
            
            # Информация о калибровке
            if calibrated:
                st.info(f"🔧 **Применена калибровка:** исходные нарушения {mono_v_raw:.1f}% → после калибровки {mono_v:.1f}%")
            elif mono_v_raw > 20:
                st.warning(f"⚠️ **Обнаружена немонотонность:** {mono_v_raw:.1f}% нарушений, но калибровка не применялась")

            # Отладочный режим (ТЗ 10)
            with st.expander("🔍 Техническая диагностика (Debug)"):
                # Формируем полный JSON лог по ТЗ 10 (используем model_result)
                debug_info = {
                    "model_result": model_result,  # Единый источник данных
                    "features_used": model_result.get("features_used", []),
                    "pipeline_log": model_result.get("pipeline_log", []),  # Канонический атрибут
                    "sanity_check": {
                        "errors": sanity_errors,
                        "status": "PASS" if not sanity_errors else "FAIL"
                    }
                }
                st.json(debug_info)
                
                st.write("**Средняя эластичность на диапазоне (наклон ln(q)~ln(p)):**")
                e_global = e_info.get("elasticity_med", 0)
                global_reg = e_info.get("global_regression", {})
                r2 = global_reg.get("r_squared")
                n_points = global_reg.get("n_points", 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Средняя эластичность", fmt_e(e_global))
                with col2:
                    if r2 is not None:
                        st.metric("R² регрессии", f"{r2:.3f}")
                    else:
                        st.metric("R² регрессии", "N/A")
                with col3:
                    st.metric("Точек регрессии", f"{n_points}")
                    
                st.write("**Predicted Orders (grid):**")
                st.line_chart(e_info.get("q_grid", []))
                
                # Локальная эластичность по сетке
                e_grid = e_info.get("e_grid", [])
                e_stats = e_info.get("e_stats", {})
                q_grid_used = e_info.get("q_grid", [])
                
                # Проверяем количество точек по q_grid_used (ТЗ)
                if len(q_grid_used) < 5:
                    st.warning("⚠ Недостаточно данных для расчета локальной эластичности (len(q_grid_used) < 5)")
                elif e_grid and len(e_grid) > 0 and e_stats.get("len", 0) > 0:
                    st.write("**Локальная эластичность по сетке:**")
                    
                    # Фильтруем NaN для визуализации
                    e_grid_clean = [e for e in e_grid if not np.isnan(e)]
                    if e_grid_clean:
                        st.line_chart(e_grid_clean)
                    
                    # Статистика локальной эластичности
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Min", fmt_e(e_stats.get("min", 0)))
                    with col2:
                        st.metric("Median", fmt_e(e_stats.get("median", 0)))
                    with col3:
                        st.metric("Max", fmt_e(e_stats.get("max", 0)))
                    with col4:
                        st.metric("Std", f"{e_stats.get('std', 0):.3f}")
                    
                    # Проверка на баг "копирования" (ТЗ 2.3)
                    if len(e_grid_clean) > 1:
                        all_same = all(abs(e - e_grid_clean[0]) < 1e-10 for e in e_grid_clean)
                        if all_same:
                            st.error("⚠️ Обнаружен баг: все значения локальной эластичности одинаковы!")
                        else:
                            std_val = e_stats.get('std', 0)
                            if std_val > 0:
                                st.success(f"✅ Локальная эластичность варьируется (std = {std_val:.3f})")
                            else:
                                st.warning("⚠️ Стандартное отклонение равно 0 (возможно, идеальная степенная функция)")
                    
                    st.write(f"**Длина e_grid:** {e_stats.get('len', 0)} точек")
                    st.write(
                        f"**Доля нулевых локальных эластичностей:** {e_stats.get('zero_share', 0.0):.1%}"
                    )
                    st.write(
                        f"**Исключено невалидных точек:** {e_stats.get('excluded_invalid_points', 0)}"
                    )
                else:
                    st.warning("⚠ Недостаточно данных для расчета локальной эластичности")
                
                # Показываем кривые спроса
                st.write("**Кривые спроса:**")
                
                # Основная кривая (используется в расчетах)
                st.line_chart(e_info.get("q_grid", []))
                
                # Дополнительные кривые если есть калибровка
                q_grid_raw = e_info.get("q_grid_raw", [])
                q_grid_calibrated = e_info.get("q_grid_calibrated", [])
                calibrated = e_info.get("calibrated", False)
                
                if calibrated and q_grid_calibrated:
                    st.write("**Сравнение кривых (сырая → калиброванная):**")
                    comparison_df = pd.DataFrame({
                        'Raw': q_grid_raw,
                        'Calibrated': q_grid_calibrated,
                        'Used': e_info.get("q_grid", [])
                    })
                    st.line_chart(comparison_df)
                elif q_grid_raw and len(q_grid_raw) > 0:
                    st.write("**Сырая кривая (калибровка не применялась):**")
                    st.line_chart(q_grid_raw)
                # RMSE vs Baseline (используем данные из model_result)
                improvement = model_result.get("improvement_vs_baseline", 0)
                st.write(f"🏆 Точность vs Baseline: **{improvement:+.1f}%**")
                
            # Проверка разрешения положительной эластичности (ТЗ 3.1) - строгие критерии
            # Новое условие: сообщение только если эластичность реально положительная и стабильная
            if e_med is not None and e_med > 0.1 and e_iqr < 0.3 and q_info.get("corr", 0) > 0.1:
                st.info("💡 **Разрешена положительная эластичность:** эластичность > 0.1, IQR < 0.3, корреляция > 0.1. Сигнал качества/сезона.")
            # Во всех остальных случаях сообщение не показывается (ТЗ)
            
            # 3. Режимы стабильности и защитные режимов по ТЗ 7.2 - используем только model_result (ТЗ 6.1)
            stability = model_result.get("stability_mode", "S1")
            protective = model_result.get("protective_mode")
            protective_logic = e_info.get("protective_logic", {})
            
            # Режим стабильности
            if stability == "S1":
                st.success(f"✅ **Стабильный спрос**: разрешён широкий поиск в пределах истории.")
            elif stability == "S2":
                st.warning(f"⚠️ **Умеренно нестабилен**: оптимизация ограничена историческим диапазоном цен.")
            else:  # S3
                st.warning(f"🛡️ **Нестабилен**: включён консервативный режим (локальный поиск/штраф/сценарии).")
            
            # Защитный режим
            if protective == "scenario":
                st.error(f"🚫 **Режим сценарного анализа:** {protective_logic.get('reason', 'Причина неизвестна')}")
            elif protective == "conservative":
                st.warning(f"⚠️ **Консервативный режим:** {protective_logic.get('reason', 'Причина неизвестна')}")
            else:
                st.success(f"✅ **Стандартная оптимизация:** {protective_logic.get('reason', 'Хорошие условия модели')}")
                
            # Детальная информация о принятии решения (только из model_result)
            with st.expander("🔍 Логика защитных режимов"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Improvement vs Baseline", f"{model_result.get('improvement_vs_baseline', 0):.3f}")
                    st.metric("Data Quality", "✅ Хорошо" if model_result.get('quality', {}).get('data_ok', False) else "❌ Плохо")
                with col2:
                    st.metric("Стабильность", model_result.get('stability_mode', 'N/A'))
                    st.metric("Монотонность", model_result.get('monotonicity_flag', 'N/A'))
                
                st.write(f"**Причина режима:** {protective_logic.get('reason', 'N/A')}")
                st.write(f"**Нарушения монотонности:** {protective_logic.get('mono_violations', 0):.1%}")
            
            # 4. Оптимизация или сценарный анализ (только из model_result)
            protective = model_result.get("protective_mode")
            if protective != "scenario":
                p_min_allowed = results["price_before_spp"].min()
                p_max_allowed = results["price_before_spp"].max()
                st.caption(f"📏 Допустимый диапазон (до СПП): {p_min_allowed:.0f} – {p_max_allowed:.0f} ₽")
            else:
                st.info("💡 **Режим сценарного анализа:** модель недостаточно надежна для автоматического поиска оптимума. Оцените варианты изменения цены вручную.")

        # --- Результаты сравнения с последней ценой ---
        # ЖЕСТКИЙ GATING: оптимизация только при OK состоянии данных
        if data_state != "OK":
            st.error("🚫 **Оптимизация недоступна:** состояние данных не позволяет провести анализ")
            return
        
        # Gating по improvement (порог доверия)
        improvement = model_result.get("improvement_vs_baseline", 0)
        if improvement < 0.05:
            st.warning("⚠️ **Низкое доверие к модели:** improvement < 5%")
            st.info("💡 Показаны только сценарные расчеты. Точный оптимум недоступен.")
            show_scenario_only = True
        elif improvement < 0.10:
            st.warning("⚠️ **Умеренное доверие к модели:** improvement 5-10%")
            st.info("💡 Рекомендуется использовать сценарный анализ вместо точного оптимума.")
            show_scenario_only = False  # Оставляем выбор пользователю
        else:
            show_scenario_only = False
        
        st.subheader("📊 Итоги оптимизации")
        
        # Данные за последнюю дату
        last_row = sku_df.sort_values("date").iloc[-1]
        last_p_before = float(last_row["price_before_spp"])
        
        # Расчет маржи для последней цены (по текущим правилам)
        p_last = last_p_before
        s_val = spp_pct / 100.0
        p_after_last = p_last * (1.0 - s_val)
        comm_last = p_last * (commission_pct / 100.0)
        vat_last = p_after_last * (vat_pct / 100.0)
        margin_last = p_last - comm_last - vat_last - cogs - logistics - storage

        col1, col2, col3 = st.columns(3)
        
        if best_info:
            # Расчет прогнозов для текущей цены
            q_last = float(sf.predict_sales(p_after_last, {
                "ad_internal": ad_internal,
                "ad_bloggers": ad_bloggers,
                "ad_vk": ad_vk
            }))
            # unit_margin считаем без рекламы (уже посчитано в margin_last)
            # profit_last вычитаем рекламу
            profit_last = (q_last * margin_last) - total_ad_spend
            profitability_last = (margin_last / last_p_before) * 100
            
            is_boundary_search = best_info.get("is_boundary_search", best_info.get("is_boundary", False))
            is_boundary_history = best_info.get("is_boundary_history", False)
            
            # Логика защитных режимов и gating по improvement
            protective = model_result.get("protective_mode")
            
            # Получаем значения эластичности для проверок
            e_med = e_info.get("elasticity_med", 0.0)
            e_iqr = e_info.get("elasticity_iqr", 0.0)
            
            # Дополнительное ограничение для положительной эластичности (ТЗ 3.3)
            # Используем те же критерии что и в основном блоке
            allow_positive_elasticity = (
                e_med is not None and e_med > 0.1 and 
                e_iqr < 0.3 and 
                q_info.get("corr", 0) > 0.1
            )
            
            # Объединенная логика show_scenario_only
            if protective == "scenario" or improvement < 0.05:
                # Приоритет 1: scenario режим OR improvement < 5%
                show_scenario_only = True
                if improvement < 0.05:
                    st.warning("⚠️ **Низкое доверие к модели:** improvement < 5%")
                    st.info("💡 Показаны только сценарные расчеты. Точный оптимум недоступен.")
                else:
                    st.info("💡 **Режим сценарного анализа:** модель недостаточно надежна для автоматического поиска оптимума.")
            elif improvement < 0.10:
                # Приоритет 2: improvement 5-10% → сценарии с выбором
                show_scenario_only = False
                st.warning("⚠️ **Умеренное доверие к модели:** improvement 5-10%")
                st.info("💡 Рекомендуется использовать сценарный анализ вместо точного оптимума.")
            elif protective == "conservative":
                # Приоритет 3: conservative режим (только при improvement >= 10%)
                show_scenario_only = False
                st.warning("⚠️ **Консервативный режим:** модель показывает умеренную нестабильность. Оптимум ограничен, рекомендуем дополнительно оценить сценарии.")
            elif is_boundary_search and (model_result.get("stability_mode") in ["S2", "S3"] or e_iqr > 0.4):
                # Приоритет 4: граничное решение по поисковой сетке
                show_scenario_only = False
                st.warning("⚠️ **Граничное решение по сетке:** оптимальная цена находится на краю поискового диапазона. Реальный максимум прибыли может лежать вне перебираемого интервала.")
            else:
                # Приоритет 5: стандартный режим (только при improvement >= 10%)
                show_scenario_only = False
            
            if e_med > 0 and allow_positive_elasticity:
                # Ограничиваем оптимизацию: рост спроса допускается только в нижней части диапазона (p<=p60)
                st.info("🔒 **Ограничение для положительной эластичности:** оптимизация ограничена нижней частью диапазона цен.")
                # Это можно использовать в оптимизаторе для дополнительного ограничения
            
            if show_scenario_only:
                # --- Только сценарный анализ ---
                st.markdown("#### 🧪 Сценарный анализ (Чувствительность)")
                scenarios = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]
                scenario_data = []                
                for s in scenarios:
                    p_test_before = last_p_before * (1 + s)
                    p_test_after = p_test_before * (1.0 - s_val)
                    q_test = float(sf.predict_sales(p_test_after, {"ad_internal": ad_internal, "ad_bloggers": ad_bloggers, "ad_vk": ad_vk}))
                    
                    # Unit Econ
                    comm_test = p_test_before * (commission_pct / 100.0)
                    vat_test = p_test_after * (vat_pct / 100.0)
                    m_test = p_test_before - comm_test - vat_test - cogs - logistics - storage
                    prof_test = (m_test * q_test) - total_ad_spend
                    
                    scenario_data.append({
                        "Изменение": f"{s*100:+.0f}%",
                        "Цена до СПП": f"{p_test_before:.0f} ₽",
                        "Маржа (ед)": f"{m_test:.0f} ₽",
                        "Заказы": f"{q_test:.1f}",
                        "Прибыль": f"{prof_test:.0f} ₽",
                        "Эффект П": f"{prof_test - profit_last:+.0f} ₽"
                    })
                
                st.table(pd.DataFrame(scenario_data))
                
                if protective == "scenario":
                    st.warning("⚠️ Внимание: Прогнозы в таблице выше могут быть неточными, так как модель спроса не показала значимого улучшения над базовым средним. Используйте эти данные как ориентир по маржинальности, а не как прогноз точности спроса.")
                else:
                    st.info("💡 Поскольку модель в данной зоне нестабильна, рекомендуется опираться на сценарную таблицу, а не на единичную точку оптимума.")
            
            else:
                # --- Стандартный вывод Оптимума ---
                # Оптимальные значения
                q_opt = best_info['best_sales']
                profit_opt = best_info['best_profit']
                margin_opt = best_info['best_margin']
                p_opt_before = best_info['best_price_before']
                p_opt_after = best_info['best_customer_price']
                profitability_opt = (margin_opt / p_opt_before) * 100

                # --- Метрика 1: Цены ---
                st.markdown("#### 💰 Сравнение цен")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Цена до СПП", f"{p_opt_before:.0f} ₽", 
                            delta=f"{p_opt_before - last_p_before:+.0f} ₽")
                    st.caption(f"Текущая: {last_p_before:.0f} ₽")
                with c2:
                    st.metric("Цена клиента", f"{p_opt_after:.0f} ₽",
                            delta=f"{p_opt_after - p_after_last:+.0f} ₽")
                    st.caption(f"Текущая: {p_after_last:.0f} ₽")
                with c3:
                    st.metric("Маржинальность", f"{profitability_opt:.1f}%",
                            delta=f"{profitability_opt - profitability_last:+.1f}%")
                    st.caption(f"Текущая: {profitability_last:.1f}%")

                # --- Метрика 2: Прогнозы ---
                st.markdown("#### 📈 Прогноз эффекта")
                c4, c5, c6 = st.columns(3)
                with c4:
                    st.metric("Заказы (шт/день)", f"{q_opt:.1f}",
                            delta=f"{q_opt - q_last:+.1f}")
                    st.caption(f"Текущая: {q_last:.1f}")
                with c5:
                    st.metric("Прибыль (₽/день)", f"{profit_opt:.0f} ₽",
                            delta=f"{profit_opt - profit_last:+.0f} ₽")
                    st.caption(f"Текущая: {profit_last:.0f} ₽")
                with c6:
                    profit_delta = profit_opt - profit_last
                    st.metric("Прирост прибыли", f"{profit_delta:+.0f} ₽",
                            delta=f"{(profit_delta/max(1.0, profit_last)*100):+.1f}%")
                    st.caption("К текущему уровню")

                st.write(f"💰 Чистая маржа с единицы: **{margin_opt:.2f} ₽**")
                
                # Проверка граничного решения (по ТЗ)
                if is_boundary_search:
                    st.warning("⚠️ **Оптимальная цена на границе поисковой сетки.** Реальный максимум прибыли может лежать за пределами перебираемого диапазона.")

                if is_boundary_history:
                    st.warning("⚠️ **Оптимальная цена у границы исторического диапазона.** Повышен риск смещения оптимума за пределы наблюдавшихся цен.")

                # Асимметричная проверка экстраполяции (по ТЗ)
                tol = 0.02
                if p_opt_before > p_max_hist * (1 + tol):
                    st.warning(f"⚠️ **Риск: модель экстраполирует вверх.** Оптимальная цена ({p_opt_before:.0f} ₽) значительно выше исторического максимума. Спрос может обвалиться сильнее, чем прогнозирует модель.")
                elif p_opt_before < p_min_hist * (1 - tol):
                    st.info(f"ℹ️ **Экстраполяция вниз:** Оптимальная цена ({p_opt_before:.0f} ₽) ниже исторического минимума. Прогноз роста продаж основан на экстраполяции модели.")
                
                # --- Управленческая рекомендация на основе Grid Search ---
                st.divider()
                opt_p = best_info['best_price_before']
                last_p = last_p_before
                delta_p = (opt_p - last_p) / last_p
                
                # Рекомендованный диапазон
                stability = model_result.get("stability_mode", "S1")
                if stability == "S3":
                    # В нестабильном режиме даем узкий диапазон
                    range_low = opt_p * 0.97
                    range_high = opt_p * 1.03
                    st.info(f"📍 **Локальный оптимум:** {opt_p:.0f} ₽")
                    st.markdown(f"📏 **Рекомендованный диапазон:** `{range_low:.0f} – {range_high:.0f} ₽`")
                else:
                    st.markdown(f"🎯 **Целевая цена:** `{opt_p:.0f} ₽`")

                if delta_p > 0.01:
                    st.success(f"🚀 **Рекомендация:** Модель считает, что цену выгодно повысить на {delta_p*100:.1f}%.")
                elif delta_p < -0.01:
                    st.info(f"💡 **Рекомендация:** Модель считает, что цену выгодно снизить на {abs(delta_p)*100:.1f}%.")
                else:
                    st.success("🎯 **Рекомендация:** Текущая цена близка к математическому оптимуму.")
            
            # Проверка ограничений по режимам стабильности
            stability = model_result.get("stability_mode", "S1")
            if stability != "S1":
                st.caption(f"ℹ️ Рекомендация ограничена режимом `{stability}` из-за особенностей данных/модели.")


        # Demand Curve with markers - только при OK состоянии данных
        if data_state != "OK":
            st.error("🚫 **Визуализация недоступна:** состояние данных не позволяет провести анализ")
            return
            
        st.subheader("Кривая спроса (Прогноз)")
        fig1 = px.line(results, x="price_before_spp", y="predicted_sales", markers=True, title="Зависимость продаж от цены")
        # Add historical range vertical lines
        fig1.add_vline(x=p_min_hist, line_dash="dash", line_color="gray", annotation_text="Min Hist")
        fig1.add_vline(x=p_max_hist, line_dash="dash", line_color="gray", annotation_text="Max Hist")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Диагностика: Actual vs Predicted")
        # Debug логирование перед диагностикой (ТЗ)
        logger.debug("model_name before predict_on_df: %s", sf.best_model_name)
        logger.debug("data_state before predict_on_df: %s", getattr(sf, "data_state", "UNKNOWN"))
        logger.debug("fit_return before predict_on_df: %s", getattr(sf, "_fit_return_value", "UNKNOWN"))
        
        diag_df = sku_df.copy()
        diag_df["predicted_orders"] = sf.predict_on_df(diag_df)
        diag_df["error_pct"] = (diag_df["predicted_orders"] - diag_df["orders"]).abs() / diag_df["orders"].replace(0, 1) * 100
        
        fig_diag = px.scatter(diag_df, x="date", y=["orders", "predicted_orders"], title="Сравнение факта и прогноза на истории")
        st.plotly_chart(fig_diag, use_container_width=True)
        
        st.write("Последние 10 дней истории с прогнозом:")
        st.dataframe(diag_df[["date", "price_before_spp", "orders", "predicted_orders", "error_pct"]].tail(10))

        st.subheader("Таблица всех расчётов")
        st.dataframe(results)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Скачать CSV", csv, "optimization_results.csv", "text/csv")


if __name__ == "__main__":
    main()
