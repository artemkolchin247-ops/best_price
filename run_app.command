#!/bin/bash
# Переходим в директорию, где находится этот файл
cd "$(dirname "$0")"

# Запускаем приложение через streamlit из виртуального окружения
./.venv/bin/streamlit run ui/app.py
