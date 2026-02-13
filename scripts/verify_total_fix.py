import pandas as pd
import io
from src.ingestion.excel_parser import ExcelIngestor

def test_total_row_skipping():
    # Создаем данные со строкой "Total" в колонке даты
    df_raw = pd.DataFrame({
        "День": ["2024-01-01", "2024-01-02", "Total"],
        "Заказы, шт": [10, 20, 30],
        "Ср.чек заказа до СПП, ₽": [1000, 1100, 2100],
        "Ср.чек заказа после СПП, ₽": [800, 900, 1700],
        "СПП WB заказы, %": [20, 20, 20],
        "Себестоимость": [500, 500, 1000],
        "Логистика": [100, 100, 200],
        "Хранение": [50, 50, 100],
        "артикул": ["SKU1", "SKU1", "SKU1"]
    })
    
    # Сохраняем в байтовый поток как Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_raw.to_excel(writer, index=False)
    output.seek(0)
    
    ing = ExcelIngestor()
    # Загружаем
    df, report = ing.load_files([output])
    
    print("Columns after loading:", df.columns.tolist())
    print("Num rows:", len(df))
    print("Dates:", df["date"].tolist())
    
    if len(df) == 2 and "Total" not in df["date"].astype(str).tolist():
        print("SUCCESS: Summary row skipped correctly.")
    else:
        print(f"FAILURE: Expected 2 rows, got {len(df)}.")

if __name__ == "__main__":
    test_total_row_skipping()
