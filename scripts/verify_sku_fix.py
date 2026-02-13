import pandas as pd
import io
from src.ingestion.excel_parser import ExcelIngestor

def test_sku_fallback():
    # Создаем данные без SKU
    df_no_sku = pd.DataFrame({
        "День": ["2024-01-01", "2024-01-02"],
        "Заказы, шт": [10, 20],
        "Ср.чек заказа до СПП, ₽": [1000, 1100],
        "Ср.чек заказа после СПП, ₽": [800, 900],
        "СПП WB заказы, %": [20, 20],
        "Себестоимость": [500, 500],
        "Логистика": [100, 100],
        "Хранение": [50, 50]
    })
    
    # Сохраняем в байтовый поток как Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_no_sku.to_excel(writer, index=False)
    output.seek(0)
    
    ing = ExcelIngestor()
    # Загружаем
    df, report = ing.load_files([output])
    
    print("Columns after loading:", df.columns.tolist())
    print("Report:", report)
    
    if "sku" in df.columns:
        print("SUCCESS: 'sku' column added.")
        print("First few rows:")
        print(df[["sku", "date", "orders"]].head())
    else:
        print("FAILURE: 'sku' column missing.")

if __name__ == "__main__":
    test_sku_fallback()
