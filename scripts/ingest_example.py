from pathlib import Path
from src.ingestion.excel_parser import ExcelIngestor
import pandas as pd


def create_demo_files(tmp_dir: Path):
    a = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-02"],
            "SKU": ["SKU1", "SKU1"],
            "Orders": [10, 12],
            "Price": [1000, 990],
            "Price_after": [900, 891],
            "SPP%": [5, 5],
            "COGS": [600, 600],
            "Logistics": [50, 50],
            "Storage": [10, 10],
        }
    )

    b = pd.DataFrame(
        {
            "date": ["2026-01-03"],
            "sku": ["SKU1"],
            "orders": [8],
            "price_before_spp": [980],
            "price_after_spp": [931],
            "spp": [0.05],
            "cogs": [600],
            "logistics": [50],
            "storage": [10],
        }
    )

    p1 = tmp_dir / "a.xlsx"
    p2 = tmp_dir / "b.xlsx"
    a.to_excel(p1, index=False)
    b.to_excel(p2, index=False)
    return [str(p1), str(p2)]


def main():
    tmp = Path("./data")
    tmp.mkdir(parents=True, exist_ok=True)
    files = create_demo_files(tmp)

    ing = ExcelIngestor()
    df, report = ing.load_files(files)

    print("Loaded rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Report:", report)
    print(df.head())


if __name__ == "__main__":
    main()
