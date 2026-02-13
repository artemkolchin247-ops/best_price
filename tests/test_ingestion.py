import pandas as pd
from pathlib import Path
import tempfile

from src.ingestion.excel_parser import ExcelIngestor


def write_excel(df: pd.DataFrame, path: Path):
    df.to_excel(path, index=False)


def test_load_multiple_files_and_mapping(tmp_path):
    # create two files with different column names
    a = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "SKU_CODE": ["SKU_TEST"],
            "Orders": [5],
            "PriceBefore": [500],
            "PriceAfter": [475],
            "SPP%": [5],
            "COGS": [200],
            "Logistics": [20],
            "Storage": [5],
        }
    )

    b = pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "sku": ["SKU_TEST"],
            "orders": [7],
            "price_before_spp": [520],
            "price_after_spp": [494],
            "spp": [0.05],
            "cogs": [200],
            "logistics": [20],
            "storage": [5],
        }
    )

    p1 = tmp_path / "a.xlsx"
    p2 = tmp_path / "b.xlsx"
    write_excel(a, p1)
    write_excel(b, p2)

    ing = ExcelIngestor()
    df, report = ing.load_files([str(p1), str(p2)])

    # basic expectations
    assert not df.empty
    # expected normalized columns present
    for c in ["date", "sku", "orders", "price_before_spp", "price_after_spp"]:
        assert c in df.columns

    # orders numeric and non-negative
    assert (df["orders"] >= 0).all()

    # report should not contain fatal read errors
    assert "failed to read" not in " ".join(report.get("errors", []))

