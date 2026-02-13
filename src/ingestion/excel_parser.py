import re
from typing import Dict, List, Tuple, Optional

import pandas as pd


DEFAULT_COLUMNS = [
    "date",
    "sku",
    "orders",
    "price_before_spp",
    "price_after_spp",
    "spp",
    "cogs",
    "logistics",
    "storage",
    "ad_internal",
    "ad_bloggers",
    "ad_vk",
]


class ValidationError(Exception):
    pass


class ExcelIngestor:
    """Загрузчик нескольких Excel-файлов с маппингом колонок и базовой валидацией.

    Пример использования:
        ing = ExcelIngestor()
        df, report = ing.load_files(["a.xlsx", "b.xlsx"], mapping={"price":"price_before_spp"})
    """

    def __init__(self, expected_columns: Optional[List[str]] = None):
        self.expected_columns = expected_columns or DEFAULT_COLUMNS

    def _normalize_col(self, name: str) -> str:
        # Убираем все кроме букв и цифр, включая кириллицу
        return re.sub(r"[^\w0-9]", "", name.lower())

    def infer_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Попытаться автоматически маппить входные колонки на ожидаемые.

        Правила: сравнение по нормализованным именам, поиск ключевых подстрок.
        """
        mapping: Dict[str, str] = {}
        norm_to_orig = {self._normalize_col(c): c for c in columns}

        # Приоритетный маппинг для специфичных русских названий
        russian_map = {
            "день": "date",
            "артикул": "sku",
            "заказышт": "orders",
            "срчекзаказадоспп": "price_before_spp",
            "срчекзаказапослеспп": "price_after_spp",
            "сппwbзаказы": "spp",
            "рекламавнут": "ad_internal",
            "рекламаблогеры": "ad_bloggers",
            "рекламавк": "ad_vk",
            "себестоимость": "cogs",
            "логистика": "logistics",
            "хранение": "storage"
        }

        for nc, orig in norm_to_orig.items():
            for rus_norm, target in russian_map.items():
                if rus_norm in nc:
                    mapping[orig] = target
                    break

        for exp in self.expected_columns:
            if exp in mapping.values():
                continue
            nexp = self._normalize_col(exp)
            # точное совпадение
            if nexp in norm_to_orig:
                mapping[norm_to_orig[nexp]] = exp
                continue
            # поиск по подстроке
            for nc, orig in norm_to_orig.items():
                if exp in orig.lower() or exp.replace("_", "") in nc:
                    mapping[orig] = exp
                    break
            else:
                # дополнительные эвристики
                if "price" in exp:
                    for nc, orig in norm_to_orig.items():
                        if "price" in nc and orig not in mapping:
                            mapping[orig] = exp
                            break

        return mapping

    def _apply_mapping(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        # Only rename columns that are mapped
        rename_map = {orig: dest for orig, dest in mapping.items() if orig in df.columns}
        return df.rename(columns=rename_map)

    def _coerce_types(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        # date -> datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            # Удаляем строки, где дата не распарсилась (например, строка "Итого" или "Total")
            df = df.dropna(subset=["date"]).copy()

        # sku -> str
        if "sku" in df.columns:
            df["sku"] = df["sku"].astype(str)

        # numeric fields
        numeric_cols = [
            c for c in [
                "orders", "price_before_spp", "price_after_spp", "spp", 
                "cogs", "logistics", "storage",
                "ad_internal", "ad_bloggers", "ad_vk"
            ] if c in df.columns
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # spp normalization: if values look like percents (0..100) convert to fraction
        if "spp" in df.columns:
            s = df["spp"].dropna()
            if not s.empty:
                maxv = s.max()
                if maxv > 1 and maxv <= 100:
                    df["spp"] = df["spp"] / 100.0
                    report.setdefault("warnings", []).append("spp values >1 and <=100 interpreted as percent and divided by 100")

        return df

    def _basic_validations(self, df: pd.DataFrame, report: dict) -> None:
        # required columns
        missing = [c for c in self.expected_columns if c not in df.columns]
        if missing:
            report.setdefault("errors", []).append(f"missing columns after mapping: {missing}")

        # simple value checks
        if "orders" in df.columns:
            neg = df["orders"] < 0
            if neg.any():
                report.setdefault("errors", []).append(f"negative orders in {neg.sum()} rows")

        for col in ["price_before_spp", "price_after_spp", "cogs", "logistics", "storage"]:
            if col in df.columns:
                neg = df[col] < 0
                if neg.any():
                    report.setdefault("errors", []).append(f"negative {col} in {neg.sum()} rows")

        if "spp" in df.columns:
            bad = (df["spp"] < 0) | (df["spp"] > 1)
            if bad.any():
                report.setdefault("warnings", []).append(f"spp outside [0,1] in {bad.sum()} rows")

    def load_files(self, paths: List[str], mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, dict]:
        """Load multiple Excel files, map/normalize columns and run validations.

        Returns (df, report) where report contains errors/warnings.
        If errors present — they are listed in report['errors'] but df is still returned (caller can choose to raise).
        """
        frames = []
        report: dict = {}

        for p in paths:
            try:
                df = pd.read_excel(p)
            except Exception as e:
                report.setdefault("errors", []).append(f"failed to read {p}: {e}")
                continue

            # Удаляем колонки Unnamed
            unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
                report.setdefault("warnings", []).append(f"Удалены колонки Unnamed: {unnamed_cols}")
            
            # if user mapping not provided, attempt infer per-file
            if mapping is None:
                inferred = self.infer_mapping(list(df.columns))
            else:
                # mapping keys are source column names; allow normalized matching
                inferred = mapping.copy()

            df = self._apply_mapping(df, inferred)
            df = self._coerce_types(df, report)
            
            # Если колонка SKU отсутствует - создаем дефолтную
            if "sku" not in df.columns:
                df["sku"] = "Default SKU"
                
            frames.append(df)

        if not frames:
            report.setdefault("errors", []).append("no files loaded")
            return pd.DataFrame(), report

        combined = pd.concat(frames, ignore_index=True, sort=False)
        self._basic_validations(combined, report)

        return combined, report


if __name__ == "__main__":
    # quick local test
    ing = ExcelIngestor()
    import sys

    paths = sys.argv[1:]
    if not paths:
        print("Usage: python excel_parser.py file1.xlsx file2.xlsx")
        raise SystemExit(1)
    df, report = ing.load_files(paths)
    print(df.head())
    print(report)
