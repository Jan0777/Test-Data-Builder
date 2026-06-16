from __future__ import annotations
import io
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


def ingest_file(file_path: str | Path, filename: str = "") -> Dict[str, pd.DataFrame]:
    """
    Read a CSV, XLSX, or JSON file and return a dict of table_name → DataFrame.
    For multi-sheet XLSX, each sheet becomes a table.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    if not ext and filename:
        ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".csv":
        df = _read_csv(path)
        stem = _safe_name(path.stem or filename or "table")
        return {stem: df}

    elif ext in (".xlsx", ".xls"):
        return _read_excel(path)

    elif ext == ".json":
        df = _read_json(path)
        stem = _safe_name(path.stem or filename or "table")
        return {stem: df}

    return {}


def _read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "cp1252"]
    delimiters = [",", "\t", ";", "|"]

    for enc in encodings:
        for delim in delimiters:
            try:
                df = pd.read_csv(path, encoding=enc, sep=delim, on_bad_lines="skip")
                if df.shape[1] > 1 or (df.shape[1] == 1 and df.shape[0] > 0):
                    df.columns = [_safe_col(c) for c in df.columns]
                    return df
            except Exception:
                continue

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [_safe_col(c) for c in df.columns]
    return df


def _read_excel(path: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    try:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet)
                if df.empty:
                    continue
                df.columns = [_safe_col(c) for c in df.columns]
                name = _safe_name(str(sheet))
                tables[name] = df
            except Exception as e:
                logger.warning(f"Could not parse sheet '{sheet}': {e}")
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {e}")

    if not tables:
        raise ValueError("No readable sheets found in Excel file")
    return tables


def _read_json(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_json(path)
        if isinstance(df, pd.DataFrame):
            df.columns = [_safe_col(c) for c in df.columns]
            return df
    except Exception:
        pass

    import json
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        if all(isinstance(v, list) for v in data.values()):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    else:
        raise ValueError("JSON must be a list of objects or a dict of arrays")

    df.columns = [_safe_col(c) for c in df.columns]
    return df


def _safe_col(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def _safe_name(name: str) -> str:
    import re
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "table"
