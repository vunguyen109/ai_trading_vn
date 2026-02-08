import argparse
import datetime as dt
import glob
import logging
import os
from typing import Dict, List

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("cleaner")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_raw_files(raw_dir: str, symbols: List[str] | None) -> List[str]:
    if symbols:
        return [os.path.join(raw_dir, f"{s}.csv") for s in symbols]
    return glob.glob(os.path.join(raw_dir, "*.csv"))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    if "date" not in df.columns:
        raise ValueError("Missing date column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "close" in df.columns:
        df = df.dropna(subset=["close"])

    # Fill missing OHLC/adj_close forward, then backward for leading gaps
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]
    if price_cols:
        df[price_cols] = df[price_cols].ffill().bfill()

    # Fill missing volume with 0
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)

    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    cols = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    return df[cols]


def find_missing_dates(df: pd.DataFrame) -> List[dt.date]:
    if df.empty or "date" not in df.columns:
        return []
    dates = pd.to_datetime(df["date"]).dt.date
    start = min(dates)
    end = max(dates)
    expected = pd.bdate_range(start=start, end=end).date
    missing = sorted(set(expected) - set(dates))
    return missing


def build_stats(df_raw: pd.DataFrame, df_clean: pd.DataFrame, missing_dates: List[dt.date]) -> Dict:
    stats = {
        "rows_raw": len(df_raw),
        "rows_clean": len(df_clean),
        "missing_dates": len(missing_dates),
    }
    if "date" in df_clean.columns and not df_clean.empty:
        stats["date_min"] = str(df_clean["date"].min())
        stats["date_max"] = str(df_clean["date"].max())
    return stats


def save_csv(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS prices_clean (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume REAL,
                    UNIQUE(symbol, date)
                )
                """
            )
        )


def save_to_sqlite(engine: Engine, df: pd.DataFrame, symbol: str) -> None:
    if df.empty:
        return
    df_db = df.copy()
    df_db["date"] = pd.to_datetime(df_db["date"]).dt.date.astype(str)
    min_date = df_db["date"].min()
    max_date = df_db["date"].max()

    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM prices_clean WHERE symbol = :symbol AND date BETWEEN :start AND :end"
            ),
            {"symbol": symbol, "start": min_date, "end": max_date},
        )
        df_db.to_sql("prices_clean", conn, if_exists="append", index=False)


def extract_symbol(df: pd.DataFrame, fallback: str) -> str:
    if "symbol" in df.columns and df["symbol"].notna().any():
        return str(df["symbol"].dropna().iloc[0])
    return fallback


def clean_files(config_path: str, symbols: List[str] | None) -> None:
    logger = setup_logger()
    config = load_config(config_path)
    data_cfg = config.get("data", {})

    raw_dir = os.path.join("data", "raw")
    clean_dir = os.path.join("data", "clean")
    files = list_raw_files(raw_dir, symbols)
    if not files:
        logger.error("No raw files found in %s", raw_dir)
        return

    db_path = data_cfg.get("database", "data/database.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    init_db(engine)

    logger.info("Cleaning %d files", len(files))
    for path in files:
        if not os.path.exists(path):
            logger.warning("Missing file: %s", path)
            continue
        df_raw = pd.read_csv(path)
        symbol = extract_symbol(df_raw, os.path.splitext(os.path.basename(path))[0])

        try:
            df_clean = clean_df(df_raw)
        except Exception as exc:
            logger.warning("Skip %s: %s", path, exc)
            continue

        missing_dates = find_missing_dates(df_clean)
        stats = build_stats(df_raw, df_clean, missing_dates)
        logger.info(
            "Stats %s: rows %d -> %d, date %s to %s, missing dates %d",
            symbol,
            stats.get("rows_raw", 0),
            stats.get("rows_clean", 0),
            stats.get("date_min", "-"),
            stats.get("date_max", "-"),
            stats.get("missing_dates", 0),
        )
        if missing_dates:
            sample = ", ".join(str(d) for d in missing_dates[:5])
            logger.info("Missing dates sample for %s: %s", symbol, sample)

        out_path = os.path.join(clean_dir, f"{symbol}.csv")
        save_csv(df_clean, out_path)
        logger.info("Saved %s", out_path)

        save_to_sqlite(engine, df_clean, symbol)
        logger.info("Upserted %s into SQLite", symbol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data cleaner")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    parser.add_argument("--symbols", nargs="*", help="Override symbols list")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_files(args.config, args.symbols)


if __name__ == "__main__":
    main()
