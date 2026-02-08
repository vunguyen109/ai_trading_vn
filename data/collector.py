import argparse
import datetime as dt
import logging
import os
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from vnstock import Vnstock


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("collector")
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


def normalize_symbol_for_vnstock(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.startswith("^"):
        s = s[1:]
    if ".VN" in s:
        s = s.replace(".VN", "")
    return s


def calc_date_range(history_days: int) -> Tuple[dt.date, dt.date]:
    end = dt.date.today()
    start = end - dt.timedelta(days=history_days)
    return start, end


def normalize_vnstock_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.name:
        df = df.reset_index()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "tradingdate": "date",
        "trading_date": "date",
        "time": "date",
        "date": "date",
        "open_price": "open",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
        "closing_price": "close",
        "closeprice": "close",
        "adjclose": "adj_close",
        "adjusted_close": "adj_close",
        "total_volume": "volume",
        "vol": "volume",
    }

    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def download_symbol(symbol: str, start: dt.date, end: dt.date, provider: str) -> pd.DataFrame:
    stock = Vnstock().stock(symbol=symbol, source=provider)
    df = stock.quote.history(start=start.isoformat(), end=end.isoformat(), interval="1D")
    if df is None or df.empty:
        return pd.DataFrame()

    return normalize_vnstock_df(df)


def save_csv(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS prices (
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


def prepare_df_for_db(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = symbol
    if "adj_close" not in out.columns:
        out["adj_close"] = out.get("close")
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)

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
        if c not in out.columns:
            out[c] = None
    return out[cols]


def save_to_sqlite(engine: Engine, df: pd.DataFrame, symbol: str) -> None:
    if df.empty:
        return

    df_db = prepare_df_for_db(df, symbol)
    min_date = df_db["date"].min()
    max_date = df_db["date"].max()

    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM prices WHERE symbol = :symbol AND date BETWEEN :start AND :end"
            ),
            {"symbol": symbol, "start": min_date, "end": max_date},
        )
        df_db.to_sql("prices", conn, if_exists="append", index=False)


def collect_data(config_path: str, symbols: List[str] = None) -> None:
    logger = setup_logger()
    config = load_config(config_path)

    data_cfg = config.get("data", {})
    history_days = int(data_cfg.get("history_days", 365))
    start, end = calc_date_range(history_days)

    raw_symbols = symbols or data_cfg.get("symbols", [])
    if not raw_symbols:
        logger.error("No symbols configured. Check config.yaml")
        return

    db_path = data_cfg.get("database", "data/database.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    init_db(engine)

    out_dir = os.path.join("data", "raw")
    logger.info("Collecting %d symbols from %s to %s", len(raw_symbols), start, end)

    provider = data_cfg.get("provider", "KBS")

    for s in raw_symbols:
        symbol = normalize_symbol_for_vnstock(s)
        logger.info("Downloading %s (vnstock/%s)", symbol, provider)
        try:
            df = download_symbol(symbol, start, end, provider)
        except Exception as exc:
            logger.warning("Failed %s: %s", symbol, exc)
            continue
        if df.empty:
            logger.warning("No data for %s", symbol)
            continue

        df_csv = df.copy()
        df_csv.insert(0, "symbol", symbol)
        out_path = os.path.join(out_dir, f"{symbol}.csv")
        save_csv(df_csv, out_path)
        logger.info("Saved %s", out_path)

        save_to_sqlite(engine, df, symbol)
        logger.info("Upserted %s into SQLite", symbol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data collector")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    parser.add_argument("--symbols", nargs="*", help="Override symbols list")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_data(args.config, args.symbols)


if __name__ == "__main__":
    main()
