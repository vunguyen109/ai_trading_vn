import argparse
import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("features.technical")
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


def list_clean_files(clean_dir: str, symbols: List[str] | None) -> List[str]:
    if symbols:
        return [os.path.join(clean_dir, f"{s}.csv") for s in symbols]
    if not os.path.isdir(clean_dir):
        return []
    return [
        os.path.join(clean_dir, f)
        for f in os.listdir(clean_dir)
        if f.lower().endswith(".csv")
    ]


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
        }
    )


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
        }
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def stochastic_kd(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smooth_k: int = 3
) -> pd.DataFrame:
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(smooth_k).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def compute_features(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    indicators = cfg.get("features", {}).get("technical_indicators", [])
    window_sizes = cfg.get("features", {}).get("window_sizes", [1, 3, 5, 10])

    if "close" in df.columns:
        df["return_1d"] = df["close"].pct_change()
        df["log_return_1d"] = np.log(df["close"]).diff()

        for w in window_sizes:
            df[f"close_lag_{w}"] = df["close"].shift(w)

    if "rsi" in indicators and "close" in df.columns:
        df["rsi_14"] = rsi(df["close"], 14)

    if "macd" in indicators and "close" in df.columns:
        df = pd.concat([df, macd(df["close"])], axis=1)

    if "bollinger_bands" in indicators and "close" in df.columns:
        df = pd.concat([df, bollinger(df["close"], 20, 2.0)], axis=1)

    if "sma_20" in indicators and "close" in df.columns:
        df["sma_20"] = df["close"].rolling(20).mean()

    if "sma_50" in indicators and "close" in df.columns:
        df["sma_50"] = df["close"].rolling(50).mean()

    if "volume_sma" in indicators and "volume" in df.columns:
        df["volume_sma_20"] = df["volume"].rolling(20).mean()

    if "atr" in indicators and {"high", "low", "close"}.issubset(df.columns):
        df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)

    if "stochastic" in indicators and {"high", "low", "close"}.issubset(df.columns):
        df = pd.concat([df, stochastic_kd(df["high"], df["low"], df["close"], 14, 3)], axis=1)

    return df


def save_csv(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS features_technical (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    data JSON,
                    UNIQUE(symbol, date)
                )
                """
            )
        )


def save_to_sqlite(engine: Engine, df: pd.DataFrame, symbol: str) -> None:
    if df.empty or "date" not in df.columns:
        return

    df_db = df.copy()
    df_db["symbol"] = symbol
    df_db["date"] = pd.to_datetime(df_db["date"]).dt.date.astype(str)

    payload_cols = [c for c in df_db.columns if c not in ["symbol", "date"]]
    df_db["data"] = df_db[payload_cols].apply(
        lambda row: row.dropna().to_json(), axis=1
    )
    df_db = df_db[["symbol", "date", "data"]]

    min_date = df_db["date"].min()
    max_date = df_db["date"].max()

    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM features_technical WHERE symbol = :symbol AND date BETWEEN :start AND :end"
            ),
            {"symbol": symbol, "start": min_date, "end": max_date},
        )
        df_db.to_sql("features_technical", conn, if_exists="append", index=False)


def run_pipeline(config_path: str, symbols: List[str] | None, input_dir: str, output_dir: str) -> None:
    logger = setup_logger()
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})

    files = list_clean_files(input_dir, symbols)
    if not files:
        logger.error("No clean files found in %s", input_dir)
        return

    db_path = data_cfg.get("database", "data/database.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    init_db(engine)

    logger.info("Generating features from %d files", len(files))
    for path in files:
        if not os.path.exists(path):
            logger.warning("Missing file: %s", path)
            continue
        df = pd.read_csv(path)
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else os.path.splitext(os.path.basename(path))[0]
        df_feat = compute_features(df, cfg)

        out_path = os.path.join(output_dir, f"{symbol}.csv")
        save_csv(df_feat, out_path)
        logger.info("Saved %s", out_path)

        save_to_sqlite(engine, df_feat, symbol)
        logger.info("Upserted %s into SQLite", symbol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Technical features pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    parser.add_argument("--symbols", nargs="*", help="Override symbols list")
    parser.add_argument("--input_dir", default=os.path.join("data", "clean"))
    parser.add_argument("--output_dir", default=os.path.join("data", "features", "technical"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, args.symbols, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
