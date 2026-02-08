import argparse
import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("models.predictor")
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


def list_feature_files(feature_dir: str, symbols: List[str] | None) -> List[str]:
    if symbols:
        return [os.path.join(feature_dir, f"{s}.csv") for s in symbols]
    if not os.path.isdir(feature_dir):
        return []
    return [
        os.path.join(feature_dir, f)
        for f in os.listdir(feature_dir)
        if f.lower().endswith(".csv")
    ]


def load_model(path: str):
    return pd.read_pickle(path)


def prepare_features(df: pd.DataFrame, model) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    ignore = {"symbol", "date", "target", "future_close"}
    feature_cols = [c for c in df.columns if c not in ignore]

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in df.columns:
                df[col] = 0.0
        X = df[expected]
    else:
        X = df[feature_cols]

    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return X


def predict_for_symbol(path: str, model_dir: str) -> Dict:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Empty feature file")

    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else os.path.splitext(os.path.basename(path))[0]
    model_path = os.path.join(model_dir, f"{symbol}_rf.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path)
    X = prepare_features(df, model)
    if X.empty:
        raise ValueError("No features to predict")

    latest = X.iloc[[-1]]
    proba = model.predict_proba(latest)[0][1]
    pred = int(proba >= 0.5)

    date = df["date"].iloc[-1] if "date" in df.columns else ""
    return {
        "symbol": symbol,
        "date": str(date),
        "pred_up_next_3_days": pred,
        "prob_up": float(proba),
        "model_path": model_path,
    }


def run_pipeline(
    config_path: str,
    symbols: List[str] | None,
    feature_dir: str,
    model_dir: str,
    out_path: str,
) -> None:
    logger = setup_logger()
    _ = load_config(config_path)

    files = list_feature_files(feature_dir, symbols)
    if not files:
        logger.error("No feature files found in %s", feature_dir)
        return

    results = []
    for path in files:
        if not os.path.exists(path):
            logger.warning("Missing file: %s", path)
            continue
        try:
            res = predict_for_symbol(path, model_dir)
        except Exception as exc:
            logger.warning("Skip %s: %s", path, exc)
            continue
        results.append(res)
        logger.info("Predicted %s | prob_up=%.4f", res["symbol"], res["prob_up"])

    if results:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info("Saved %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML predictor")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    parser.add_argument("--symbols", nargs="*", help="Override symbols list")
    parser.add_argument(
        "--feature_dir",
        default=os.path.join("data", "features", "technical"),
        help="Feature directory",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join("models", "trained_models"),
        help="Model directory",
    )
    parser.add_argument(
        "--out_path",
        default=os.path.join("models", "predictions", "latest_predictions.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, args.symbols, args.feature_dir, args.model_dir, args.out_path)


if __name__ == "__main__":
    main()
