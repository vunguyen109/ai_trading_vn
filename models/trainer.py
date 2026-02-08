import argparse
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("models.trainer")
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


def build_target(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df = df.dropna(subset=["future_close"])
    return df


def drop_non_feature_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    ignore = {"symbol", "date", "target", "future_close"}
    feature_cols = [c for c in df.columns if c not in ignore]
    return df[feature_cols], feature_cols


def time_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, cfg: Dict) -> RandomForestClassifier:
    hp = cfg.get("ml", {}).get("hyperparameters", {})
    model = RandomForestClassifier(
        n_estimators=int(hp.get("n_estimators", 100)),
        max_depth=int(hp.get("max_depth", 10)),
        min_samples_split=int(hp.get("min_samples_split", 5)),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds, zero_division=0),
    }


def train_for_symbol(path: str, cfg: Dict, out_dir: str) -> Dict:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    df = build_target(df, horizon=3)
    df = df.dropna()

    train_df, test_df = time_split(df, float(cfg.get("ml", {}).get("test_size", 0.2)))
    X_train, _ = drop_non_feature_cols(train_df)
    y_train = train_df["target"]
    X_test, _ = drop_non_feature_cols(test_df)
    y_test = test_df["target"]

    model = train_model(X_train, y_train, cfg)
    metrics = evaluate(model, X_test, y_test)

    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{symbol}_rf.pkl")
    pd.to_pickle(model, model_path)

    return {
        "symbol": symbol,
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": metrics["accuracy"],
        "model_path": model_path,
        "report": metrics["report"],
    }


def run_pipeline(config_path: str, symbols: List[str] | None, feature_dir: str, out_dir: str) -> None:
    logger = setup_logger()
    cfg = load_config(config_path)

    files = list_feature_files(feature_dir, symbols)
    if not files:
        logger.error("No feature files found in %s", feature_dir)
        return

    logger.info("Training models from %d files", len(files))
    summary = []
    for path in files:
        if not os.path.exists(path):
            logger.warning("Missing file: %s", path)
            continue
        try:
            result = train_for_symbol(path, cfg, out_dir)
        except Exception as exc:
            logger.warning("Skip %s: %s", path, exc)
            continue

        summary.append(result)
        logger.info(
            "Trained %s | rows=%d train=%d test=%d acc=%.4f",
            result["symbol"],
            result["rows"],
            result["train_rows"],
            result["test_rows"],
            result["accuracy"],
        )

    if summary:
        report_path = os.path.join(out_dir, "training_summary.csv")
        pd.DataFrame(summary).to_csv(report_path, index=False)
        logger.info("Saved %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML trainer")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    parser.add_argument("--symbols", nargs="*", help="Override symbols list")
    parser.add_argument(
        "--feature_dir",
        default=os.path.join("data", "features", "technical", "_pruned"),
        help="Path to pruned features",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join("models", "trained_models"),
        help="Output directory for trained models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, args.symbols, args.feature_dir, args.out_dir)


if __name__ == "__main__":
    main()
