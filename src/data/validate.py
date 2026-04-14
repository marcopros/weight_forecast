"""Data validation using Pandera schemas.

Validates data at pipeline boundaries to catch quality issues early.
Can be run standalone: python -m src.data.validate
"""

import pandas as pd
import pandera.pandas as pa
import yaml


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Schemas ─────────────────────────────────────────────────────────────────
receivals_schema = pa.DataFrameSchema(
    {
        "rm_id": pa.Column(int, nullable=False),
        "net_weight": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "date_arrival": pa.Column("object", nullable=False),
    },
    coerce=True,
    strict=False,  # allow extra columns
)

master_schema = pa.DataFrameSchema(
    {
        "rm_id": pa.Column(int, nullable=False),
        "date": pa.Column("datetime64[ns]", nullable=False),
        "net_weight": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "cumulative_weight": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
    },
    coerce=True,
    strict=False,
)


def featured_schema(feature_cols: list[str]) -> pa.DataFrameSchema:
    """Build a schema that validates all feature columns exist and are numeric."""
    columns = {
        "rm_id": pa.Column(int, nullable=False),
        "date": pa.Column("datetime64[ns]", nullable=False),
    }
    for col in feature_cols:
        columns[col] = pa.Column(float, nullable=True, coerce=True)
    return pa.DataFrameSchema(columns, coerce=True, strict=False)


prediction_schema = pa.DataFrameSchema(
    {
        "rm_id": pa.Column(int, nullable=False),
        "date": pa.Column("datetime64[ns]", nullable=False),
        "predicted_cumulative_weight": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
    },
    coerce=True,
    strict=False,
)


# ── Validation helpers ──────────────────────────────────────────────────────
def validate_receivals(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return receivals data. Raises SchemaError on failure."""
    return receivals_schema.validate(df)


def validate_master(df: pd.DataFrame) -> pd.DataFrame:
    """Validate master table after cleaning."""
    return master_schema.validate(df)


def validate_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Validate featured master table."""
    schema = featured_schema(feature_cols)
    return schema.validate(df)


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate prediction output."""
    return prediction_schema.validate(df)


def run(config_path: str = "configs/params.yaml"):
    """Validate all pipeline artifacts."""
    cfg = load_config(config_path)
    feature_cols = cfg["features"]["feature_cols"]
    errors = []

    # Validate master table
    try:
        master = pd.read_csv(cfg["data"]["processed"]["master"], parse_dates=["date"])
        validate_master(master)
        print(f"✓ Master table valid ({len(master)} rows)")

        # Check features exist
        validate_features(master, feature_cols)
        print(f"✓ Featured master table valid ({len(feature_cols)} features)")
    except pa.errors.SchemaError as e:
        errors.append(f"Master table validation failed: {e}")
        print(f"✗ Master table: {e}")

    # Validate predictions if they exist
    try:
        preds = pd.read_csv(cfg["data"]["processed"]["predictions"], parse_dates=["date"])
        validate_predictions(preds)
        print(f"✓ Predictions valid ({len(preds)} rows)")
    except FileNotFoundError:
        print("⊘ No predictions file found (skipped)")
    except pa.errors.SchemaError as e:
        errors.append(f"Predictions validation failed: {e}")
        print(f"✗ Predictions: {e}")

    if errors:
        print(f"\n{len(errors)} validation error(s) found!")
        return False
    print("\nAll validations passed.")
    return True


if __name__ == "__main__":
    run()
