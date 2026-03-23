"""
ETL Pipeline for Data Warehouse
Covers: Extract, Transform, Load with error handling and logging
"""

import pandas as pd
import sqlite3
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STAGE 1: EXTRACT
# Pull raw data from various source systems
# ─────────────────────────────────────────────

class Extractor:
    """Extract data from databases, CSV files, and REST APIs."""

    def extract_from_database(self, db_path: str, query: str) -> pd.DataFrame:
        """Pull data from a SQL database."""
        log.info(f"Extracting from database: {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            log.info(f"  → Extracted {len(df):,} rows from DB")
            return df
        except Exception as e:
            log.error(f"DB extraction failed: {e}")
            raise

    def extract_from_csv(self, file_path: str) -> pd.DataFrame:
        """Read data from a CSV file."""
        log.info(f"Extracting from CSV: {file_path}")
        df = pd.read_csv(file_path)
        log.info(f"  → Extracted {len(df):,} rows from CSV")
        return df

    def extract_from_api(self, url: str, params: dict = None) -> pd.DataFrame:
        """Fetch data from a REST API endpoint."""
        log.info(f"Extracting from API: {url}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        # Handles both list and {"results": [...]} shaped responses
        records = data if isinstance(data, list) else data.get("results", [data])
        df = pd.DataFrame(records)
        log.info(f"  → Extracted {len(df):,} rows from API")
        return df


# ─────────────────────────────────────────────
# STAGE 2: TRANSFORM
# Clean, validate, enrich, and reshape the data
# ─────────────────────────────────────────────

class Transformer:
    """Apply all business logic transformations to raw data."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and rows with critical nulls."""
        log.info("Cleaning: removing duplicates and nulls")
        before = len(df)
        df = df.drop_duplicates()
        df = df.dropna(subset=["id", "email"])   # require these columns
        log.info(f"  → Removed {before - len(df):,} invalid rows")
        return df

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and string formats."""
        log.info("Normalizing column names and string values")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
        return df

    def cast_types(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """Cast columns to their expected data types.

        schema = {"signup_date": "datetime", "amount": "float", "age": "int"}
        """
        log.info(f"Casting types: {schema}")
        for col, dtype in schema.items():
            if col not in df.columns:
                continue
            if dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df

    def add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich data with calculated / derived fields."""
        log.info("Adding derived columns")
        df["etl_loaded_at"] = datetime.utcnow()
        if "signup_date" in df.columns:
            df["account_age_days"] = (datetime.utcnow() - df["signup_date"]).dt.days
        if "amount" in df.columns:
            df["amount_bucket"] = pd.cut(
                df["amount"],
                bins=[0, 100, 500, 1000, float("inf")],
                labels=["low", "medium", "high", "premium"],
            )
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag rows that fail business rules."""
        log.info("Validating business rules")
        df["is_valid"] = True
        if "email" in df.columns:
            df.loc[~df["email"].str.contains("@", na=False), "is_valid"] = False
        if "amount" in df.columns:
            df.loc[df["amount"] < 0, "is_valid"] = False
        invalid = (~df["is_valid"]).sum()
        log.warning(f"  → {invalid:,} rows flagged as invalid")
        return df

    def run(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """Run the full transformation chain."""
        df = self.normalize_columns(df)
        df = self.clean(df)
        df = self.cast_types(df, schema)
        df = self.add_derived_columns(df)
        df = self.validate(df)
        log.info(f"Transform complete. Final shape: {df.shape}")
        return df


# ─────────────────────────────────────────────
# STAGE 3: LOAD
# Write transformed data to the data warehouse
# ─────────────────────────────────────────────

class Loader:
    """Load data into the data warehouse (SQLite shown; swap for Redshift/Snowflake/BigQuery)."""

    def __init__(self, warehouse_path: str):
        self.warehouse_path = warehouse_path

    def load(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str = "append",   # "append" | "replace" | "upsert"
        unique_key: str = None,
    ) -> int:
        """Write the dataframe to the warehouse table.

        mode:
          append  – insert all rows (fastest, may duplicate)
          replace – truncate + reload (safe for full refreshes)
          upsert  – update existing rows, insert new ones
        """
        conn = sqlite3.connect(self.warehouse_path)
        log.info(f"Loading {len(df):,} rows → {table_name} (mode={mode})")

        if mode == "replace":
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        elif mode == "append":
            df.to_sql(table_name, conn, if_exists="append", index=False)

        elif mode == "upsert" and unique_key:
            # Simple upsert: delete matching keys then insert
            if unique_key in df.columns:
                keys = tuple(df[unique_key].tolist())
                placeholders = ",".join("?" * len(keys))
                conn.execute(f"DELETE FROM {table_name} WHERE {unique_key} IN ({placeholders})", keys)
                conn.commit()
            df.to_sql(table_name, conn, if_exists="append", index=False)

        conn.close()
        log.info(f"  → Load complete")
        return len(df)


# ─────────────────────────────────────────────
# ORCHESTRATOR
# Wire all three stages into one pipeline run
# ─────────────────────────────────────────────

class ETLPipeline:
    """Orchestrates Extract → Transform → Load."""

    def __init__(self, warehouse_path: str = "warehouse.db"):
        self.extractor   = Extractor()
        self.transformer = Transformer()
        self.loader      = Loader(warehouse_path)
        self.stats       = {}

    def run(
        self,
        source_type: str,
        source: str,
        target_table: str,
        schema: dict,
        load_mode: str = "append",
        unique_key: str = None,
        query: str = None,
    ):
        start = datetime.utcnow()
        log.info(f"=== ETL Pipeline start: {target_table} ===")

        # EXTRACT
        if source_type == "csv":
            raw = self.extractor.extract_from_csv(source)
        elif source_type == "db":
            raw = self.extractor.extract_from_database(source, query or "SELECT * FROM source")
        elif source_type == "api":
            raw = self.extractor.extract_from_api(source)
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        # TRANSFORM
        transformed = self.transformer.run(raw, schema)

        # LOAD (only valid rows)
        valid = transformed[transformed["is_valid"]]
        rows_loaded = self.loader.load(valid, target_table, mode=load_mode, unique_key=unique_key)

        elapsed = (datetime.utcnow() - start).total_seconds()
        self.stats = {
            "table": target_table,
            "rows_extracted": len(raw),
            "rows_loaded": rows_loaded,
            "rows_rejected": len(transformed) - rows_loaded,
            "duration_sec": round(elapsed, 2),
        }
        log.info(f"=== Done in {elapsed:.2f}s | Stats: {self.stats} ===")
        return self.stats


# ─────────────────────────────────────────────
# DEMO — run with sample in-memory CSV data
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import io, os

    SAMPLE_CSV = """id,email,signup_date,amount
1,alice@example.com,2024-01-10,250.0
2,bob@example.com,2024-02-15,75.5
3,bad-email,2024-03-01,50.0
4,carol@example.com,2024-03-20,-10.0
5,dave@example.com,2024-04-05,1200.0
1,alice@example.com,2024-01-10,250.0
"""
    # Write sample CSV
    with open("sample_users.csv", "w") as f:
        f.write(SAMPLE_CSV)

    pipeline = ETLPipeline(warehouse_path="warehouse.db")

    stats = pipeline.run(
        source_type="csv",
        source="sample_users.csv",
        target_table="users",
        schema={"signup_date": "datetime", "amount": "float", "id": "int"},
        load_mode="replace",
        unique_key="id",
    )

    print("\n── Pipeline Summary ──")
    for k, v in stats.items():
        print(f"  {k:<20} {v}")

    # Inspect result
    conn = sqlite3.connect("warehouse.db")
    result = pd.read_sql_query("SELECT id, email, amount, amount_bucket, is_valid FROM users", conn)
    conn.close()
    print("\n── Warehouse Preview ──")
    print(result.to_string(index=False))

    # Cleanup
    for f in ["sample_users.csv", "warehouse.db"]:
        if os.path.exists(f):
            os.remove(f)
