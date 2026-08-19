"""Write/read the scored portfolio (PD, LGD, EAD, EL, stage, segment
fields) to a lightweight SQLite database.

SQLite (stdlib) chosen over DuckDB: zero extra dependency, sufficient at
~6000 rows, and a portable single-file db that's trivially inspectable
without a running server.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.expected_loss import SCORED_PORTFOLIO_CSV

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "credit_risk.db"
TABLE_NAME = "scored_portfolio"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    account_id INTEGER PRIMARY KEY,
    split TEXT,
    bad INTEGER,
    pd_score REAL,
    lgd REAL,
    ead REAL,
    el REAL,
    stage INTEGER,
    credit_band TEXT,
    loan REAL,
    mortdue REAL,
    value REAL,
    reason TEXT,
    job TEXT,
    yoj REAL,
    derog REAL,
    delinq REAL,
    clage REAL,
    ninq REAL,
    clno REAL,
    debtinc REAL
);
"""

COLUMN_MAP = {
    "account_id": "account_id",
    "split": "split",
    "BAD": "bad",
    "PD": "pd_score",
    "LGD": "lgd",
    "EAD": "ead",
    "EL": "el",
    "STAGE": "stage",
    "CREDIT_BAND": "credit_band",
    "LOAN": "loan",
    "MORTDUE": "mortdue",
    "VALUE": "value",
    "REASON": "reason",
    "JOB": "job",
    "YOJ": "yoj",
    "DEROG": "derog",
    "DELINQ": "delinq",
    "CLAGE": "clage",
    "NINQ": "ninq",
    "CLNO": "clno",
    "DEBTINC": "debtinc",
}


def write_scored_portfolio(
    df: pd.DataFrame, db_path: Path = DB_PATH, table_name: str = TABLE_NAME, if_exists: str = "replace"
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    table = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}" if if_exists == "replace" else "SELECT 1")
        conn.execute(SCHEMA_SQL)
        table.to_sql(table_name, conn, if_exists="append", index=False)


def read_scored_portfolio(db_path: Path = DB_PATH, table_name: str = TABLE_NAME) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)


def main() -> None:
    scored = pd.read_csv(SCORED_PORTFOLIO_CSV)
    write_scored_portfolio(scored)
    print(f"Wrote {len(scored)} rows to {DB_PATH} ({TABLE_NAME})")

    check = read_scored_portfolio()
    assert len(check) == len(scored), "row count mismatch after write/read"

    sample = check.iloc[0]
    expected_el = sample["pd_score"] * sample["lgd"] * sample["ead"]
    assert abs(sample["el"] - expected_el) < 1e-6, "EL spot-check failed"
    print(f"Spot-check OK: EL ({sample['el']:.4f}) == PD*LGD*EAD ({expected_el:.4f})")


if __name__ == "__main__":
    main()
