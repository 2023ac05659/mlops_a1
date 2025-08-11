import os
import sqlite3
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


DB_PATH = os.getenv("PRED_DB_PATH", "logs/predictions.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  sepal_length REAL,
  sepal_width REAL,
  petal_length REAL,
  petal_width REAL,
  prediction INTEGER,
  latency_ms REAL
);
"""

def get_conn():
    """
    Returns a connection to the SQLite database.
    Ensures the directory exists before connecting.
    """
    p = Path(DB_PATH)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db(conn: sqlite3.Connection):
    """
    Creates the predictions table if it doesn't exist.
    """
    with conn:
        conn.executescript(SCHEMA)

def log_prediction(conn: sqlite3.Connection, uid: str, payload: Dict[str, Any], pred: int, latency_ms: float):
    """
    Inserts a prediction record into the database.
    """
    with conn:
        conn.execute(
            """
            INSERT INTO predictions (
                id, ts, sepal_length, sepal_width, petal_length, petal_width, prediction, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                datetime.utcnow().isoformat(),
                payload["sepal_length"],
                payload["sepal_width"],
                payload["petal_length"],
                payload["petal_width"],
                pred,
                latency_ms,
            ),
        )
