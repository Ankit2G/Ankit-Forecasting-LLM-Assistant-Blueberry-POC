# services/dbx_sql.py
# ------------------------------------------------------------
# Databricks SQL Connector (Production-Ready POC + Offline-Friendly)
#
# ✅ Works in TWO modes:
# 1) ONLINE (default): connects to Databricks SQL Warehouse using env vars
# 2) OFFLINE (no creds): UI still runs; queries return empty results + friendly status
#
# Required env vars for ONLINE mode:
#   DBX_SQL_HOST        e.g. "adb-1234567890123456.7.azuredatabricks.net" (NO https://)
#   DBX_SQL_HTTP_PATH   e.g. "/sql/1.0/warehouses/xxxxxxxxxxxxxxxx"
#   DBX_SQL_TOKEN       Databricks PAT
#
# Optional env vars:
#   DBX_SQL_CATALOG     e.g. "ppe_dscoe"
#   DBX_SQL_SCHEMA      e.g. "default"
#   DBX_SQL_TIMEOUT_S   e.g. "30"
#   DBX_SQL_MAX_RETRIES e.g. "2"
#   DBX_SQL_OFFLINE_OK  "true"/"false" (default: true) -> allow UI to run without creds
#
# Notes:
# - Requires: databricks-sql-connector
#     pip install databricks-sql-connector
# - This client is READ-ONLY by default (SELECT/WITH/SHOW/DESCRIBE/EXPLAIN).
# - No multi-statement SQL allowed.
# - Parameter placeholders use '?' (DB-API style).
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Protocol

# Public params type accepted by query()
Params = Optional[Union[Tuple[Any, ...], Sequence[Any]]]


# -----------------------------
# Exceptions
# -----------------------------
class DBXSQLConfigError(RuntimeError):
    """Raised when Databricks SQL configuration is missing/invalid."""


class DBXSQLRuntimeError(RuntimeError):
    """Raised when a query fails or violates safety guards."""


# -----------------------------
# Internal helpers
# -----------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _normalize_params(params: Params) -> Optional[Tuple[Any, ...]]:
    """Normalize params into a tuple (or None) to match expected query_fn signatures."""
    if params is None:
        return None
    if isinstance(params, tuple):
        return params
    try:
        return tuple(params)
    except Exception:
        # If something odd is passed, fail closed.
        raise DBXSQLRuntimeError("Invalid params. Expected tuple or sequence of values.")


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class DBXSQLConfig:
    host: str
    http_path: str
    token: str
    catalog: Optional[str] = None
    schema: Optional[str] = None
    timeout_s: int = 30
    max_retries: int = 2
    offline_ok: bool = True

    @staticmethod
    def from_env() -> "DBXSQLConfig":
        host = (os.getenv("DBX_SQL_HOST", "") or "").strip()
        http_path = (os.getenv("DBX_SQL_HTTP_PATH", "") or "").strip()
        token = (os.getenv("DBX_SQL_TOKEN", "") or "").strip()

        # Normalize host if user includes protocol and/or trailing slashes
        host = host.replace("https://", "").replace("http://", "").strip().strip("/")

        catalog = (os.getenv("DBX_SQL_CATALOG", "") or "").strip() or None
        schema = (os.getenv("DBX_SQL_SCHEMA", "") or "").strip() or None
        timeout_s = _env_int("DBX_SQL_TIMEOUT_S", 30)
        max_retries = _env_int("DBX_SQL_MAX_RETRIES", 2)
        offline_ok = _env_bool("DBX_SQL_OFFLINE_OK", True)

        return DBXSQLConfig(
            host=host,
            http_path=http_path,
            token=token,
            catalog=catalog,
            schema=schema,
            timeout_s=timeout_s,
            max_retries=max_retries,
            offline_ok=offline_ok,
        )

    def is_configured(self) -> bool:
        return bool(self.host and self.http_path and self.token)


# -----------------------------
# Client interface (for typing)
# -----------------------------
class _Client(Protocol):
    def query(self, sql_text: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]: ...
    def close(self) -> None: ...


# -----------------------------
# Offline client
# -----------------------------
class OfflineDatabricksSQLClient:
    """
    Used when credentials are not present AND offline_ok is true.
    Returns empty results so Streamlit UI can render without hard failing.
    """

    def __init__(self, reason: str):
        self.reason = reason

    def query(self, sql_text: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
        return []

    def close(self) -> None:
        return


# -----------------------------
# Online client (Databricks)
# -----------------------------
class DatabricksSQLClient:
    """
    Thin wrapper around databricks-sql-connector.

    - Thread-local connection (safe for Streamlit)
    - Optional USE CATALOG/SCHEMA
    - Retries transient failures
    - Read-only + single-statement guards
    """

    def __init__(self, config: Optional[DBXSQLConfig] = None):
        self.config = config or DBXSQLConfig.from_env()

        if not self.config.is_configured():
            raise DBXSQLConfigError("Databricks SQL config is incomplete (missing host/http_path/token).")

        try:
            from databricks import sql as dbsql  # type: ignore
        except Exception as e:  # pragma: no cover
            raise DBXSQLConfigError(
                "databricks-sql-connector is not installed. Install with: pip install databricks-sql-connector"
            ) from e

        self._dbsql = dbsql
        self._local = threading.local()

    # -------- Connection management --------
    def _get_conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        # NOTE: databricks-sql-connector doesn't expose a clean per-query timeout;
        # keep retries limited and let the connector handle socket timeouts internally.
        conn = self._dbsql.connect(
            server_hostname=self.config.host,
            http_path=self.config.http_path,
            access_token=self.config.token,
            _user_agent_entry="forecast_poc_streamlit",
        )

        # Set default catalog/schema for the session (best effort)
        try:
            if self.config.catalog:
                self._exec_no_fetch(conn, f"USE CATALOG {self._quote_ident(self.config.catalog)}")
            if self.config.schema:
                self._exec_no_fetch(conn, f"USE SCHEMA {self._quote_ident(self.config.schema)}")
        except Exception:
            # Non-fatal — some workspaces/warehouses may restrict USE statements
            pass

        self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    # -------- Public query API --------
    def query(self, sql_text: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
        """
        Execute a single read-only statement and return rows as list of dicts.

        Params must match '?' placeholders:
            query("SELECT * FROM t WHERE a=? AND b=?", (1, "x"))
        """
        self._validate_sql(sql_text)

        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                conn = self._get_conn()
                with conn.cursor() as cur:
                    if params is None:
                        cur.execute(sql_text)
                    else:
                        cur.execute(sql_text, params)

                    if cur.description is None:
                        return []

                    cols = [c[0] for c in cur.description]
                    rows = cur.fetchall()
                    return [dict(zip(cols, r)) for r in rows]

            except Exception as e:
                last_err = e
                # reconnect on failure
                self.close()

                if attempt < self.config.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

        raise DBXSQLRuntimeError(f"Databricks SQL query failed: {last_err}") from last_err

    # -------- Internal helpers --------
    def _exec_no_fetch(self, conn, sql_text: str) -> None:
        self._validate_sql(sql_text, allow_non_select=True)
        with conn.cursor() as cur:
            cur.execute(sql_text)

    @staticmethod
    def _quote_ident(ident: str) -> str:
        ident = ident.replace("`", "")
        return f"`{ident}`"

    @staticmethod
    def _validate_sql(sql_text: str, allow_non_select: bool = False) -> None:
        """
        Prevent multi-statement execution and dangerous statements.

        - Default: read-only only (SELECT/WITH/SHOW/DESCRIBE/EXPLAIN)
        - When allow_non_select=True: allow USE CATALOG/SCHEMA (session-scoped)
        """
        s = (sql_text or "").strip()
        if not s:
            raise DBXSQLRuntimeError("Empty SQL statement.")

        # Disallow multiple statements (simple guard)
        # NOTE: This is a conservative check; it may reject queries with ';' in string literals.
        if ";" in s.rstrip(";"):
            raise DBXSQLRuntimeError("Multi-statement SQL is not allowed.")

        lower = s.strip().lower()

        if allow_non_select and (lower.startswith("use catalog") or lower.startswith("use schema")):
            return

        # Block mutating statements ALWAYS (word-boundary based)
        blocked_words = [
            "drop",
            "truncate",
            "delete",
            "update",
            "merge",
            "insert",
            "alter",
            "create",
            "grant",
            "revoke",
        ]
        if re.search(r"\b(" + "|".join(blocked_words) + r")\b", lower):
            raise DBXSQLRuntimeError("Only read-only queries are allowed from the app client.")

        allowed_starts = ("select", "with", "show", "describe", "explain")
        if not lower.startswith(allowed_starts):
            raise DBXSQLRuntimeError("Only read-only queries are allowed (SELECT/WITH/SHOW/DESCRIBE/EXPLAIN).")


# ------------------------------------------------------------
# Singleton + wrappers
# ------------------------------------------------------------
_client_singleton: Optional[_Client] = None
_client_lock = threading.Lock()


def get_client() -> _Client:
    """
    Returns:
      - DatabricksSQLClient when configured
      - OfflineDatabricksSQLClient when not configured and DBX_SQL_OFFLINE_OK=true
    """
    global _client_singleton
    with _client_lock:
        if _client_singleton is not None:
            return _client_singleton

        cfg = DBXSQLConfig.from_env()
        if not cfg.is_configured():
            if cfg.offline_ok:
                _client_singleton = OfflineDatabricksSQLClient(
                    reason="Databricks SQL credentials not set. Running in OFFLINE UI mode."
                )
                return _client_singleton
            missing = [k for k in ["DBX_SQL_HOST", "DBX_SQL_HTTP_PATH", "DBX_SQL_TOKEN"] if not (os.getenv(k) or "").strip()]
            raise DBXSQLConfigError(
                f"Missing Databricks SQL configuration: {', '.join(missing)}. "
                "Set environment variables or set DBX_SQL_OFFLINE_OK=true to run UI without data."
            )

        _client_singleton = DatabricksSQLClient(cfg)
        return _client_singleton


def is_configured() -> bool:
    """True if DBX_SQL_HOST/HTTP_PATH/TOKEN are present."""
    return DBXSQLConfig.from_env().is_configured()


# Common alias some files like to use
def is_online() -> bool:
    """Alias for is_configured()."""
    return is_configured()


def offline_reason() -> Optional[str]:
    """
    If running offline, returns a human-readable reason string (else None).
    Useful for banners/tooltips in Streamlit.
    """
    c = get_client()
    if isinstance(c, OfflineDatabricksSQLClient):
        return c.reason
    return None


def query(sql_text: str, params: Params = None) -> List[Dict[str, Any]]:
    """
    Functional wrapper:
        from services.dbx_sql import query
        rows = query("SELECT 1 AS x")
    """
    client = get_client()
    norm = _normalize_params(params)
    return client.query(sql_text, norm)


def close() -> None:
    """Close the underlying connection for this thread (best effort)."""
    client = get_client()
    try:
        client.close()
    except Exception:
        pass


__all__ = [
    "DBXSQLConfig",
    "DBXSQLConfigError",
    "DBXSQLRuntimeError",
    "get_client",
    "is_configured",
    "is_online",
    "offline_reason",
    "query",
    "close",
]
