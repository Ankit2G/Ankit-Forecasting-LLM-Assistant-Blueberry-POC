# services/queries.py
# ------------------------------------------------------------
# Production-ready POC query layer (safe + reusable)
#
# Purpose:
# - Centralize all SQL used by the Streamlit app
# - Keep queries parameterized, consistent, and testable
# - Provide clean, buyer-friendly outputs (DataFrames) to pages
#
# Design:
# - No free-form SQL execution exposed here
# - Each function returns QueryResult(df, sql, params) so UI can optionally show SQL
# - Uses an injected query_fn(sql, params) -> list[dict]
#
# Example:
#   from services.queries import QueryService, TableRegistry
#   qs = QueryService(query_fn=run_query, tables=TableRegistry(...))
#   res = qs.forecast_table(run_id="2026-01-05", item="Combined")
#   df = res.df
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

QueryFn = Callable[[str, Optional[Tuple[Any, ...]]], List[Dict[str, Any]]]
Params = Optional[Union[Tuple[Any, ...], Sequence[Any]]]


# -----------------------------
# Types
# -----------------------------
@dataclass(frozen=True)
class TableRegistry:
    forecast: str  # ppe_dscoe.default.placeholder_forecast_ouputtable
    backtest: str  # ppe_dscoe.default.placeholder_backtest_outputtable
    actuals: str   # ppe_dscoe.default.ex_sales_table_historical
    dnd_company: str
    dnd_depot_current: str
    dnd_depot_hist: str


@dataclass(frozen=True)
class QueryResult:
    df: pd.DataFrame
    sql: str
    params: Optional[Tuple[Any, ...]]


# -----------------------------
# Internal helpers
# -----------------------------
def _to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _normalize_params(params: Params) -> Optional[Tuple[Any, ...]]:
    if params is None:
        return None
    if isinstance(params, tuple):
        return params
    try:
        return tuple(params)
    except Exception:
        raise ValueError("Invalid params. Expected tuple or sequence of values.")


def _safe_limit(limit: int, max_limit: int) -> int:
    try:
        limit = int(limit)
    except Exception:
        limit = 1
    if limit < 1:
        limit = 1
    if limit > max_limit:
        limit = max_limit
    return limit


def _coerce_numeric(df: pd.DataFrame, cols: List[str], int_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if int_cols:
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def _coerce_datestring(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # Keep as string for display, but ensure consistent dtype (object) and drop timezone surprises.
    if df.empty or col not in df.columns:
        return df
    df[col] = df[col].astype(str)
    return df


# -----------------------------
# QueryService
# -----------------------------
class QueryService:
    """
    Central query layer. All queries are parameterized and safe.

    query_fn: callable(sql_text, params_tuple_or_none) -> list[dict]
    """

    def __init__(self, query_fn: QueryFn, tables: TableRegistry, max_rows: int = 20000):
        self.query_fn = query_fn
        self.tables = tables
        self.max_rows = int(max_rows)

    # -------- Forecasts --------
    def forecast_run_ids(self, limit: int = 60) -> QueryResult:
        limit = _safe_limit(limit, 500)
        sql = f"""
            SELECT DISTINCT CAST(run_id AS STRING) AS run_id
            FROM {self.tables.forecast}
            ORDER BY run_id DESC
            LIMIT {limit}
        """
        rows = self.query_fn(sql, None)
        df = _to_df(rows)
        return QueryResult(df=df, sql=sql, params=None)

    def forecast_depots_for_run(self, run_id: str, item: Optional[str] = None) -> QueryResult:
        where = "WHERE run_id = ?"
        params: List[Any] = [run_id]
        if item and item != "All":
            where += " AND ITEM = ?"
            params.append(item)

        sql = f"""
            SELECT DISTINCT DEPOT
            FROM {self.tables.forecast}
            {where}
            ORDER BY DEPOT
        """
        rows = self.query_fn(sql, tuple(params))
        df = _to_df(rows)
        df = _coerce_numeric(df, cols=["DEPOT"], int_cols=["DEPOT"])
        return QueryResult(df=df, sql=sql, params=tuple(params))

    def forecast_target_weeks(self, run_id: str, depot: Optional[int] = None, item: Optional[str] = None) -> QueryResult:
        where = "WHERE run_id = ?"
        params: List[Any] = [run_id]

        if depot is not None:
            where += " AND DEPOT = ?"
            params.append(int(depot))
        if item and item != "All":
            where += " AND ITEM = ?"
            params.append(item)

        sql = f"""
            SELECT DISTINCT CAST(target_week_start AS STRING) AS target_week_start
            FROM {self.tables.forecast}
            {where}
            ORDER BY target_week_start
        """
        rows = self.query_fn(sql, tuple(params))
        df = _to_df(rows)
        df = _coerce_datestring(df, "target_week_start")
        return QueryResult(df=df, sql=sql, params=tuple(params))

    def forecast_table(
        self,
        run_id: str,
        depot: Optional[int] = None,
        item: str = "All",
        horizon: Optional[int] = None,
        week_start_min: Optional[str] = None,
        week_start_max: Optional[str] = None,
        limit: int = 10000,
    ) -> QueryResult:
        limit = _safe_limit(limit, self.max_rows)

        where = "WHERE run_id = ?"
        params: List[Any] = [run_id]

        if depot is not None:
            where += " AND DEPOT = ?"
            params.append(int(depot))
        if item and item != "All":
            where += " AND ITEM = ?"
            params.append(item)
        if horizon is not None:
            where += " AND horizon = ?"
            params.append(int(horizon))
        if week_start_min:
            where += " AND target_week_start >= ?"
            params.append(week_start_min)
        if week_start_max:
            where += " AND target_week_start <= ?"
            params.append(week_start_max)

        sql = f"""
            SELECT
                DEPOT,
                ITEM,
                CAST(target_week_start AS STRING) AS target_week_start,
                horizon,
                yhat
            FROM {self.tables.forecast}
            {where}
            ORDER BY DEPOT, ITEM, target_week_start, horizon
            LIMIT {limit}
        """
        rows = self.query_fn(sql, tuple(params))
        df = _to_df(rows)
        df = _coerce_numeric(df, cols=["DEPOT", "horizon", "yhat"], int_cols=["DEPOT", "horizon"])
        df = _coerce_datestring(df, "target_week_start")
        return QueryResult(df=df, sql=sql, params=tuple(params))

    def forecast_totals(
        self,
        run_id: str,
        depot: Optional[int] = None,
        item: str = "All",
        horizon: Optional[int] = None,
    ) -> QueryResult:
        where = "WHERE run_id = ?"
        params: List[Any] = [run_id]

        if depot is not None:
            where += " AND DEPOT = ?"
            params.append(int(depot))
        if item and item != "All":
            where += " AND ITEM = ?"
            params.append(item)
        if horizon is not None:
            where += " AND horizon = ?"
            params.append(int(horizon))

        sql = f"""
            SELECT
                CAST(run_id AS STRING) AS run_id,
                ITEM,
                horizon,
                SUM(yhat) AS total_forecast_units
            FROM {self.tables.forecast}
            {where}
            GROUP BY run_id, ITEM, horizon
            ORDER BY ITEM, horizon
        """
        rows = self.query_fn(sql, tuple(params))
        df = _to_df(rows)
        df = _coerce_numeric(df, cols=["horizon", "total_forecast_units"], int_cols=["horizon"])
        return QueryResult(df=df, sql=sql, params=tuple(params))

    def forecast_trend_across_runs(
        self,
        depot: int,
        item: str,
        horizon: int,
        target_week_start: Optional[str] = None,
        runs_back: int = 26,
    ) -> QueryResult:
        runs_back = _safe_limit(runs_back, 200)

        where = "WHERE DEPOT = ? AND ITEM = ? AND horizon = ?"
        params: List[Any] = [int(depot), item, int(horizon)]
        if target_week_start:
            where += " AND target_week_start = ?"
            params.append(target_week_start)

        sql = f"""
            WITH base AS (
                SELECT
                    CAST(run_id AS STRING) AS run_id,
                    CAST(target_week_start AS STRING) AS target_week_start,
                    yhat
                FROM {self.tables.forecast}
                {where}
            ),
            limited AS (
                SELECT *
                FROM base
                ORDER BY run_id DESC
                LIMIT {runs_back}
            )
            SELECT *
            FROM limited
            ORDER BY run_id
        """
        rows = self.query_fn(sql, tuple(params))
        df = _to_df(rows)
        df = _coerce_numeric(df, cols=["yhat"])
        df = _coerce_datestring(df, "target_week_start")
        return QueryResult(df=df, sql=sql, params=tuple(params))

    # -------- Backtest / Performance --------
    def backtest_eval_weeks(self, limit: int = 80) -> QueryResult:
        limit = _safe_limit(limit, 500)
        sql = f"""
            SELECT DISTINCT CAST(eval_week_start AS STRING) AS eval_week_start
            FROM {self.tables.backtest}
            ORDER BY eval_week_start DESC
            LIMIT {limit}
        """
        rows = self.query_fn(sql, None)
        df = _to_df(rows)
        df = _coerce_datestring(df, "eval_week_start")
        return QueryResult(df=df, sql=sql, params=None)

    def backtest_overall_metrics(
        self,
        eval_week_start: str,
        depot: Optional[int] = None,
        item: Optional[str] = None,
    ) -> QueryResult:
        where = "WHERE eval_week_start = ?"
        params: List[Any] = [eval_week_start]

        if depot is not None:
            where += " AND DEPOT = ?"
            params.append(int(depot))
        if item and item != "All":
            where += " AND ITEM = ?"
            params.append(item)

        sql = f"""
            SELECT
                SUM(abs_error) AS sum_abs_error,
                SUM(y) AS sum_actual,
                SUM(yhat) AS sum_forecast,
                AVG(pct_error) AS avg_pct_error,
                AVG(ABS(pct_error)) AS avg_abs_pct_error
            FROM {self.tables.backtest}
            {where}
        """
        rows = self.query_fn(sql, tuple(params))
        df = _to_df(rows)
        df = _coerce_numeric(df, cols=["sum_abs_error", "sum_actual", "sum_forecast", "avg_pct_error", "avg_abs_pct_error"])
        return QueryResult(df=df, sql=sql, params=tuple(params))

    def backtest_by_item(self, eval_week_start: str, depot: Optional[int] = None) -> QueryResult:
        where = "WHERE eval_week_start = ?"
        params: List[Any] = [eval_week_start]
        if depot is not None:
            where += " AND DEPOT = ?"
            params.append(int(depot))

        sql = f"""
            SELECT
                ITEM,
                SUM(abs_error) AS sum_abs_error,
                SUM(y) AS sum_actual,
                SUM(yhat) AS sum_forecast,
                AVG(pct_error) AS avg_pct_error,
                AVG(ABS(pct_error)) AS avg_abs_pct_error
            FROM {self.tables.backtest}
            {where}
            GROUP BY ITEM
            ORDER BY ITEM
        """
        rows = self.query_fn(sql, tuple(params))
        df = _to_df(rows)
        df = _coerce_numeric(df, cols=["sum_abs_error", "sum_actual", "sum_forecast", "avg_pct_error", "avg_abs_pct_error"])
        return QueryResult(df=df, sql=sql, params=tuple(params))

    def backtest_by_depot(self, eval_week_start: str, item: Optional[str] = None) -> QueryResult:
        where = "WHERE eval_week_start = ?"
        params: List[Any] = [eval_week_start]
        if item and item != "All":
            where += " AND ITEM = ?"
            params.append(item)

        sql = f"""
            SELECT
                DEPOT,
                SUM(abs_error) AS sum_abs_error,
                SUM(y) AS sum_actual,
                SUM(yhat) AS sum_forecast,
                AVG(pct_error) AS avg_pct_error,
                AVG(ABS(pct_error)) AS avg_abs_pct_error
            FROM {self.tables.backtest}
            {where}
            GROUP BY DEPOT
            ORDER BY (SUM(abs_error) / NULLIF(SUM(y), 0)) DESC
        """
        rows = self.query_fn(sql, tuple(params))
        df = _to_df(rows)
        df = _coerce_numeric(
            df,
            cols=["DEPOT", "sum_abs_error", "sum_actual", "sum_forecast", "avg_pct_error", "avg_abs_pct_error"],
            int_cols=["DEPOT"],
        )
        return QueryResult(df=df, sql=sql, params=tuple(params))

    def backtest_worst_offenders(
        self,
        eval_week_start: str,
        depot: Optional[int] = None,
        item: Optional[str] = None,
        top_n: int = 15,
    ) -> QueryResult:
        top_n = _safe_limit(top_n, 200)

        where = "WHERE eval_week_start = ?"
        params: List[Any] = [eval_week_start]
        if depot is not None:
            where += " AND DEPOT = ?"
            params.append(int(depot))
        if item and item != "All":
            where += " AND ITEM = ?"
            params.append(item)

        sql = f"""
            SELECT
                DEPOT,
                ITEM,
                CAST(eval_week_start AS STRING) AS eval_week_start,
                y AS actual_units,
                yhat AS forecast_units,
                abs_error,
                pct_error
            FROM {self.tables.backtest}
            {where}
            ORDER BY ABS(pct_error) DESC
            LIMIT {top_n}
        """
        rows = self.query_fn(sql, tuple(params))
        df = _to_df(rows)
        df = _coerce_numeric(df, cols=["DEPOT", "actual_units", "forecast_units", "abs_error", "pct_error"], int_cols=["DEPOT"])
        df = _coerce_datestring(df, "eval_week_start")
        return QueryResult(df=df, sql=sql, params=tuple(params))

    def backtest_trend_over_weeks(
        self,
        weeks_back: int = 26,
        depot: Optional[int] = None,
        item: Optional[str] = None,
    ) -> QueryResult:
        weeks_back = _safe_limit(weeks_back, 260)

        where = "WHERE 1=1"
        params: List[Any] = []
        if depot is not None:
            where += " AND DEPOT = ?"
            params.append(int(depot))
        if item and item != "All":
            where += " AND ITEM = ?"
            params.append(item)

        sql = f"""
            WITH base AS (
                SELECT
                    eval_week_start,
                    SUM(abs_error) AS sum_abs_error,
                    SUM(y) AS sum_actual,
                    SUM(yhat) AS sum_forecast
                FROM {self.tables.backtest}
                {where}
                GROUP BY eval_week_start
            ),
            limited AS (
                SELECT *
                FROM base
                ORDER BY eval_week_start DESC
                LIMIT {weeks_back}
            )
            SELECT
                CAST(eval_week_start AS STRING) AS eval_week_start,
                sum_abs_error,
                sum_actual,
                sum_forecast,
                (sum_abs_error / NULLIF(sum_actual, 0)) AS wape,
                ((sum_forecast - sum_actual) / NULLIF(sum_actual, 0)) AS bias
            FROM limited
            ORDER BY eval_week_start
        """
        rows = self.query_fn(sql, _normalize_params(tuple(params)) if params else None)
        df = _to_df(rows)
        df = _coerce_numeric(df, cols=["sum_abs_error", "sum_actual", "sum_forecast", "wape", "bias"])
        df = _coerce_datestring(df, "eval_week_start")
        return QueryResult(df=df, sql=sql, params=tuple(params) if params else None)

    # -------- D&D --------
    def dnd_company_history(self, weeks_back: int = 52) -> QueryResult:
        weeks_back = _safe_limit(weeks_back, 260)
        sql = f"""
            SELECT
                CAST(FSCL_WK_START_DATE AS STRING) AS FSCL_WK_START_DATE,
                Total_Units_Sold,
                Total_Sales_Amt,
                Total_DND_QTY,
                Total_Units,
                Total_DND_$_loss,
                (SUM(Total_DND_QTY) / NULLIF(SUM(Total_Units), 0)) AS dnd_rate
            FROM {self.tables.dnd_company}
            GROUP BY FSCL_WK_START_DATE, Total_Units_Sold, Total_Sales_Amt, Total_DND_QTY, Total_Units, Total_DND_$_loss
            ORDER BY FSCL_WK_START_DATE DESC
            LIMIT {weeks_back}
        """
        rows = self.query_fn(sql, None)
        df = _to_df(rows)
        df = _coerce_numeric(
            df,
            cols=["Total_Units_Sold", "Total_Sales_Amt", "Total_DND_QTY", "Total_Units", "Total_DND_$_loss", "dnd_rate"],
        )
        df = _coerce_datestring(df, "FSCL_WK_START_DATE")
        return QueryResult(df=df, sql=sql, params=None)

    def dnd_depot_available_weeks(self, limit: int = 40) -> QueryResult:
        limit = _safe_limit(limit, 200)
        sql = f"""
            SELECT DISTINCT CAST(FSCL_WK_START_DATE AS STRING) AS FSCL_WK_START_DATE
            FROM {self.tables.dnd_depot_current}
            ORDER BY FSCL_WK_START_DATE DESC
            LIMIT {limit}
        """
        rows = self.query_fn(sql, None)
        df = _to_df(rows)
        df = _coerce_datestring(df, "FSCL_WK_START_DATE")
        return QueryResult(df=df, sql=sql, params=None)

    def dnd_depot_snapshot(self, week_start: Optional[str] = None) -> QueryResult:
        params: List[Any] = []
        where = ""
        if week_start:
            where = "WHERE FSCL_WK_START_DATE = ?"
            params.append(week_start)

        sql = f"""
            SELECT
                DEPOT,
                CAST(FSCL_WK_START_DATE AS STRING) AS FSCL_WK_START_DATE,
                Volume,
                Total_DND_QTY,
                Total_Units,
                Total_Sales_Amt,
                Total_DND_$_loss,
                (SUM(Total_DND_QTY) / NULLIF(SUM(Total_Units), 0)) AS dnd_rate
            FROM {self.tables.dnd_depot_current}
            {where}
            GROUP BY DEPOT, FSCL_WK_START_DATE, Volume, Total_DND_QTY, Total_Units, Total_Sales_Amt, Total_DND_$_loss
            ORDER BY DEPOT
        """
        rows = self.query_fn(sql, _normalize_params(tuple(params)) if params else None)
        df = _to_df(rows)
        df = _coerce_numeric(
            df,
            cols=["DEPOT", "Volume", "Total_DND_QTY", "Total_Units", "Total_Sales_Amt", "Total_DND_$_loss", "dnd_rate"],
            int_cols=["DEPOT"],
        )
        df = _coerce_datestring(df, "FSCL_WK_START_DATE")
        return QueryResult(df=df, sql=sql, params=tuple(params) if params else None)

    def dnd_depot_history(self, depot: int, weeks_back: int = 52) -> QueryResult:
        weeks_back = _safe_limit(weeks_back, 260)

        sql = f"""
            SELECT
                DEPOT,
                CAST(FSCL_WK_START_DATE AS STRING) AS FSCL_WK_START_DATE,
                Volume,
                Total_DND_QTY,
                Total_Units,
                Total_Sales_Amt,
                Total_DND_$_loss,
                (SUM(Total_DND_QTY) / NULLIF(SUM(Total_Units), 0)) AS dnd_rate
            FROM {self.tables.dnd_depot_hist}
            WHERE DEPOT = ?
            GROUP BY DEPOT, FSCL_WK_START_DATE, Volume, Total_DND_QTY, Total_Units, Total_Sales_Amt, Total_DND_$_loss
            ORDER BY FSCL_WK_START_DATE DESC
            LIMIT {weeks_back}
        """
        rows = self.query_fn(sql, (int(depot),))
        df = _to_df(rows)
        df = _coerce_numeric(
            df,
            cols=["DEPOT", "Volume", "Total_DND_QTY", "Total_Units", "Total_Sales_Amt", "Total_DND_$_loss", "dnd_rate"],
            int_cols=["DEPOT"],
        )
        df = _coerce_datestring(df, "FSCL_WK_START_DATE")
        return QueryResult(df=df, sql=sql, params=(int(depot),))
