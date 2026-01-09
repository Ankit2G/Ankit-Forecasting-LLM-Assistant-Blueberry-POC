# pages/3_Model_Performance.py
# ------------------------------------------------------------
# Model Performance (Backtesting) — Buyer-Facing, Production-Grade POC
#
# Goals:
# - Deterministic (non-LLM) performance metrics from your backtest Delta table
# - Buyer-friendly KPIs + breakdowns + worst offenders + recent trend
# - Offline-safe: renders UI even if Databricks SQL is not configured
#
# Data source (from app.py TableRegistry):
#   ppe_dscoe.default.placeholder_backtest_outputtable  (default in TableRegistry)
#
# Expected columns (minimum; names can be aliased in queries if needed):
#   - eval_week_start (date)
#   - DEPOT (int)
#   - ITEM (string)
#   - y (actual units, numeric)
#   - yhat (forecast units, numeric)
#   - abs_error (numeric)      [if missing, computed as ABS(yhat - y)]
#   - pct_error (numeric)      [if missing, computed as (yhat - y)/NULLIF(y,0)]
#
# Notes:
# - This page does not depend on app.py imports (avoids circular imports).
# - Uses services.dbx_sql.query + st.session_state["tables"].
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import altair as alt

from services.dbx_sql import is_configured as dbx_is_configured, query as dbx_query

import streamlit as st

tables = st.session_state.get("tables")
qs = st.session_state.get("query_service")

if tables is None or qs is None:
    st.error("App is not initialized. Open the Home page (app.py) first, then return here.")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def _get_tables():
    if "tables" not in st.session_state:
        st.error("Tables registry not initialized. Please reload the app.")
        st.stop()
    return st.session_state["tables"]


def _sql_df(sql: str, params: Optional[Tuple[Any, ...]] = None) -> pd.DataFrame:
    """
    Normalizes dbx_query outputs to a DataFrame.
    Supports dbx_query returning:
      - pandas.DataFrame
      - list[dict]
      - list[tuple] (with cursor description not available → treated empty)
    """
    try:
        out = dbx_query(sql, params)
    except Exception:
        return pd.DataFrame()

    if out is None:
        return pd.DataFrame()
    if isinstance(out, pd.DataFrame):
        return out
    if isinstance(out, list):
        if len(out) == 0:
            return pd.DataFrame()
        if isinstance(out[0], dict):
            return pd.DataFrame(out)
    return pd.DataFrame()


def _parse_int_or_none(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _fmt_int(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"


def _fmt_pct(x: Any, decimals: int = 1) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x) * 100:.{decimals}f}%"
    except Exception:
        return "—"


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def _compute_wape(sum_abs_error: Any, sum_actual: Any) -> Optional[float]:
    try:
        denom = float(sum_actual)
        if denom == 0:
            return None
        return float(sum_abs_error) / denom
    except Exception:
        return None


def _compute_bias(sum_forecast: Any, sum_actual: Any) -> Optional[float]:
    try:
        denom = float(sum_actual)
        if denom == 0:
            return None
        return (float(sum_forecast) - float(sum_actual)) / denom
    except Exception:
        return None


# -----------------------------
# Cached fetches
# -----------------------------
@st.cache_data(show_spinner=False, ttl=120)
def fetch_available_eval_weeks(backtest_table: str, limit: int = 120) -> List[str]:
    sql = f"""
        SELECT DISTINCT CAST(eval_week_start AS STRING) AS eval_week_start
        FROM {backtest_table}
        ORDER BY eval_week_start DESC
        LIMIT {int(limit)}
    """
    df = _sql_df(sql)
    if df.empty or "eval_week_start" not in df.columns:
        return []
    return [str(x) for x in df["eval_week_start"].dropna().tolist()]


@st.cache_data(show_spinner=False, ttl=120)
def fetch_available_items(backtest_table: str, eval_week_start: str) -> List[str]:
    sql = f"""
        SELECT DISTINCT ITEM
        FROM {backtest_table}
        WHERE eval_week_start = ?
        ORDER BY ITEM
    """
    df = _sql_df(sql, (eval_week_start,))
    if df.empty or "ITEM" not in df.columns:
        return []
    return [str(x) for x in df["ITEM"].dropna().tolist()]


@st.cache_data(show_spinner=False, ttl=120)
def fetch_overall_metrics(backtest_table: str, eval_week_start: str, depot: Optional[int], item: Optional[str]) -> Dict[str, Any]:
    where = "WHERE eval_week_start = ?"
    params: List[Any] = [eval_week_start]

    if depot is not None:
        where += " AND DEPOT = ?"
        params.append(depot)
    if item and item != "All":
        where += " AND ITEM = ?"
        params.append(item)

    # If abs_error / pct_error do not exist, compute them safely.
    sql = f"""
        SELECT
            SUM(COALESCE(abs_error, ABS(yhat - y))) AS sum_abs_error,
            SUM(y) AS sum_actual,
            SUM(yhat) AS sum_forecast,
            AVG(COALESCE(pct_error, (yhat - y) / NULLIF(y, 0))) AS avg_pct_error,
            AVG(ABS(COALESCE(pct_error, (yhat - y) / NULLIF(y, 0)))) AS avg_abs_pct_error
        FROM {backtest_table}
        {where}
    """
    df = _sql_df(sql, tuple(params))
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


@st.cache_data(show_spinner=False, ttl=120)
def fetch_metrics_by_item(backtest_table: str, eval_week_start: str, depot: Optional[int]) -> pd.DataFrame:
    where = "WHERE eval_week_start = ?"
    params: List[Any] = [eval_week_start]
    if depot is not None:
        where += " AND DEPOT = ?"
        params.append(depot)

    sql = f"""
        SELECT
            ITEM,
            SUM(COALESCE(abs_error, ABS(yhat - y))) AS sum_abs_error,
            SUM(y) AS sum_actual,
            SUM(yhat) AS sum_forecast,
            AVG(COALESCE(pct_error, (yhat - y) / NULLIF(y, 0))) AS avg_pct_error,
            AVG(ABS(COALESCE(pct_error, (yhat - y) / NULLIF(y, 0)))) AS avg_abs_pct_error
        FROM {backtest_table}
        {where}
        GROUP BY ITEM
        ORDER BY ITEM
    """
    return _sql_df(sql, tuple(params))


@st.cache_data(show_spinner=False, ttl=120)
def fetch_metrics_by_depot(backtest_table: str, eval_week_start: str, item: Optional[str]) -> pd.DataFrame:
    where = "WHERE eval_week_start = ?"
    params: List[Any] = [eval_week_start]
    if item and item != "All":
        where += " AND ITEM = ?"
        params.append(item)

    sql = f"""
        SELECT
            DEPOT,
            SUM(COALESCE(abs_error, ABS(yhat - y))) AS sum_abs_error,
            SUM(y) AS sum_actual,
            SUM(yhat) AS sum_forecast,
            AVG(COALESCE(pct_error, (yhat - y) / NULLIF(y, 0))) AS avg_pct_error,
            AVG(ABS(COALESCE(pct_error, (yhat - y) / NULLIF(y, 0)))) AS avg_abs_pct_error
        FROM {backtest_table}
        {where}
        GROUP BY DEPOT
        ORDER BY (SUM(COALESCE(abs_error, ABS(yhat - y))) / NULLIF(SUM(y), 0)) DESC
    """
    return _sql_df(sql, tuple(params))


@st.cache_data(show_spinner=False, ttl=120)
def fetch_worst_offenders(
    backtest_table: str,
    eval_week_start: str,
    depot: Optional[int],
    item: Optional[str],
    top_n: int,
) -> pd.DataFrame:
    where = "WHERE eval_week_start = ?"
    params: List[Any] = [eval_week_start]
    if depot is not None:
        where += " AND DEPOT = ?"
        params.append(depot)
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
            COALESCE(abs_error, ABS(yhat - y)) AS abs_error,
            COALESCE(pct_error, (yhat - y) / NULLIF(y, 0)) AS pct_error
        FROM {backtest_table}
        {where}
        ORDER BY ABS(COALESCE(pct_error, (yhat - y) / NULLIF(y, 0))) DESC
        LIMIT {int(top_n)}
    """
    df = _sql_df(sql, tuple(params))
    if df.empty:
        return df
    for c in ["actual_units", "forecast_units", "abs_error", "pct_error"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False, ttl=120)
def fetch_trend_over_weeks(backtest_table: str, item: Optional[str], depot: Optional[int], weeks_back: int) -> pd.DataFrame:
    where = "WHERE 1=1"
    params: List[Any] = []

    if depot is not None:
        where += " AND DEPOT = ?"
        params.append(depot)
    if item and item != "All":
        where += " AND ITEM = ?"
        params.append(item)

    sql = f"""
        WITH base AS (
            SELECT
                eval_week_start,
                SUM(COALESCE(abs_error, ABS(yhat - y))) AS sum_abs_error,
                SUM(y) AS sum_actual,
                SUM(yhat) AS sum_forecast
            FROM {backtest_table}
            {where}
            GROUP BY eval_week_start
        ),
        limited AS (
            SELECT *
            FROM base
            ORDER BY eval_week_start DESC
            LIMIT {int(weeks_back)}
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
    df = _sql_df(sql, tuple(params))
    if df.empty:
        return df
    for c in ["sum_abs_error", "sum_actual", "sum_forecast", "wape", "bias"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Model Performance", page_icon="✅", layout="wide")
st.header("✅ Model Performance")
st.caption("Deterministic performance metrics for the evaluated week (buyer typically orders one week out).")

tables = _get_tables()
BACKTEST_TABLE = getattr(tables, "backtest", "ppe_dscoe.default.placeholder_backtest_outputtable")

# Offline-safe shell
if not dbx_is_configured():
    st.warning(
        "Databricks SQL is not configured. This page will load once DBX_SQL_HOST / DBX_SQL_HTTP_PATH / DBX_SQL_TOKEN are set."
    )
    st.info("You can still navigate the app UI. Metrics will populate when the connection is enabled.")
    st.stop()

# Controls
eval_weeks = fetch_available_eval_weeks(BACKTEST_TABLE, limit=120)
if not eval_weeks:
    st.error(f"No evaluation weeks found in `{BACKTEST_TABLE}`.")
    st.stop()

# Remember common filters
default_depot_txt = str(st.session_state.get("selected_depot") or "")
default_item = st.session_state.get("selected_perf_item") or "All"

c1, c2, c3, c4 = st.columns([2.0, 1.6, 2.2, 1.8], vertical_alignment="bottom")

with c1:
    eval_week = st.selectbox(
        "Evaluation week (actuals scored)",
        options=eval_weeks,
        index=0,
        help="Week start date for the actual demand being evaluated.",
    )

with c2:
    depot_txt = st.text_input(
        "Depot (optional)",
        value=default_depot_txt,
        placeholder="e.g., 172",
        help="Leave blank to view all depots.",
    )
    depot = _parse_int_or_none(depot_txt)
    st.session_state["selected_depot"] = depot_txt.strip() or None

with c3:
    # Pull items from the table for the selected week; fall back to defaults if empty
    available_items = fetch_available_items(BACKTEST_TABLE, eval_week)
    item_options = ["All"] + (available_items if available_items else ["Combined", "Blueberries", "OrganicBlueberries"])
    if default_item not in item_options:
        default_item = "All"

    item = st.selectbox(
        "Item",
        options=item_options,
        index=item_options.index(default_item),
        help="Filters breakdowns and worst offenders.",
    )
    st.session_state["selected_perf_item"] = item

with c4:
    worst_n = st.slider("Worst offenders (top N)", min_value=5, max_value=50, value=15, step=5)

st.divider()

# Overall metrics (deterministic)
with st.spinner("Computing overall metrics…"):
    overall = fetch_overall_metrics(BACKTEST_TABLE, eval_week, depot, item)

sum_abs_error = overall.get("sum_abs_error")
sum_actual = overall.get("sum_actual")
sum_forecast = overall.get("sum_forecast")
avg_pct_error = overall.get("avg_pct_error")
avg_abs_pct_error = overall.get("avg_abs_pct_error")

wape = _compute_wape(sum_abs_error, sum_actual)
bias = _compute_bias(sum_forecast, sum_actual)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Actual units (sum)", _fmt_int(sum_actual))
k2.metric("Forecast units (sum)", _fmt_int(sum_forecast))
k3.metric("WAPE", _fmt_pct(wape, 1))
k4.metric("Bias", _fmt_pct(bias, 1))
k5.metric("Avg |% error|", _fmt_pct(avg_abs_pct_error, 1))

st.caption(
    "Definitions: **WAPE** = sum(abs error) / sum(actual). "
    "**Bias** = (sum(forecast) − sum(actual)) / sum(actual). "
    "All metrics are computed deterministically from the backtest table."
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📌 Breakdowns", "🚨 Worst Offenders", "📈 Trend"])

# -----------------------------
# Tab 1: Breakdowns
# -----------------------------
with tab1:
    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.subheader("Breakdown by item")
        st.caption("Which item(s) drove error for this evaluated week?")

        by_item = fetch_metrics_by_item(BACKTEST_TABLE, eval_week, depot)
        if by_item.empty:
            st.info("No rows for this evaluation week / depot filter.")
        else:
            # compute wape/bias
            by_item["wape"] = by_item["sum_abs_error"] / by_item["sum_actual"].replace({0: pd.NA})
            by_item["bias"] = (by_item["sum_forecast"] - by_item["sum_actual"]) / by_item["sum_actual"].replace({0: pd.NA})

            disp = by_item[["ITEM", "sum_actual", "sum_forecast", "wape", "bias", "avg_abs_pct_error"]].copy()
            disp = disp.rename(
                columns={
                    "ITEM": "Item",
                    "sum_actual": "Actual (sum)",
                    "sum_forecast": "Forecast (sum)",
                    "wape": "WAPE",
                    "bias": "Bias",
                    "avg_abs_pct_error": "Avg |% error|",
                }
            )

            # Table
            disp_tbl = disp.copy()
            disp_tbl["Actual (sum)"] = disp_tbl["Actual (sum)"].apply(_fmt_int)
            disp_tbl["Forecast (sum)"] = disp_tbl["Forecast (sum)"].apply(_fmt_int)
            disp_tbl["WAPE"] = disp_tbl["WAPE"].apply(lambda x: _fmt_pct(x, 1))
            disp_tbl["Bias"] = disp_tbl["Bias"].apply(lambda x: _fmt_pct(x, 1))
            disp_tbl["Avg |% error|"] = disp_tbl["Avg |% error|"].apply(lambda x: _fmt_pct(x, 1))
            st.dataframe(disp_tbl, use_container_width=True, hide_index=True)

            # Chart (WAPE by item)
            chart_df = disp.dropna(subset=["WAPE"]).copy()
            if not chart_df.empty:
                bar = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("WAPE:Q", title="WAPE"),
                        y=alt.Y("Item:N", sort="-x", title=None),
                        tooltip=["Item:N", alt.Tooltip("WAPE:Q", format=".3f"), alt.Tooltip("Bias:Q", format=".3f")],
                    )
                    .properties(height=240)
                )
                st.altair_chart(bar, use_container_width=True)

    with right:
        st.subheader("Breakdown by depot")
        st.caption("Sorted by WAPE (highest first).")

        by_depot = fetch_metrics_by_depot(BACKTEST_TABLE, eval_week, item)
        if by_depot.empty:
            st.info("No rows for this evaluation week / item filter.")
        else:
            by_depot["wape"] = by_depot["sum_abs_error"] / by_depot["sum_actual"].replace({0: pd.NA})
            by_depot["bias"] = (by_depot["sum_forecast"] - by_depot["sum_actual"]) / by_depot["sum_actual"].replace({0: pd.NA})

            disp = by_depot[["DEPOT", "sum_actual", "sum_forecast", "wape", "bias", "avg_abs_pct_error"]].copy()
            disp = disp.rename(
                columns={
                    "DEPOT": "Depot",
                    "sum_actual": "Actual (sum)",
                    "sum_forecast": "Forecast (sum)",
                    "wape": "WAPE",
                    "bias": "Bias",
                    "avg_abs_pct_error": "Avg |% error|",
                }
            )

            # Chart (top depots by WAPE)
            top_n = st.slider("Top depots by WAPE", min_value=5, max_value=15, value=10, step=1)
            chart_df = disp.dropna(subset=["WAPE"]).head(top_n).copy()
            if not chart_df.empty:
                bar = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("WAPE:Q", title="WAPE"),
                        y=alt.Y("Depot:N", sort="-x", title=None),
                        tooltip=["Depot:N", alt.Tooltip("WAPE:Q", format=".3f"), alt.Tooltip("Bias:Q", format=".3f")],
                    )
                    .properties(height=280)
                )
                st.altair_chart(bar, use_container_width=True)

            # Table
            disp_tbl = disp.copy()
            disp_tbl["Actual (sum)"] = disp_tbl["Actual (sum)"].apply(_fmt_int)
            disp_tbl["Forecast (sum)"] = disp_tbl["Forecast (sum)"].apply(_fmt_int)
            disp_tbl["WAPE"] = disp_tbl["WAPE"].apply(lambda x: _fmt_pct(x, 1))
            disp_tbl["Bias"] = disp_tbl["Bias"].apply(lambda x: _fmt_pct(x, 1))
            disp_tbl["Avg |% error|"] = disp_tbl["Avg |% error|"].apply(lambda x: _fmt_pct(x, 1))
            st.dataframe(disp_tbl, use_container_width=True, hide_index=True, height=420)

# -----------------------------
# Tab 2: Worst offenders
# -----------------------------
with tab2:
    st.subheader("Worst offenders")
    st.caption("Largest absolute percent errors for the evaluated week (quick triage).")

    worst = fetch_worst_offenders(BACKTEST_TABLE, eval_week, depot, item, int(worst_n))
    if worst.empty:
        st.info("No rows found for the selected filters.")
    else:
        st.dataframe(
            worst,
            use_container_width=True,
            hide_index=True,
            column_config={
                "DEPOT": st.column_config.NumberColumn("DEPOT", format="%d"),
                "ITEM": st.column_config.TextColumn("ITEM"),
                "eval_week_start": st.column_config.TextColumn("Eval week"),
                "actual_units": st.column_config.NumberColumn("Actual units", format="%,.0f"),
                "forecast_units": st.column_config.NumberColumn("Forecast units", format="%,.0f"),
                "abs_error": st.column_config.NumberColumn("Absolute error", format="%,.0f"),
                "pct_error": st.column_config.NumberColumn("% error (fraction)", format="%.4f"),
            },
        )

        csv = worst.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download worst offenders CSV",
            data=csv,
            file_name=f"worst_offenders_eval_{eval_week}.csv",
            mime="text/csv",
        )

# -----------------------------
# Tab 3: Trend
# -----------------------------
with tab3:
    st.subheader("Performance trend")
    st.caption("WAPE and Bias over recent evaluation weeks (deterministic aggregates).")

    t1, t2, t3 = st.columns([2.0, 1.6, 1.6], vertical_alignment="bottom")
    with t1:
        trend_item = st.selectbox(
            "Trend item",
            options=["All"] + (available_items if available_items else ["Combined", "Blueberries", "OrganicBlueberries"]),
            index=0,
        )
    with t2:
        trend_depot_txt = st.text_input(
            "Trend depot (optional)",
            value="" if depot is None else str(depot),
            placeholder="e.g., 172",
        )
        trend_depot = _parse_int_or_none(trend_depot_txt)
    with t3:
        weeks_back = st.slider("Weeks back", min_value=8, max_value=104, value=26, step=2)

    trend_df = fetch_trend_over_weeks(BACKTEST_TABLE, trend_item, trend_depot, int(weeks_back))
    if trend_df.empty:
        st.info("No trend data found for the selected filters.")
    else:
        trend_df["eval_week_start"] = pd.to_datetime(trend_df["eval_week_start"], errors="coerce")
        trend_df = trend_df.sort_values("eval_week_start")

        # Melt for nicer Altair lines
        plot = trend_df[["eval_week_start", "wape", "bias"]].melt(
            id_vars=["eval_week_start"], var_name="metric", value_name="value"
        )

        line = (
            alt.Chart(plot)
            .mark_line(point=True)
            .encode(
                x=alt.X("eval_week_start:T", title="Evaluation week"),
                y=alt.Y("value:Q", title="Metric (fraction)"),
                color=alt.Color("metric:N", title=None),
                tooltip=[
                    alt.Tooltip("eval_week_start:T", title="Week"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value", format=".4f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(line, use_container_width=True)

        with st.expander("Trend table"):
            disp = trend_df.copy()
            disp["Actual (sum)"] = disp["sum_actual"].apply(_fmt_int)
            disp["Forecast (sum)"] = disp["sum_forecast"].apply(_fmt_int)
            disp["WAPE"] = disp["wape"].apply(lambda x: _fmt_pct(x, 1))
            disp["Bias"] = disp["bias"].apply(lambda x: _fmt_pct(x, 1))

            st.dataframe(
                disp[["eval_week_start", "Actual (sum)", "Forecast (sum)", "WAPE", "Bias"]],
                use_container_width=True,
                hide_index=True,
            )

st.caption(
    f"Data source: `{BACKTEST_TABLE}` (Delta). All metrics are computed deterministically from stored backtest outputs."
)
