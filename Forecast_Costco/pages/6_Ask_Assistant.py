# pages/6_Ask_Assistant.py
# ------------------------------------------------------------
# Ask Assistant — Buyer-friendly, production-leaning POC
#
# Goals:
# - Chat UI for buyers to ask questions about forecasts, model performance, D&D
# - Deterministic SQL + deterministic Python math (no “LLM math”)
# - Guardrails: only whitelisted tools; no free-form SQL execution
# - Offline-safe: UI works without Databricks configured (answers explain why empty)
#
# Data sources (from TableRegistry in st.session_state["tables"]):
#   - forecast:   ppe_dscoe.default.placeholder_forecast_ouputtable
#   - backtest:   ppe_dscoe.default.placeholder_backtest_outputtable
#   - actuals:    ppe_dscoe.default.ex_sales_table_historical
#   - dnd tables: ppe_dscoe.default.company_level_blueberries_dnd_monitoring
#                ppe_dscoe.default.depot_level_current_week_blueberries_dnd_monitoring
#                ppe_dscoe.default.depot_level_historical_dnd_monitoring_data
#
# Depends on:
#   - services.dbx_sql (query, is_configured)
# ------------------------------------------------------------

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from services.dbx_sql import is_configured as dbx_is_configured, query as dbx_query
import streamlit as st

tables = st.session_state.get("tables")
qs = st.session_state.get("query_service")

if tables is None or qs is None:
    st.error("App is not initialized. Open the Home page (app.py) first, then return here.")
    st.stop()

# -----------------------------
# Settings
# -----------------------------
MAX_ROWS_DEFAULT = 5000  # hard cap for safety
SHOW_SQL_DEFAULT = True
SHOW_DATA_DEFAULT = True

# Optional LLM router (OFF by default)
USE_LLM_ROUTER = os.getenv("USE_LLM_ROUTER", "false").strip().lower() in {"1", "true", "yes", "y"}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()  # optional
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"


# -----------------------------
# Helpers
# -----------------------------
def _now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_tables():
    if "tables" not in st.session_state:
        st.error("Tables registry not initialized. Please reload the app.")
        st.stop()
    return st.session_state["tables"]


def _sql_df(sql: str, params: Optional[Tuple[Any, ...]] = None) -> pd.DataFrame:
    """
    Runs a SQL query via services.dbx_sql.query and returns a DataFrame.
    Offline-safe: returns empty DataFrame on any error.
    """
    try:
        out = dbx_query(sql, params)
    except Exception:
        return pd.DataFrame()

    if out is None:
        return pd.DataFrame()
    if isinstance(out, pd.DataFrame):
        return out
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return pd.DataFrame(out)
    return pd.DataFrame()


def _fmt_int(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"


def _fmt_money(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"${float(x):,.0f}"
    except Exception:
        return "—"


def _fmt_rate(x: Any, decimals: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x) * 100:.{decimals}f}%"
    except Exception:
        return "—"


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


def _parse_iso_date_or_none(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        dt.date.fromisoformat(s)
        return s
    except Exception:
        return None


def _extract_date(text: str) -> Optional[str]:
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if not m:
        return None
    return _parse_iso_date_or_none(m.group(1))


def _extract_depot(text: str) -> Optional[int]:
    # e.g. "depot 172" or "DEPOT:172"
    m = re.search(r"\bdepot\D*(\d{2,6})\b", text.lower())
    if not m:
        # fallback: any standalone 2-6 digit number
        m2 = re.search(r"\b(\d{2,6})\b", text)
        if not m2:
            return None
        try:
            return int(m2.group(1))
        except Exception:
            return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_week_horizon(text: str) -> Optional[int]:
    t = text.lower()
    m = re.search(r"\bweek\s*([1-3])\b", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\bhorizon\s*([1-3])\b", t)
    if m:
        return int(m.group(1))
    return None


def _contains_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms)


# -----------------------------
# Tool Results
# -----------------------------
@dataclass
class ToolResult:
    title: str
    summary_md: str
    df: Optional[pd.DataFrame] = None
    sql: Optional[str] = None
    params: Optional[Tuple[Any, ...]] = None


def _offline_result(title: str) -> ToolResult:
    return ToolResult(
        title=title,
        summary_md=(
            "Databricks SQL is not configured, so I can’t query live tables yet.\n\n"
            "To enable answers:\n"
            "- Set `DBX_SQL_HOST`, `DBX_SQL_HTTP_PATH`, `DBX_SQL_TOKEN`\n"
        ),
        df=pd.DataFrame(),
    )


# -----------------------------
# Deterministic Tools (whitelisted)
# -----------------------------
def tool_forecast_total(run_id: str, item: str, depot: Optional[int], horizon: Optional[int]) -> ToolResult:
    tables = _get_tables()
    forecast_table = getattr(tables, "forecast", "ppe_dscoe.default.placeholder_forecast_ouputtable")

    if not dbx_is_configured():
        return _offline_result("Forecast totals")

    if not run_id:
        return ToolResult(
            title="Forecast totals",
            summary_md="I need a **run_id** (run date) to compute totals. Set a default in the sidebar or include a date like `2025-12-08`.",
            df=None,
        )

    where = "WHERE CAST(run_id AS STRING) = ?"
    params: List[Any] = [run_id]

    if item and item != "All":
        where += " AND ITEM = ?"
        params.append(item)
    if depot is not None:
        where += " AND DEPOT = ?"
        params.append(int(depot))
    if horizon is not None:
        where += " AND horizon = ?"
        params.append(int(horizon))

    sql = f"""
        SELECT
            CAST(run_id AS STRING) AS run_id,
            ITEM,
            horizon,
            SUM(yhat) AS total_forecast_units
        FROM {forecast_table}
        {where}
        GROUP BY CAST(run_id AS STRING), ITEM, horizon
        ORDER BY ITEM, horizon
    """
    df = _sql_df(sql, tuple(params))
    if df.empty:
        return ToolResult(
            title="Forecast totals",
            summary_md="No forecast rows found for the selected filters.",
            df=df,
            sql=sql,
            params=tuple(params),
        )

    df["total_forecast_units"] = pd.to_numeric(df["total_forecast_units"], errors="coerce")
    total = float(df["total_forecast_units"].sum())

    scope = [f"**Run date:** `{run_id}`"]
    if item and item != "All":
        scope.append(f"**Item:** `{item}`")
    if depot is not None:
        scope.append(f"**Depot:** `{depot}`")
    if horizon is not None:
        scope.append(f"**Horizon:** `{horizon}`")
    md = " • ".join(scope) + f"\n\n**Total forecast units:** **{_fmt_int(total)}**"

    return ToolResult(title="Forecast totals", summary_md=md, df=df, sql=sql, params=tuple(params))


def tool_forecast_rows(run_id: str, item: str, depot: Optional[int], horizon: Optional[int], limit: int) -> ToolResult:
    tables = _get_tables()
    forecast_table = getattr(tables, "forecast", "ppe_dscoe.default.placeholder_forecast_ouputtable")

    if not dbx_is_configured():
        return _offline_result("Forecast rows")

    if not run_id:
        return ToolResult(
            title="Forecast rows",
            summary_md="I need a **run_id** (run date) to show rows. Set a default in the sidebar or include a date like `2025-12-08`.",
            df=None,
        )

    limit = int(min(max(int(limit), 1), MAX_ROWS_DEFAULT))

    where = "WHERE CAST(run_id AS STRING) = ?"
    params: List[Any] = [run_id]

    if item and item != "All":
        where += " AND ITEM = ?"
        params.append(item)
    if depot is not None:
        where += " AND DEPOT = ?"
        params.append(int(depot))
    if horizon is not None:
        where += " AND horizon = ?"
        params.append(int(horizon))

    # NOTE: Your forecast output example is (DEPOT, ITEM, ds, yhat).
    # If your table uses ds as the week start, we expose it as target_week_start here.
    sql = f"""
        SELECT
            DEPOT,
            ITEM,
            CAST(ds AS STRING) AS target_week_start,
            horizon,
            yhat
        FROM {forecast_table}
        {where}
        ORDER BY DEPOT, ITEM, target_week_start, horizon
        LIMIT {limit}
    """
    df = _sql_df(sql, tuple(params))
    if not df.empty and "yhat" in df.columns:
        df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")

    md = f"Showing up to **{limit:,}** rows for run `{run_id}`."
    return ToolResult(title="Forecast rows", summary_md=md, df=df, sql=sql, params=tuple(params))


def tool_backtest_summary(eval_week_start: str, item: str, depot: Optional[int]) -> ToolResult:
    tables = _get_tables()
    backtest_table = getattr(tables, "backtest", "ppe_dscoe.default.placeholder_backtest_outputtable")

    if not dbx_is_configured():
        return _offline_result("Model performance summary")

    if not eval_week_start:
        return ToolResult(
            title="Model performance summary",
            summary_md="I need an **eval_week_start** (e.g., `2025-12-01`) to summarize performance.",
            df=None,
        )

    where = "WHERE CAST(eval_week_start AS STRING) = ?"
    params: List[Any] = [eval_week_start]

    if item and item != "All":
        where += " AND ITEM = ?"
        params.append(item)
    if depot is not None:
        where += " AND DEPOT = ?"
        params.append(int(depot))

    sql = f"""
        SELECT
            SUM(AbsoluteError) AS sum_abs_error,
            SUM(y) AS sum_actual,
            SUM(yhat) AS sum_forecast,
            AVG(Percent_Error) AS avg_pct_error,
            AVG(ABS(Percent_Error)) AS avg_abs_pct_error
        FROM {backtest_table}
        {where}
    """
    df = _sql_df(sql, tuple(params))
    if df.empty or df.iloc[0].isna().all():
        return ToolResult(
            title="Model performance summary",
            summary_md="No backtest rows found for the selected filters.",
            df=None,
            sql=sql,
            params=tuple(params),
        )

    r = df.iloc[0].to_dict()
    sum_abs = float(r.get("sum_abs_error") or 0.0)
    sum_act = float(r.get("sum_actual") or 0.0)
    sum_fc = float(r.get("sum_forecast") or 0.0)

    wape = (sum_abs / sum_act) if sum_act else None
    bias = ((sum_fc - sum_act) / sum_act) if sum_act else None

    scope = [f"**Eval week:** `{eval_week_start}`"]
    if item and item != "All":
        scope.append(f"**Item:** `{item}`")
    if depot is not None:
        scope.append(f"**Depot:** `{depot}`")

    md = (
        " • ".join(scope)
        + "\n\n"
        + f"- **Actual units (sum):** {_fmt_int(sum_act)}\n"
        + f"- **Forecast units (sum):** {_fmt_int(sum_fc)}\n"
        + f"- **WAPE:** {_fmt_rate(wape, 1)}\n"
        + f"- **Bias:** {_fmt_rate(bias, 1)}\n"
        + f"- **Avg |% error|:** {_fmt_rate(r.get('avg_abs_pct_error'), 1)}\n"
    )

    return ToolResult(title="Model performance summary", summary_md=md, df=df, sql=sql, params=tuple(params))


def tool_dnd_company_latest(weeks_back: int = 52) -> ToolResult:
    tables = _get_tables()
    dnd_company_table = getattr(tables, "dnd_company", "ppe_dscoe.default.company_level_blueberries_dnd_monitoring")

    if not dbx_is_configured():
        return _offline_result("Company D&D (latest)")

    weeks_back = int(min(max(int(weeks_back), 1), 260))

    # Pull recent rows; compute rate deterministically
    sql = f"""
        SELECT
            CAST(FSCL_WK_START_DATE AS STRING) AS FSCL_WK_START_DATE,
            Total_DND_QTY,
            Total_Units,
            Total_DND_$_loss
        FROM {dnd_company_table}
        ORDER BY FSCL_WK_START_DATE DESC
        LIMIT {weeks_back}
    """
    df = _sql_df(sql)
    if df.empty:
        return ToolResult(title="Company D&D (latest)", summary_md="No company D&D rows found.", df=df, sql=sql)

    for c in ["Total_DND_QTY", "Total_Units", "Total_DND_$_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dnd_rate"] = df["Total_DND_QTY"] / df["Total_Units"].replace({0: pd.NA})

    latest = df.iloc[0]
    md = (
        f"**Latest company D&D week:** `{latest.get('FSCL_WK_START_DATE')}`\n\n"
        f"- **D&D rate:** **{_fmt_rate(latest.get('dnd_rate'), 2)}**\n"
        f"- **D&D $ loss:** **{_fmt_money(latest.get('Total_DND_$_loss'))}**\n"
        f"- **DND qty:** {_fmt_int(latest.get('Total_DND_QTY'))}\n"
        f"- **Total units:** {_fmt_int(latest.get('Total_Units'))}\n"
    )
    return ToolResult(title="Company D&D (latest)", summary_md=md, df=df, sql=sql)


def tool_dnd_depot_snapshot(week_start: Optional[str], rank_by: str = "rate", top_n: int = 10) -> ToolResult:
    tables = _get_tables()
    dnd_depot_current = getattr(tables, "dnd_depot_current", "ppe_dscoe.default.depot_level_current_week_blueberries_dnd_monitoring")

    if not dbx_is_configured():
        return _offline_result("Depot D&D snapshot")

    top_n = int(min(max(int(top_n), 1), 50))

    params: List[Any] = []
    where = ""
    if week_start:
        where = "WHERE CAST(FSCL_WK_START_DATE AS STRING) = ?"
        params.append(week_start)

    sql = f"""
        SELECT
            DEPOT,
            CAST(FSCL_WK_START_DATE AS STRING) AS FSCL_WK_START_DATE,
            Volume,
            Total_DND_QTY,
            Total_Units,
            Total_DND_$_loss
        FROM {dnd_depot_current}
        {where}
    """
    df = _sql_df(sql, tuple(params) if params else None)
    if df.empty:
        return ToolResult(title="Depot D&D snapshot", summary_md="No depot snapshot rows found.", df=df, sql=sql, params=tuple(params) if params else None)

    for c in ["Volume", "Total_DND_QTY", "Total_Units", "Total_DND_$_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dnd_rate"] = df["Total_DND_QTY"] / df["Total_Units"].replace({0: pd.NA})

    if rank_by == "loss":
        df = df.sort_values("Total_DND_$_loss", ascending=False)
        label = "D&D $ loss"
    elif rank_by == "qty":
        df = df.sort_values("Total_DND_QTY", ascending=False)
        label = "DND qty"
    else:
        df = df.sort_values("dnd_rate", ascending=False)
        label = "D&D rate"

    out = df.head(top_n).copy()
    md = f"Top **{top_n}** depots by **{label}**" + (f" for week `{week_start}`." if week_start else ".")
    return ToolResult(title="Depot D&D snapshot", summary_md=md, df=out, sql=sql, params=tuple(params) if params else None)


# -----------------------------
# Routing (heuristic + optional LLM)
# -----------------------------
TOOLS: Dict[str, Callable[..., ToolResult]] = {
    "forecast_total": tool_forecast_total,
    "forecast_rows": tool_forecast_rows,
    "backtest_summary": tool_backtest_summary,
    "dnd_company_latest": tool_dnd_company_latest,
    "dnd_depot_snapshot": tool_dnd_depot_snapshot,
}


def heuristic_route(user_text: str, ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    t = user_text.strip().lower()
    date_in_text = _extract_date(user_text)
    depot_in_text = _extract_depot(user_text)
    horizon_in_text = _extract_week_horizon(user_text)

    # D&D
    if _contains_any(t, ["d&d", "dnd", "dump", "destroy", "d and d"]):
        if _contains_any(t, ["top", "worst", "rank", "highest"]):
            if _contains_any(t, ["loss", "$"]):
                return "dnd_depot_snapshot", {"week_start": date_in_text or ctx.get("dnd_week") or None, "rank_by": "loss", "top_n": 10}
            if _contains_any(t, ["qty", "quantity", "units"]):
                return "dnd_depot_snapshot", {"week_start": date_in_text or ctx.get("dnd_week") or None, "rank_by": "qty", "top_n": 10}
            return "dnd_depot_snapshot", {"week_start": date_in_text or ctx.get("dnd_week") or None, "rank_by": "rate", "top_n": 10}
        return "dnd_company_latest", {"weeks_back": 52}

    # Performance / backtest
    if _contains_any(t, ["backtest", "performance", "error", "wape", "mape", "bias", "accuracy"]):
        return "backtest_summary", {
            "eval_week_start": date_in_text or ctx.get("eval_week") or "",
            "item": ctx.get("item", "All"),
            "depot": depot_in_text if depot_in_text is not None else ctx.get("depot"),
        }

    # Forecasts
    if _contains_any(t, ["forecast", "yhat", "predict", "prediction", "demand"]):
        if _contains_any(t, ["show", "list", "table", "rows", "detail"]):
            return "forecast_rows", {
                "run_id": date_in_text or ctx.get("run_id") or "",
                "item": ctx.get("item", "All"),
                "depot": depot_in_text if depot_in_text is not None else ctx.get("depot"),
                "horizon": horizon_in_text,
                "limit": 2000,
            }
        return "forecast_total", {
            "run_id": date_in_text or ctx.get("run_id") or "",
            "item": ctx.get("item", "All"),
            "depot": depot_in_text if depot_in_text is not None else ctx.get("depot"),
            "horizon": horizon_in_text,
        }

    # Default safe action
    return "forecast_total", {"run_id": ctx.get("run_id") or "", "item": ctx.get("item", "Combined"), "depot": ctx.get("depot"), "horizon": horizon_in_text}


def maybe_llm_route(user_text: str, ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Optional LLM routing: returns (tool, args, raw).
    IMPORTANT: tool selection only; execution is still whitelisted tools.
    """
    if not (USE_LLM_ROUTER and OPENAI_API_KEY):
        return "", {}, None

    try:
        import requests  # type: ignore
    except Exception:
        return "", {}, None

    system = (
        "You are a router for a forecasting analytics app.\n"
        "Choose exactly one tool and args as STRICT JSON only.\n\n"
        "TOOLS:\n"
        "1) forecast_total(run_id, item, depot, horizon)\n"
        "2) forecast_rows(run_id, item, depot, horizon, limit)\n"
        "3) backtest_summary(eval_week_start, item, depot)\n"
        "4) dnd_company_latest(weeks_back)\n"
        "5) dnd_depot_snapshot(week_start, rank_by['rate'|'loss'|'qty'], top_n)\n\n"
        "RULES:\n"
        "- Output ONLY JSON: {\"tool\":..., \"args\":{...}}\n"
        "- Do not invent dates; use ctx defaults if absent.\n"
        "- Enforce: limit<=5000, top_n<=50.\n"
    )

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"question": user_text, "ctx": ctx})},
        ],
        "temperature": 0,
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        if resp.status_code >= 300:
            return "", {}, None
        data = resp.json()

        raw = ""
        for o in data.get("output", []):
            for c in o.get("content", []):
                if c.get("type") == "output_text":
                    raw += c.get("text", "")
        raw = raw.strip()
        if not raw:
            return "", {}, None

        j = json.loads(raw)
        tool = j.get("tool", "")
        args = j.get("args", {}) or {}
        return tool, args, raw
    except Exception:
        return "", {}, None


def execute_tool(tool: str, args: Dict[str, Any]) -> ToolResult:
    if tool not in TOOLS:
        return ToolResult(
            title="Unsupported request",
            summary_md="I couldn’t map that request to a supported analytics action yet. Try asking about forecasts, performance, or D&D.",
        )

    # Safety caps
    if tool == "forecast_rows":
        args["limit"] = int(min(max(int(args.get("limit", 2000)), 1), MAX_ROWS_DEFAULT))
    if tool == "dnd_depot_snapshot":
        args["top_n"] = int(min(max(int(args.get("top_n", 10)), 1), 50))

    return TOOLS[tool](**args)


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Ask Assistant", page_icon="💬", layout="wide")

st.title("💬 Ask Assistant")
st.caption("Ask questions. I answer using deterministic SQL + math (and can show the SQL + data used).")

# Sidebar: controls + defaults
with st.sidebar:
    st.markdown("### Assistant settings")
    show_sql = st.checkbox("Show executed SQL", value=SHOW_SQL_DEFAULT)
    show_data = st.checkbox("Show data used", value=SHOW_DATA_DEFAULT)

    st.divider()
    st.markdown("### Defaults used for routing")

    # Persisted defaults from other pages, if present
    default_run_id = _parse_iso_date_or_none(st.session_state.get("selected_run_id")) or ""
    default_item = st.session_state.get("selected_item", "Combined")
    default_depot = _parse_int_or_none(st.session_state.get("selected_depot"))

    run_id = st.text_input(
        "Default run_id (forecast)",
        value=default_run_id,
        help="Used when you ask forecast questions without specifying a run date (YYYY-MM-DD).",
    ).strip()

    item = st.selectbox(
        "Default item",
        options=["Combined", "Blueberries", "OrganicBlueberries", "All"],
        index=["Combined", "Blueberries", "OrganicBlueberries", "All"].index(default_item)
        if default_item in ["Combined", "Blueberries", "OrganicBlueberries", "All"]
        else 0,
    )

    depot_txt = st.text_input(
        "Default depot (optional)",
        value="" if default_depot is None else str(default_depot),
        placeholder="e.g., 172",
    ).strip()
    depot = _parse_int_or_none(depot_txt)

    eval_week_txt = st.text_input("Default eval_week_start (optional)", value="", placeholder="YYYY-MM-DD").strip()
    eval_week = _parse_iso_date_or_none(eval_week_txt) or ""

    dnd_week_txt = st.text_input("Default D&D week (optional)", value="", placeholder="YYYY-MM-DD").strip()
    dnd_week = _parse_iso_date_or_none(dnd_week_txt) or ""

    # Write back to session for cross-page consistency
    if run_id:
        st.session_state["selected_run_id"] = run_id
    st.session_state["selected_item"] = item
    st.session_state["selected_depot"] = None if depot is None else str(depot)

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state["chat_messages"] = []
        st.rerun()

# Chat state
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# Quick prompts
st.markdown("#### Quick questions")
qcols = st.columns(4)
suggestions = [
    "What are the total forecast units for week 1 for Combined?",
    "Show me the forecast rows for week 1 for depot 172.",
    "How did the model perform for eval week 2025-12-01?",
    "Show the top 10 depots by D&D rate for the latest week.",
]
for i, s in enumerate(suggestions):
    with qcols[i]:
        if st.button(s, use_container_width=True):
            st.session_state["pending_user_input"] = s
            st.rerun()

st.divider()

# Render prior messages
for m in st.session_state["chat_messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        if show_sql and m.get("sql"):
            with st.expander("SQL used"):
                st.code(m["sql"], language="sql")
                if m.get("params") is not None:
                    st.caption(f"params: {m['params']}")

        if show_data and isinstance(m.get("df_json"), str):
            try:
                df = pd.read_json(m["df_json"], orient="split")
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception:
                pass

# Input
user_input = st.session_state.pop("pending_user_input", None)
user_input = user_input or st.chat_input("Ask about forecasts, performance, D&D…")

if user_input:
    # Save user message
    st.session_state["chat_messages"].append({"role": "user", "content": user_input, "ts": _now_str()})

    # Routing context
    ctx = {
        "run_id": run_id,
        "item": item,
        "depot": depot,
        "eval_week": eval_week,
        "dnd_week": dnd_week,
    }

    tool, args, raw_llm = ("", {}, None)
    if USE_LLM_ROUTER and OPENAI_API_KEY:
        tool, args, raw_llm = maybe_llm_route(user_input, ctx)

    if not tool:
        tool, args = heuristic_route(user_input, ctx)

    # Execute tool
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = execute_tool(tool, args)

        header = f"**{result.title}**"
        body = result.summary_md

        # Clear, explicit “added” suggestions (extra buyer value)
        added: List[str] = []
        if tool == "forecast_total":
            added.append("**Added:** You can ask “show forecast rows” to view depot-level detail.")
        if tool == "forecast_rows":
            added.append("**Added:** You can ask “total forecast units for week 1” to get KPI-style totals.")
        if tool == "backtest_summary":
            added.append("**Added:** Use the Model Performance page for worst-offenders and trends.")
        if tool.startswith("dnd_"):
            added.append("**Added:** Ask “top depots by D&D $ loss” to prioritize financial impact.")

        if added:
            body += "\n\n" + "\n".join(added)

        st.markdown(f"{header}\n\n{body}")

        if show_sql and result.sql:
            with st.expander("SQL used"):
                st.code(result.sql, language="sql")
                if result.params is not None:
                    st.caption(f"params: {result.params}")

        df_json = None
        if show_data and isinstance(result.df, pd.DataFrame):
            st.dataframe(result.df, use_container_width=True, hide_index=True)
            try:
                df_json = result.df.to_json(orient="split")
            except Exception:
                df_json = None

        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": f"{header}\n\n{body}",
                "sql": result.sql,
                "params": result.params,
                "df_json": df_json,
                "ts": _now_str(),
            }
        )

st.caption(
    "Guardrail: the assistant never executes free-form SQL and never computes forecasts. "
    "It queries stored Delta outputs and computes metrics deterministically."
)
