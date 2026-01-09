# app.py
# ------------------------------------------------------------
# Forecasting & D&D Analytics Assistant (Production-Ready POC)
#
# Goals:
# - Buyer-friendly multipage UI (Streamlit pages/ directory)
# - Centralized configuration for Databricks + Google Sheets
# - Offline-safe: app renders even if Databricks and/or Google Sheets are not configured
# - Expose shared helpers for pages:
#     - TABLES (single source of truth for Delta tables)
#     - run_query(sql_text, params=None) (read-only, parameterized)
#     - get_app_health() (status flags for pages)
#
# IMPORTANT:
# - Do NOT call ensure_session_state_ready() at import time.
#   Streamlit imports pages in different orders; calling it too early can cause NameError.
# - Instead, call ensure_session_state_ready() inside main() (and pages can call it too if needed).
# ------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import streamlit as st

# --- Core services (Databricks + queries) ---
from services.dbx_sql import is_configured as dbx_is_configured, query as dbx_query
from services.queries import QueryService, TableRegistry

Params = Optional[Union[Tuple[Any, ...], Sequence[Any]]]


# ----------------------------
# App Config
# ----------------------------
APP_NAME = os.getenv("APP_NAME", "Forecasting & D&D Analytics Assistant")
APP_ENV = os.getenv("APP_ENV", "POC")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

# Databricks catalog/schema defaults (per your standard: ppe_dscoe.default)
DEFAULT_CATALOG = os.getenv("DBX_SQL_CATALOG", "ppe_dscoe")
DEFAULT_SCHEMA = os.getenv("DBX_SQL_SCHEMA", "default")

# Google Sheets URL (optional)
GOOGLE_SHEETS_PRICE_URL = (os.getenv("GOOGLE_SHEETS_PRICE_URL") or "").strip()

# (Optional) service account credential options (support either)
GOOGLE_SERVICE_ACCOUNT_JSON = (os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()
GOOGLE_SERVICE_ACCOUNT_FILE = (os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE") or "").strip()


# ----------------------------
# Table Registry (single source of truth)
# ----------------------------
def build_table_registry(catalog: str = DEFAULT_CATALOG, schema: str = DEFAULT_SCHEMA) -> TableRegistry:
    """
    Central place to define the Delta tables the app reads from.
    Update these names once and the whole UI follows.
    """
    def fq(t: str) -> str:
        return f"{catalog}.{schema}.{t}"

    return TableRegistry(
        # Forecasts (append weekly)
        forecast=fq("placeholder_forecast_ouputtable"),
        # Backtesting (append weekly)
        backtest=fq("placeholder_backtest_outputtable"),
        # 7 years historical sales
        actuals=fq("ex_sales_table_historical"),
        # D&D monitoring
        dnd_company=fq("company_level_blueberries_dnd_monitoring"),
        dnd_depot_current=fq("depot_level_current_week_blueberries_dnd_monitoring"),
        dnd_depot_hist=fq("depot_level_historical_dnd_monitoring_data"),
    )


# Public constant used by pages (do NOT rename without updating pages)
# NOTE: This reads env vars at import time; if you change env vars after launch, restart Streamlit.
_TABLE_REGISTRY = build_table_registry()
TABLES: Dict[str, str] = {
    "forecast": _TABLE_REGISTRY.forecast,
    "backtest": _TABLE_REGISTRY.backtest,
    "actuals": _TABLE_REGISTRY.actuals,
    "dnd_company": _TABLE_REGISTRY.dnd_company,
    "dnd_depot_current": _TABLE_REGISTRY.dnd_depot_current,
    "dnd_depot_hist": _TABLE_REGISTRY.dnd_depot_hist,
}


# ----------------------------
# Shared Databricks query helper (used by pages)
# ----------------------------
def run_query(sql_text: str, params: Params = None) -> List[Dict[str, Any]]:
    """
    Single read-only query entrypoint for pages.
    - Uses services.dbx_sql.query
    - Always returns: list[dict]
    """
    return dbx_query(sql_text, params)


# ----------------------------
# Shared Query Service (optional, for new pages)
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_query_service() -> QueryService:
    """
    Cached QueryService so pages can reuse it without recreating.
    """
    tables = build_table_registry()
    return QueryService(query_fn=lambda sql, params=None: run_query(sql, params), tables=tables)


# ----------------------------
# Google Sheets (optional/offline friendly)
# ----------------------------
@dataclass(frozen=True)
class GSheetsStatus:
    url_configured: bool
    auth_configured: bool
    display_url: str


def _gsheets_auth_configured() -> bool:
    # Either raw JSON (recommended for Cloud Run) OR a file path (local dev)
    return bool(GOOGLE_SERVICE_ACCOUNT_JSON) or bool(GOOGLE_SERVICE_ACCOUNT_FILE)


def get_gsheets_status() -> GSheetsStatus:
    """
    Resilient approach:
    - If services.gsheets_config exists later, use it
    - Otherwise fall back to env vars
    """
    try:
        from services.gsheets_config import (  # type: ignore
            get_sheet_url,
            is_url_configured,
            is_auth_configured,
        )

        return GSheetsStatus(
            url_configured=bool(is_url_configured()),
            auth_configured=bool(is_auth_configured()),
            display_url=get_sheet_url() or "(not set)",
        )
    except Exception:
        return GSheetsStatus(
            url_configured=bool(GOOGLE_SHEETS_PRICE_URL),
            auth_configured=_gsheets_auth_configured(),
            display_url=GOOGLE_SHEETS_PRICE_URL or "(not set)",
        )


# ----------------------------
# App health (used by pages)
# ----------------------------
@dataclass(frozen=True)
class AppHealth:
    db_configured: bool
    gsheets_url_configured: bool
    gsheets_auth_configured: bool
    offline_mode: bool


def get_app_health() -> AppHealth:
    gs = get_gsheets_status()
    db_ok = bool(dbx_is_configured())
    return AppHealth(
        db_configured=db_ok,
        gsheets_url_configured=bool(gs.url_configured),
        gsheets_auth_configured=bool(gs.auth_configured),
        offline_mode=(not db_ok),
    )


# ----------------------------
# Session state bootstrap (pages can rely on this)
# ----------------------------
def ensure_session_state_ready() -> None:
    """
    Ensures shared objects exist in st.session_state for all pages:
      - st.session_state["tables"]        -> TableRegistry
      - st.session_state["query_service"] -> QueryService
      - st.session_state["gsheets_status"]-> GSheetsStatus
      - st.session_state["app_health"]    -> AppHealth
    Safe to call multiple times.
    """
    if "tables" not in st.session_state:
        st.session_state["tables"] = build_table_registry()
    if "query_service" not in st.session_state:
        st.session_state["query_service"] = get_query_service()
    if "gsheets_status" not in st.session_state:
        st.session_state["gsheets_status"] = get_gsheets_status()
    if "app_health" not in st.session_state:
        st.session_state["app_health"] = get_app_health()


# ----------------------------
# UI Helpers
# ----------------------------
def render_topbar() -> None:
    left, right = st.columns([3, 1])
    with left:
        st.title(APP_NAME)
        st.caption(f"Environment: **{APP_ENV}** · Version: **{APP_VERSION}**")
    with right:
        st.write("")
        st.write("")
        st.caption("Status")


def render_status_banners() -> None:
    # Databricks status
    if dbx_is_configured():
        st.success("Databricks SQL: Connected")
    else:
        st.warning(
            "Databricks SQL: Not configured. Running in **offline UI mode** (tables will appear empty). "
            "Set DBX_SQL_HOST / DBX_SQL_HTTP_PATH / DBX_SQL_TOKEN to enable live data."
        )

    # Google Sheets status
    gs = get_gsheets_status()
    if gs.url_configured and gs.auth_configured:
        st.info("Google Sheets: Configured (price entry writes enabled)")
    elif gs.url_configured and not gs.auth_configured:
        st.warning(
            "Google Sheets: URL is set, but credentials are not. "
            "Price Input page will render, but **writes will be disabled** until service account credentials are added."
        )
    else:
        st.warning(
            "Google Sheets: Not configured. Price Input can be viewed, but **writes will be disabled**. "
            "Set GOOGLE_SHEETS_PRICE_URL (and service account credentials) to enable writes."
        )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Navigation")
        st.caption("Use the pages menu (top-left) or these quick links:")

        st.page_link("pages/1_Run_Dashboard.py", label="Run Dashboard", icon="📊")
        st.page_link("pages/2_Forecast_Explorer.py", label="Forecast Explorer", icon="🔎")
        st.page_link("pages/3_Model_Performance.py", label="Model Performance", icon="✅")
        st.page_link("pages/5_DnD_Monitoring.py", label="D&D Monitoring", icon="📉")
        st.page_link("pages/6_Ask_Assistant.py", label="Ask Assistant", icon="🧠")

        st.divider()
        st.page_link("pages/4_Price_Input.py", label="Price Input", icon="💲")
        st.caption(
            "Buyer enters **Depot × Item × Week** prices here.\n"
            "This writes to the configured Google Sheet which your Databricks model parses."
        )

        st.divider()
        st.subheader("Guardrails")
        st.caption(
            "• Forecast numbers come from Databricks model runs.\n"
            "• Combined is stored as a first-class item (not LLM-derived).\n"
            "• App queries are read-only and row-limited.\n"
            "• Assistant uses deterministic math for calculations."
        )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ✅ Critical: initialize session_state AFTER all functions exist
    ensure_session_state_ready()

    render_topbar()
    render_sidebar()
    render_status_banners()

    st.markdown(
        """
### Welcome
This application provides a single buyer-facing interface for:
- **Weekly forecasts** (depots × items × 3-week horizon)
- **Model performance** (backtesting vs actuals)
- **Dump & Destroy (D&D)** monitoring (company + depot)
- **Price Input** (buyer enters horizon prices → updates Google Sheet → Databricks model parses it)
- **Guided analytics** via the assistant (deterministic math + safe routing)

Use the **Navigation** links in the sidebar to get started.
"""
    )

    # Helpful next steps based on configuration
    if not dbx_is_configured():
        with st.expander("Enable live Databricks data", expanded=False):
            st.code(
                "Set environment variables:\n"
                "  DBX_SQL_HOST\n"
                "  DBX_SQL_HTTP_PATH\n"
                "  DBX_SQL_TOKEN\n"
                "Optional:\n"
                "  DBX_SQL_CATALOG=ppe_dscoe\n"
                "  DBX_SQL_SCHEMA=default\n",
                language="text",
            )

    gs = get_gsheets_status()
    if not gs.url_configured:
        with st.expander("Enable buyer price writes to Google Sheet", expanded=False):
            st.code(
                "Set environment variable:\n"
                "  GOOGLE_SHEETS_PRICE_URL=<full google sheet url>\n\n"
                "Then add ONE of the following for Google Sheets API auth:\n"
                "  GOOGLE_SERVICE_ACCOUNT_JSON=<service account json string>\n"
                "  OR\n"
                "  GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/service_account.json\n",
                language="text",
            )
    else:
        st.caption(f"Configured price sheet URL: {gs.display_url}")
        if gs.url_configured and not gs.auth_configured:
            st.info("Price sheet URL is configured. Add service account credentials to enable writing prices from Streamlit.")


if __name__ == "__main__":
    main()
