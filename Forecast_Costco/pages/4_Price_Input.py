 # pages/4_Price_Input.py
# ------------------------------------------------------------
# Price Input (Buyer UI) - Production-ready POC
#
# Buyer selects:
#   - DEPOT (15 depots)
#   - ITEM (Conventional / Organic)
#   - WEEK (1,2,3) with computed week start dates (ds)
#
# Write behavior:
#   1) Primary (optional): Google Sheets (if configured + creds available)
#   2) Fallback (always): Delta table upsert (DEPOT, ITEM, ds) -> price
#
# Read behavior (auto-revert):
#   - If Google Sheet isn't configured OR read fails OR blank:
#       fall back to Delta table last known price for that DEPOT+ITEM+ds
#
# Requires:
#   - Databricks SQL env vars (for reads + Delta fallback writes):
#       DBX_SQL_HOST, DBX_SQL_HTTP_PATH, DBX_SQL_TOKEN
#   - Optional Google Sheets env vars (for primary write):
#       GOOGLE_SHEETS_PRICE_URL
#       GOOGLE_SERVICE_ACCOUNT_JSON  OR  GOOGLE_SERVICE_ACCOUNT_FILE
#
# NOTE:
# - This page uses Databricks SQL "statements" REST API for writes (MERGE),
#   because services/dbx_sql.py is intentionally read-only for the rest of the app.
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import datetime as dt
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import requests

from app import get_app_health  # uses db_configured guard + gsheets status
from services.dbx_sql import query as dbx_query


# -----------------------------
# Config
# -----------------------------
# EXACT 15 depots (edit once if needed)
DEPOTS_15 = [172, 263, 265, 268, 270, 272, 275, 277, 280, 283, 286, 289, 292, 295, 298]

ITEMS = ["Conventional", "Organic"]

# Delta fallback table (create this in Databricks once; or let MERGE create the data if table exists)
# Recommended schema:
#   DEPOT INT, ITEM STRING, ds DATE, price DOUBLE, updated_at TIMESTAMP
PRICE_FALLBACK_TABLE = os.getenv(
    "PRICE_FALLBACK_TABLE",
    "ppe_dscoe.default.buyer_price_inputs_fallback",
).strip()

# Google Sheets URL (optional)
GOOGLE_SHEETS_PRICE_URL = (os.getenv("GOOGLE_SHEETS_PRICE_URL") or "").strip()
GOOGLE_SERVICE_ACCOUNT_JSON = (os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()
GOOGLE_SERVICE_ACCOUNT_FILE = (os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE") or "").strip()


# -----------------------------
# Small helpers
# -----------------------------
def _is_monday(d: dt.date) -> bool:
    return d.weekday() == 0  # Monday=0

def _next_monday(today: Optional[dt.date] = None) -> dt.date:
    today = today or dt.date.today()
    days_ahead = (0 - today.weekday()) % 7
    return today + dt.timedelta(days=days_ahead)

def _week_ds(base_monday: dt.date, week_n: int) -> dt.date:
    return base_monday + dt.timedelta(days=7 * (week_n - 1))

def _fmt_money(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

def _validate_inputs(depot: Any, item: Any, ds: Any, price: Any) -> Tuple[bool, str]:
    try:
        depot_i = int(depot)
    except Exception:
        return False, "Depot must be an integer."
    if depot_i not in DEPOTS_15:
        return False, "Depot must be one of the 15 supported depots."
    if item not in ITEMS:
        return False, "Item must be Conventional or Organic."
    if not isinstance(ds, dt.date):
        return False, "Week start date (ds) is invalid."
    try:
        p = float(price)
    except Exception:
        return False, "Price must be a number."
    if p < 0:
        return False, "Price must be non-negative."
    return True, ""


# -----------------------------
# Databricks SQL Statements API write (MERGE)
# -----------------------------
def _dbx_statement_endpoint() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Returns (host, http_path, token, warehouse_id) if configured else (None,...).
    """
    host = (os.getenv("DBX_SQL_HOST") or "").strip().replace("https://", "").replace("http://", "").strip().strip("/")
    http_path = (os.getenv("DBX_SQL_HTTP_PATH") or "").strip()
    token = (os.getenv("DBX_SQL_TOKEN") or "").strip()
    if not (host and http_path and token):
        return None, None, None, None

    # http_path example: /sql/1.0/warehouses/xxxxxxxxxxxxxxxx
    warehouse_id = http_path.split("/")[-1] if "/" in http_path else ""
    warehouse_id = warehouse_id.strip()
    if not warehouse_id:
        return host, http_path, token, None
    return host, http_path, token, warehouse_id

def _dbx_merge_price(depot: int, item: str, ds: dt.date, price: float) -> None:
    """
    Upserts into fallback Delta table using Databricks SQL Statement Execution API.
    This bypasses the read-only guard in services/dbx_sql.py (by design).
    """
    host, http_path, token, warehouse_id = _dbx_statement_endpoint()
    if not (host and token and warehouse_id):
        raise RuntimeError("Databricks SQL write is not configured (missing host/token/warehouse id).")

    # Strong validation + controlled formatting (avoid injection)
    depot = int(depot)
    if item not in ITEMS:
        raise RuntimeError("Invalid item.")
    ds_str = ds.isoformat()
    price = float(price)

    # MERGE by (DEPOT, ITEM, ds)
    # Table must already exist (recommended). If it doesn't, create it once in Databricks.
    sql = f"""
MERGE INTO {PRICE_FALLBACK_TABLE} AS t
USING (
  SELECT
    CAST({depot} AS INT) AS DEPOT,
    '{'Conventional' if item == 'Conventional' else 'Organic'}' AS ITEM,
    CAST('{ds_str}' AS DATE) AS ds,
    CAST({price} AS DOUBLE) AS price,
    current_timestamp() AS updated_at
) AS s
ON t.DEPOT = s.DEPOT AND t.ITEM = s.ITEM AND t.ds = s.ds
WHEN MATCHED THEN UPDATE SET
  t.price = s.price,
  t.updated_at = s.updated_at
WHEN NOT MATCHED THEN INSERT (DEPOT, ITEM, ds, price, updated_at)
VALUES (s.DEPOT, s.ITEM, s.ds, s.price, s.updated_at)
"""

    url = f"https://{host}/api/2.0/sql/statements"
    payload = {
        "warehouse_id": warehouse_id,
        "statement": sql.strip(),
        "wait_timeout": "15s",
        "on_wait_timeout": "CANCEL",
    }

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=20,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Databricks write failed ({resp.status_code}): {resp.text}")


def _read_delta_price(depot: int, item: str, ds: dt.date) -> Optional[float]:
    """
    Reads fallback price for exact (depot,item,ds). Returns None if missing.
    """
    sql = f"""
SELECT price
FROM {PRICE_FALLBACK_TABLE}
WHERE DEPOT = ? AND ITEM = ? AND ds = ?
ORDER BY updated_at DESC
LIMIT 1
"""
    rows = dbx_query(sql, (int(depot), item, ds.isoformat()))
    if not rows:
        return None
    v = rows[0].get("price")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


# -----------------------------
# Google Sheets optional read/write
# (Lightweight, best-effort. If libs not installed or creds missing, we skip.)
# -----------------------------
def _gsheets_enabled() -> bool:
    return bool(GOOGLE_SHEETS_PRICE_URL) and (bool(GOOGLE_SERVICE_ACCOUNT_JSON) or bool(GOOGLE_SERVICE_ACCOUNT_FILE))

def _open_gsheet():
    """
    Returns gspread worksheet object (first sheet) if available.
    Expected sheet format (recommended):
      Columns: DEPOT | ITEM | ds | price | updated_at
    """
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception as e:
        raise RuntimeError("Google Sheets libraries not installed. Install: gspread google-auth") from e

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    if GOOGLE_SERVICE_ACCOUNT_JSON:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = Credentials.from_service_account_info(info, scopes=scopes)
    else:
        creds = Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=scopes)

    gc = gspread.authorize(creds)

    # Open by URL
    sh = gc.open_by_url(GOOGLE_SHEETS_PRICE_URL)
    ws = sh.sheet1
    return ws

def _gsheets_upsert_price(depot: int, item: str, ds: dt.date, price: float) -> None:
    """
    Upsert by key (DEPOT, ITEM, ds) into Google Sheet.
    """
    ws = _open_gsheet()
    ds_str = ds.isoformat()

    # Read header + rows
    values = ws.get_all_values()
    if not values:
        # Create headers
        ws.append_row(["DEPOT", "ITEM", "ds", "price", "updated_at"])
        values = ws.get_all_values()

    header = values[0]
    # Ensure columns exist
    required = ["DEPOT", "ITEM", "ds", "price", "updated_at"]
    if header != required:
        raise RuntimeError("Google Sheet header must be: DEPOT, ITEM, ds, price, updated_at")

    # Find existing row
    for idx in range(1, len(values)):
        r = values[idx]
        if len(r) >= 3 and r[0] == str(depot) and r[1] == item and r[2] == ds_str:
            # Update price + updated_at
            ws.update(f"D{idx+1}:E{idx+1}", [[str(price), dt.datetime.utcnow().isoformat()]])
            return

    # Otherwise append
    ws.append_row([str(depot), item, ds_str, str(price), dt.datetime.utcnow().isoformat()])

def _gsheets_read_price(depot: int, item: str, ds: dt.date) -> Optional[float]:
    """
    Best-effort read from Google Sheet for exact (DEPOT, ITEM, ds). Returns None if missing/unavailable.
    """
    ws = _open_gsheet()
    ds_str = ds.isoformat()
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return None
    header = values[0]
    if header[:4] != ["DEPOT", "ITEM", "ds", "price"]:
        return None

    for idx in range(1, len(values)):
        r = values[idx]
        if len(r) >= 4 and r[0] == str(depot) and r[1] == item and r[2] == ds_str:
            try:
                return float(r[3]) if r[3] != "" else None
            except Exception:
                return None
    return None


# -----------------------------
# Page UI
# -----------------------------
st.set_page_config(page_title="Price Input", page_icon="💲", layout="wide")
st.title("💲 Price Input")
st.caption("Buyer enters **Depot × Item × Week (1–3)** prices. Week start dates (ds) are shown explicitly.")

health = get_app_health()
if not health.db_configured:
    st.warning(
        "Databricks SQL is not configured. This page needs Databricks for fallback storage (Delta). "
        "Set DBX_SQL_HOST / DBX_SQL_HTTP_PATH / DBX_SQL_TOKEN."
    )
    st.stop()

gs_enabled = _gsheets_enabled()
if gs_enabled:
    st.success("Google Sheets: Configured (primary write enabled)")
else:
    st.info("Google Sheets: Not configured (prices will still save to Delta fallback table).")

st.divider()

# Base Monday (week 1 ds)
default_monday = _next_monday(dt.date.today())
base_monday = st.date_input(
    "Forecast Week 1 start date (Monday)",
    value=default_monday,
    help="Week 1 is this date. Week 2/3 are +7/+14 days. (Must be Monday.)",
)

if not _is_monday(base_monday):
    st.error("Please pick a Monday for Week 1 start date.")
    st.stop()

c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
with c1:
    depot = st.selectbox("DEPOT (15)", options=DEPOTS_15, index=0)
with c2:
    item = st.selectbox("ITEM", options=ITEMS, index=0)
with c3:
    week_n = st.selectbox("WEEK", options=[1, 2, 3], index=0)

ds = _week_ds(base_monday, int(week_n))

st.markdown("#### Selected key")
k1, k2, k3 = st.columns(3)
k1.metric("DEPOT", str(depot))
k2.metric("ITEM", item)
k3.metric("ds (week start)", ds.isoformat())

# Determine "effective" current price (sheet first, else delta)
sheet_price = None
delta_price = None

with st.spinner("Loading last known price…"):
    # Sheet read (best-effort)
    if gs_enabled:
        try:
            sheet_price = _gsheets_read_price(int(depot), item, ds)
        except Exception:
            sheet_price = None

    # Delta read (fallback)
    try:
        delta_price = _read_delta_price(int(depot), item, ds)
    except Exception:
        delta_price = None

effective_price = sheet_price if sheet_price is not None else delta_price

st.markdown("#### Current price (auto-revert logic)")
m1, m2, m3 = st.columns(3)
m1.metric("From Google Sheet", _fmt_money(sheet_price) if sheet_price is not None else "—")
m2.metric("From Delta fallback", _fmt_money(delta_price) if delta_price is not None else "—")
m3.metric("Effective price used", _fmt_money(effective_price) if effective_price is not None else "—")

st.divider()

st.markdown("### Enter / update price")
price_value = st.number_input(
    "Price ($)",
    min_value=0.0,
    value=float(effective_price) if effective_price is not None else 0.0,
    step=0.01,
    format="%.2f",
    help="This will upsert to Delta fallback. If Google Sheets is configured, it will upsert there too.",
)

colA, colB = st.columns([1, 2])
with colA:
    save = st.button("✅ Save price", use_container_width=True)
with colB:
    st.caption(
        f"Delta fallback table: `{PRICE_FALLBACK_TABLE}` • "
        f"{'Google Sheets write ON' if gs_enabled else 'Google Sheets write OFF'}"
    )

if save:
    ok, err = _validate_inputs(depot, item, ds, price_value)
    if not ok:
        st.error(err)
    else:
        # 1) Write to Google Sheets (best-effort)
        gs_ok = False
        gs_err = None
        if gs_enabled:
            try:
                _gsheets_upsert_price(int(depot), item, ds, float(price_value))
                gs_ok = True
            except Exception as e:
                gs_ok = False
                gs_err = str(e)

        # 2) Always write to Delta fallback (required for revert)
        try:
            _dbx_merge_price(int(depot), item, ds, float(price_value))
            st.success(
                f"Saved **{_fmt_money(price_value)}** for DEPOT **{depot}**, ITEM **{item}**, ds **{ds.isoformat()}** "
                f"to Delta fallback."
            )
        except Exception as e:
            st.error(f"Delta fallback write failed: {e}")
            st.stop()

        if gs_enabled:
            if gs_ok:
                st.success("Also saved to Google Sheet (primary).")
            else:
                st.warning(f"Google Sheet write failed (Delta fallback still saved): {gs_err}")

        st.rerun()

st.divider()

with st.expander("How auto-revert works", expanded=False):
    st.markdown(
        """
- The UI tries to read **Google Sheet** first for the selected `(DEPOT, ITEM, ds)`.
- If that read fails or is blank, the UI falls back to the **Delta fallback** table.
- Every save **always** upserts into Delta fallback, so the “last known price” is always available.
        """
    )

with st.expander("Delta fallback table schema (recommended)", expanded=False):
    st.code(
        f"""
-- Run once in Databricks if the table doesn't exist yet:
CREATE TABLE IF NOT EXISTS {PRICE_FALLBACK_TABLE} (
  DEPOT INT,
  ITEM STRING,
  ds DATE,
  price DOUBLE,
  updated_at TIMESTAMP
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite'='true','delta.autoOptimize.autoCompact'='true');
        """.strip(),
        language="sql",
    )
