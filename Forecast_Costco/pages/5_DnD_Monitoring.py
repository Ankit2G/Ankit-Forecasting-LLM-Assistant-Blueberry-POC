# pages/5_DnD_Monitoring.py
# ------------------------------------------------------------
# D&D Monitoring — Buyer-Facing, Production-Grade POC
#
# What’s improved vs. the older draft:
# ✅ No circular imports (does NOT import app.py)
# ✅ Uses TableRegistry from st.session_state["tables"]
# ✅ Offline-safe: UI renders without Databricks configured
# ✅ Proper SQL (no unnecessary GROUP BY that can distort metrics)
# ✅ Clean buyer UX: tabs + KPIs + charts + drilldowns + exports
# ✅ YoY computed deterministically in Python (aligned by 52 weeks = 364 days)
#
# Tables (from TableRegistry):
#   - company_level_blueberries_dnd_monitoring
#   - depot_level_current_week_blueberries_dnd_monitoring
#   - depot_level_historical_dnd_monitoring_data
#
# Expected columns (company):
#   FSCL_WK_START_DATE, Total_Units_Sold, Total_Sales_Amt, Total_DND_QTY,
#   Total_Units, Total_DND_$_loss
#
# Expected columns (depot current & hist):
#   DEPOT, FSCL_WK_START_DATE, Volume, Total_DND_QTY, Total_Units,
#   Total_Sales_Amt, Total_DND_$_loss
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, List, Optional, Tuple

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


def _ensure_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _compute_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures a numeric 'dnd_rate' column exists:
      dnd_rate = Total_DND_QTY / Total_Units
    """
    if df.empty:
        return df
    if "dnd_rate" not in df.columns:
        if "Total_DND_QTY" in df.columns and "Total_Units" in df.columns:
            num = pd.to_numeric(df["Total_DND_QTY"], errors="coerce")
            den = pd.to_numeric(df["Total_Units"], errors="coerce").replace({0: pd.NA})
            df["dnd_rate"] = num / den
        else:
            df["dnd_rate"] = pd.NA
    else:
        df["dnd_rate"] = pd.to_numeric(df["dnd_rate"], errors="coerce")
    return df


def _add_yoy(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Adds YOY columns by aligning date - 364 days (52 weeks).
    Output columns:
      - yoy_rate
      - yoy_pp_change (percentage-point change as fraction)
      - yoy_pct_change (relative change)
    """
    if df.empty:
        return df

    df = df.copy()
    df = _ensure_dates(df, date_col)
    df = _compute_rate(df)

    # map date -> rate
    base = df[[date_col, "dnd_rate"]].dropna(subset=[date_col]).drop_duplicates(subset=[date_col]).copy()
    base["_date"] = base[date_col].dt.date
    rate_map = dict(zip(base["_date"], base["dnd_rate"]))

    df["_date"] = df[date_col].dt.date
    df["_yoy_date"] = (df[date_col] - pd.to_timedelta(364, unit="D")).dt.date

    df["yoy_rate"] = df["_yoy_date"].map(rate_map)
    df["yoy_pp_change"] = df["dnd_rate"] - pd.to_numeric(df["yoy_rate"], errors="coerce")
    df["yoy_pct_change"] = (df["dnd_rate"] / pd.to_numeric(df["yoy_rate"], errors="coerce")) - 1.0

    return df.drop(columns=["_date", "_yoy_date"], errors="ignore")


def _chart_rate_trend(df: pd.DataFrame, date_col: str, rate_col: str, yoy_col: str, height: int = 320) -> alt.Chart:
    plot = df[[date_col, rate_col, yoy_col]].copy()
    plot = plot.dropna(subset=[date_col])
    plot = plot.rename(columns={date_col: "date", rate_col: "Current", yoy_col: "YoY"})
    plot = plot.melt(id_vars=["date"], var_name="series", value_name="rate")
    return (
        alt.Chart(plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Week start"),
            y=alt.Y("rate:Q", title="D&D rate (fraction)"),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip("date:T", title="Week"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("rate:Q", title="Rate", format=".4f"),
            ],
        )
        .properties(height=height)
    )


def _chart_money_trend(df: pd.DataFrame, date_col: str, money_col: str, height: int = 280) -> alt.Chart:
    plot = df[[date_col, money_col]].copy().dropna(subset=[date_col])
    plot = plot.rename(columns={date_col: "date", money_col: "loss"})
    return (
        alt.Chart(plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Week start"),
            y=alt.Y("loss:Q", title="$ loss"),
            tooltip=[
                alt.Tooltip("date:T", title="Week"),
                alt.Tooltip("loss:Q", title="$ loss", format=",.0f"),
            ],
        )
        .properties(height=height)
    )


# -----------------------------
# Cached fetches
# -----------------------------
@st.cache_data(show_spinner=False, ttl=120)
def fetch_company_history(company_table: str, weeks_back: int) -> pd.DataFrame:
    # Pull latest N rows by date (no extra GROUP BY)
    sql = f"""
        SELECT
            FSCL_WK_START_DATE,
            Total_Units_Sold,
            Total_Sales_Amt,
            Total_DND_QTY,
            Total_Units,
            Total_DND_$_loss
        FROM {company_table}
        ORDER BY FSCL_WK_START_DATE DESC
        LIMIT {int(weeks_back)}
    """
    df = _sql_df(sql)
    if df.empty:
        return df
    df = _ensure_dates(df, "FSCL_WK_START_DATE")
    for c in ["Total_Units_Sold", "Total_Sales_Amt", "Total_DND_QTY", "Total_Units", "Total_DND_$_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("FSCL_WK_START_DATE")
    df = _compute_rate(df)
    return df


@st.cache_data(show_spinner=False, ttl=120)
def fetch_available_current_weeks(depot_current_table: str, limit: int = 60) -> List[str]:
    sql = f"""
        SELECT DISTINCT CAST(FSCL_WK_START_DATE AS STRING) AS wk
        FROM {depot_current_table}
        ORDER BY wk DESC
        LIMIT {int(limit)}
    """
    df = _sql_df(sql)
    if df.empty or "wk" not in df.columns:
        return []
    return [str(x) for x in df["wk"].dropna().tolist()]


@st.cache_data(show_spinner=False, ttl=120)
def fetch_depot_current_snapshot(depot_current_table: str, week_start: str) -> pd.DataFrame:
    sql = f"""
        SELECT
            DEPOT,
            FSCL_WK_START_DATE,
            Volume,
            Total_DND_QTY,
            Total_Units,
            Total_Sales_Amt,
            Total_DND_$_loss
        FROM {depot_current_table}
        WHERE FSCL_WK_START_DATE = ?
        ORDER BY DEPOT
    """
    df = _sql_df(sql, (week_start,))
    if df.empty:
        return df
    df = _ensure_dates(df, "FSCL_WK_START_DATE")
    for c in ["Volume", "Total_DND_QTY", "Total_Units", "Total_Sales_Amt", "Total_DND_$_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = _compute_rate(df)
    return df


@st.cache_data(show_spinner=False, ttl=120)
def fetch_depot_history(depot_hist_table: str, depot: int, weeks_back: int) -> pd.DataFrame:
    sql = f"""
        SELECT
            DEPOT,
            FSCL_WK_START_DATE,
            Volume,
            Total_DND_QTY,
            Total_Units,
            Total_Sales_Amt,
            Total_DND_$_loss
        FROM {depot_hist_table}
        WHERE DEPOT = ?
        ORDER BY FSCL_WK_START_DATE DESC
        LIMIT {int(weeks_back)}
    """
    df = _sql_df(sql, (int(depot),))
    if df.empty:
        return df
    df = _ensure_dates(df, "FSCL_WK_START_DATE")
    for c in ["Volume", "Total_DND_QTY", "Total_Units", "Total_Sales_Amt", "Total_DND_$_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("FSCL_WK_START_DATE")
    df = _compute_rate(df)
    return df


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="D&D Monitoring", page_icon="📉", layout="wide")

st.header("📉 D&D Monitoring")
st.caption("Company-level YoY D&D trends + depot-level snapshot and drilldowns (deterministic).")

tables = _get_tables()
COMPANY_TABLE = getattr(tables, "dnd_company", "ppe_dscoe.default.company_level_blueberries_dnd_monitoring")
DEPOT_CURRENT_TABLE = getattr(tables, "dnd_depot_current", "ppe_dscoe.default.depot_level_current_week_blueberries_dnd_monitoring")
DEPOT_HIST_TABLE = getattr(tables, "dnd_depot_hist", "ppe_dscoe.default.depot_level_historical_dnd_monitoring_data")

# Offline-safe shell
if not dbx_is_configured():
    st.warning(
        "Databricks SQL is not configured. Set DBX_SQL_HOST / DBX_SQL_HTTP_PATH / DBX_SQL_TOKEN to load D&D data."
    )
    st.info("You can still navigate the app UI. This page will populate once the connection is enabled.")
    st.stop()

# Controls
c1, c2, c3 = st.columns([1.6, 1.6, 2.8], vertical_alignment="bottom")
with c1:
    weeks_back = st.selectbox("Company trend window", options=[13, 26, 52, 104], index=2)
with c2:
    depot_hist_weeks = st.selectbox("Depot drilldown window", options=[26, 52, 104, 156], index=1)
with c3:
    st.markdown(
        "This page shows:\n"
        "- Company D&D rate + $ loss trend\n"
        "- YoY comparisons (aligned by 52 weeks)\n"
        "- Depot snapshot for the selected week + drilldown trends"
    )

st.divider()

tab_company, tab_depot = st.tabs(["🏢 Company", "🏬 Depots"])

# -----------------------------
# Company tab
# -----------------------------
with tab_company:
    st.subheader("Company-level D&D")
    st.caption("Rate and $ loss trend with YoY overlay (deterministic).")

    company_df = fetch_company_history(COMPANY_TABLE, int(weeks_back))
    if company_df.empty:
        st.info(f"No rows found in `{COMPANY_TABLE}`.")
    else:
        company_df = _add_yoy(company_df, "FSCL_WK_START_DATE")

        latest = company_df.dropna(subset=["FSCL_WK_START_DATE"]).iloc[-1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Latest D&D rate", _fmt_rate(latest.get("dnd_rate"), 2))
        pp = latest.get("yoy_pp_change")
        k2.metric("YoY change (pp)", "—" if pd.isna(pp) else f"{float(pp) * 100:+.2f} pp")
        k3.metric("Latest D&D $ loss", _fmt_money(latest.get("Total_DND_$_loss")))
        k4.metric("Latest DND qty", _fmt_int(latest.get("Total_DND_QTY")))

        left, right = st.columns([2.0, 1.2], gap="large")
        with left:
            st.markdown("#### D&D rate trend (Current vs YoY)")
            st.altair_chart(
                _chart_rate_trend(company_df, "FSCL_WK_START_DATE", "dnd_rate", "yoy_rate", height=340),
                use_container_width=True,
            )
        with right:
            st.markdown("#### D&D $ loss trend")
            st.altair_chart(
                _chart_money_trend(company_df, "FSCL_WK_START_DATE", "Total_DND_$_loss", height=340),
                use_container_width=True,
            )

        with st.expander("Company table (download-ready)"):
            disp = company_df.copy()
            disp["D&D rate"] = disp["dnd_rate"].apply(lambda x: _fmt_rate(x, 2))
            disp["YoY rate"] = disp["yoy_rate"].apply(lambda x: _fmt_rate(x, 2))
            disp["YoY pp"] = disp["yoy_pp_change"].apply(lambda x: "—" if pd.isna(x) else f"{float(x) * 100:+.2f} pp")
            disp["D&D $ loss"] = disp["Total_DND_$_loss"].apply(_fmt_money)
            disp["Total units"] = disp["Total_Units"].apply(_fmt_int)
            disp["DND qty"] = disp["Total_DND_QTY"].apply(_fmt_int)

            show_cols = [
                "FSCL_WK_START_DATE",
                "D&D rate",
                "YoY rate",
                "YoY pp",
                "D&D $ loss",
                "DND qty",
                "Total units",
            ]
            st.dataframe(disp[show_cols].sort_values("FSCL_WK_START_DATE", ascending=False), use_container_width=True, hide_index=True)

            csv = company_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download company D&D CSV",
                data=csv,
                file_name="company_dnd_history.csv",
                mime="text/csv",
            )

# -----------------------------
# Depot tab
# -----------------------------
with tab_depot:
    st.subheader("Depot-level D&D")
    st.caption("Snapshot for a selected week + depot drilldown trends (deterministic).")

    weeks_available = fetch_available_current_weeks(DEPOT_CURRENT_TABLE, limit=60)
    if not weeks_available:
        st.info(f"No snapshot weeks found in `{DEPOT_CURRENT_TABLE}`.")
    else:
        controls = st.columns([1.6, 1.6, 2.0], vertical_alignment="bottom")
        with controls[0]:
            selected_week = st.selectbox("Snapshot week", options=weeks_available, index=0)
        with controls[1]:
            rank_by = st.selectbox("Rank depots by", options=["D&D rate", "D&D $ loss", "DND qty"], index=0)
        with controls[2]:
            top_n = st.slider("Show top N depots", min_value=5, max_value=20, value=15, step=1)

        snap = fetch_depot_current_snapshot(DEPOT_CURRENT_TABLE, selected_week)
        if snap.empty:
            st.info("No depot snapshot rows for this week.")
        else:
            rank_df = snap.copy()
            if rank_by == "D&D rate":
                rank_df = rank_df.sort_values("dnd_rate", ascending=False)
            elif rank_by == "D&D $ loss":
                rank_df = rank_df.sort_values("Total_DND_$_loss", ascending=False)
            else:
                rank_df = rank_df.sort_values("Total_DND_QTY", ascending=False)

            left, right = st.columns([1.55, 1.0], gap="large")
            with left:
                st.markdown("#### Current week snapshot")
                st.dataframe(
                    rank_df.head(int(top_n)),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "DEPOT": st.column_config.NumberColumn("DEPOT", format="%d"),
                        "FSCL_WK_START_DATE": st.column_config.DatetimeColumn("Week", format="YYYY-MM-DD"),
                        "Volume": st.column_config.NumberColumn("Volume", format="%,.0f"),
                        "Total_DND_QTY": st.column_config.NumberColumn("DND Qty", format="%,.0f"),
                        "Total_Units": st.column_config.NumberColumn("Total Units", format="%,.0f"),
                        "Total_Sales_Amt": st.column_config.NumberColumn("Sales Amt", format="$%,.0f"),
                        "Total_DND_$_loss": st.column_config.NumberColumn("D&D $ loss", format="$%,.0f"),
                        "dnd_rate": st.column_config.NumberColumn("D&D rate (fraction)", format="%.4f"),
                    },
                    height=520,
                )
                csv = rank_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download snapshot CSV",
                    data=csv,
                    file_name=f"depot_dnd_snapshot_{selected_week}.csv",
                    mime="text/csv",
                )

            with right:
                st.markdown("#### Drilldown")
                depot_list = sorted(rank_df["DEPOT"].dropna().unique().tolist())
                default_depot = st.session_state.get("selected_depot")
                try:
                    default_depot_int = int(default_depot) if default_depot else None
                except Exception:
                    default_depot_int = None

                depot_pick = st.selectbox(
                    "Select depot",
                    options=depot_list,
                    index=(depot_list.index(default_depot_int) if default_depot_int in depot_list else 0),
                )
                st.session_state["selected_depot"] = str(depot_pick)

                # Snapshot KPIs for depot
                drow = snap[snap["DEPOT"] == depot_pick]
                if not drow.empty:
                    d = drow.iloc[0]
                    st.metric("Depot D&D rate (snapshot)", _fmt_rate(d.get("dnd_rate"), 2))
                    st.metric("Depot D&D $ loss (snapshot)", _fmt_money(d.get("Total_DND_$_loss")))
                    st.metric("Depot volume (snapshot)", _fmt_int(d.get("Volume")))

                show_table = st.checkbox("Show detailed depot trend table", value=False)

            st.markdown("### Depot trend and YoY")
            depot_hist = fetch_depot_history(DEPOT_HIST_TABLE, int(depot_pick), int(depot_hist_weeks))
            if depot_hist.empty:
                st.info(f"No depot history found in `{DEPOT_HIST_TABLE}` for depot {depot_pick}.")
            else:
                depot_hist = _add_yoy(depot_hist, "FSCL_WK_START_DATE")

                latest = depot_hist.dropna(subset=["FSCL_WK_START_DATE"]).iloc[-1]
                k1, k2, k3 = st.columns(3)
                k1.metric("Latest D&D rate", _fmt_rate(latest.get("dnd_rate"), 2))
                pp = latest.get("yoy_pp_change")
                k2.metric("YoY change (pp)", "—" if pd.isna(pp) else f"{float(pp) * 100:+.2f} pp")
                k3.metric("Latest D&D $ loss", _fmt_money(latest.get("Total_DND_$_loss")))

                c1, c2 = st.columns([2.0, 1.2], gap="large")
                with c1:
                    st.markdown("#### D&D rate trend (Current vs YoY)")
                    st.altair_chart(
                        _chart_rate_trend(depot_hist, "FSCL_WK_START_DATE", "dnd_rate", "yoy_rate", height=340),
                        use_container_width=True,
                    )
                with c2:
                    st.markdown("#### D&D $ loss trend")
                    st.altair_chart(
                        _chart_money_trend(depot_hist, "FSCL_WK_START_DATE", "Total_DND_$_loss", height=340),
                        use_container_width=True,
                    )

                if show_table:
                    disp = depot_hist.copy()
                    disp["D&D rate"] = disp["dnd_rate"].apply(lambda x: _fmt_rate(x, 2))
                    disp["YoY rate"] = disp["yoy_rate"].apply(lambda x: _fmt_rate(x, 2))
                    disp["YoY pp"] = disp["yoy_pp_change"].apply(lambda x: "—" if pd.isna(x) else f"{float(x) * 100:+.2f} pp")
                    disp["D&D $ loss"] = disp["Total_DND_$_loss"].apply(_fmt_money)
                    disp["Total units"] = disp["Total_Units"].apply(_fmt_int)
                    disp["DND qty"] = disp["Total_DND_QTY"].apply(_fmt_int)
                    disp["Volume"] = disp["Volume"].apply(_fmt_int)

                    show_cols = ["FSCL_WK_START_DATE", "Volume", "Total_DND_QTY", "Total_Units", "D&D rate", "YoY rate", "YoY pp", "D&D $ loss"]
                    st.dataframe(disp[show_cols].sort_values("FSCL_WK_START_DATE", ascending=False), use_container_width=True, hide_index=True)

                csv = depot_hist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download depot trend CSV",
                    data=csv,
                    file_name=f"depot_{depot_pick}_dnd_trend.csv",
                    mime="text/csv",
                )

st.caption(
    f"Data sources: `{COMPANY_TABLE}`, `{DEPOT_CURRENT_TABLE}`, `{DEPOT_HIST_TABLE}` (Delta). "
    "YoY is computed by aligning week_start_date − 364 days (52 weeks)."
)
