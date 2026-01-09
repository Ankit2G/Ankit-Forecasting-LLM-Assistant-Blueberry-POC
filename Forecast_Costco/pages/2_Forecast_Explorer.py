# pages/2_Forecast_Explorer.py
# ------------------------------------------------------------
# Forecast Explorer (Buyer-Facing, Production-Grade POC)
#
# Upgrades vs. minimal version:
# ✅ Shows actual horizon week start dates (Week 1/2/3)
# ✅ Filters: run, item, horizon, depot (multi-select), search
# ✅ Tabs: Detail / Depot Summary / Trend (by horizon) / Export
# ✅ Charts: horizon totals, depot totals, horizon trend lines
# ✅ Strong empty/offline behavior
#
# Depends on QueryService methods (services/queries.py):
# - forecast_run_ids(limit: int)
# - forecast_detail(run_id: str, item: str, horizon: int|str, depots: list[int]|None = None)
#
# If your QueryService.forecast_detail signature doesn't support `depots`,
# this page will still work by filtering in pandas after fetching.
# ------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import altair as alt

from services.validators import (
    validate_run_id,
    validate_item,
    validate_horizon,
)


import streamlit as st

tables = st.session_state.get("tables")
qs = st.session_state.get("query_service")

if tables is None or qs is None:
    st.error("App is not initialized. Open the Home page (app.py) first, then return here.")
    st.stop()

# ----------------------------
# Helpers
# ----------------------------
def _get_qs():
    if "query_service" not in st.session_state:
        st.error("Query service not initialized. Please reload the app.")
        st.stop()
    return st.session_state["query_service"]


def _safe_df(obj) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    df = getattr(obj, "df", None)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _empty_df(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def _format_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"


def _as_date_str(v) -> str:
    try:
        return pd.to_datetime(v).date().isoformat()
    except Exception:
        return str(v)


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Forecast Explorer", layout="wide")

st.header("🔎 Forecast Explorer")
st.caption("Explore forecasts by depot, item, and week. Export-ready and buyer-friendly.")

qs = _get_qs()

# ----------------------------
# Controls
# ----------------------------
runs_df = _safe_df(qs.forecast_run_ids(limit=30))
if runs_df.empty or "run_id" not in runs_df.columns:
    st.warning("No forecast runs available (offline mode or empty table).")
    selected_run = None
else:
    selected_run = st.selectbox(
        "Run date (Monday)",
        options=runs_df["run_id"].tolist(),
        index=0,
        help="Select a weekly run date to explore the forecast outputs.",
    )

c1, c2, c3, c4 = st.columns([2, 2, 2, 4])

with c1:
    item = st.selectbox(
        "Item",
        options=["All", "Conventional", "Organic", "Combined"],
        index=3,
        help="Combined is stored as a first-class item from the model output.",
    )

with c2:
    horizon = st.selectbox(
        "Forecast horizon",
        options=["All", 1, 2, 3],
        index=0,
        help="Week number relative to the run date.",
    )

with c3:
    # We'll populate depots after load; for now placeholder.
    depot_mode = st.selectbox("Depot filter", options=["All depots", "Select depots"], index=0)

with c4:
    search_text = st.text_input(
        "Search (optional)",
        value="",
        placeholder="Search depot or item (e.g., 172 or Organic)…",
        help="Filters displayed rows after data loads.",
    )

# Guidance
st.markdown(
    """
**Tips**
- Start with **Combined** and **All horizons** for the full picture.
- Filter to **Week 1** for ordering decisions.
- Use **Depot Summary** to identify where volume is concentrated.
"""
)

# ----------------------------
# Validation
# ----------------------------
run_id, err = validate_run_id(selected_run, allow_none=True)
if err:
    st.error(err)
    st.stop()

item, err = validate_item(item)
if err:
    st.error(err)
    st.stop()

horizon, err = validate_horizon(horizon)
if err:
    st.error(err)
    st.stop()

if not run_id:
    st.info("Select a run date to begin exploring forecasts.")
    st.stop()

# ----------------------------
# Data Load
# ----------------------------
# Try to pass depots to QueryService if supported; otherwise filter after.
depots_selected = None  # populated after we know depot list

try:
    res = qs.forecast_detail(run_id=run_id, item=item, horizon=horizon, depots=None)
except TypeError:
    res = qs.forecast_detail(run_id=run_id, item=item, horizon=horizon)

df = _safe_df(res)
if df.empty:
    df = _empty_df(["DEPOT", "ITEM", "ds", "horizon", "yhat"])

# Normalize types
if not df.empty:
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    if "horizon" in df.columns:
        try:
            df["horizon"] = df["horizon"].astype(int)
        except Exception:
            pass
    if "DEPOT" in df.columns:
        # Keep depot readable in charts/tables
        try:
            df["DEPOT"] = df["DEPOT"].astype(int)
        except Exception:
            pass
    if "yhat" in df.columns:
        df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")

# Horizon week start dates (actual)
horizon_dates = []
if not df.empty and "ds" in df.columns:
    horizon_dates = sorted(df["ds"].dropna().unique())[:3]
horizon_date_labels = {i + 1: (_as_date_str(d) if i < len(horizon_dates) else "—") for i, d in enumerate(horizon_dates)}
for k in (1, 2, 3):
    horizon_date_labels.setdefault(k, "—")

# Depot selector now that we have depots
all_depots = sorted(df["DEPOT"].dropna().unique().tolist()) if (not df.empty and "DEPOT" in df.columns) else []
if depot_mode == "Select depots" and all_depots:
    depots_selected = st.multiselect("Choose depots", options=all_depots, default=all_depots[:3])
elif depot_mode == "Select depots" and not all_depots:
    st.info("No depots available to select (offline mode or empty data).")

# Apply depot filter (pandas-level)
if depots_selected:
    df = df[df["DEPOT"].isin(depots_selected)]

# Apply search filter (pandas-level)
if search_text.strip() and not df.empty:
    s = search_text.strip().lower()
    mask = pd.Series([False] * len(df))
    if "DEPOT" in df.columns:
        mask = mask | df["DEPOT"].astype(str).str.contains(s, na=False)
    if "ITEM" in df.columns:
        mask = mask | df["ITEM"].astype(str).str.lower().str.contains(s, na=False)
    df = df[mask]

# ----------------------------
# Summary KPIs
# ----------------------------
st.subheader("Summary")

k1, k2, k3, k4, k5 = st.columns([1.2, 1.2, 1.5, 1.5, 3.0])

with k1:
    st.metric("Depots", df["DEPOT"].nunique() if not df.empty and "DEPOT" in df.columns else "—")

with k2:
    st.metric("Items", df["ITEM"].nunique() if not df.empty and "ITEM" in df.columns else "—")

with k3:
    total_units = df["yhat"].sum() if not df.empty and "yhat" in df.columns else None
    st.metric("Total Forecast Units", _format_int(total_units) if total_units is not None else "—")

with k4:
    avg_units = df["yhat"].mean() if not df.empty and "yhat" in df.columns else None
    st.metric("Avg / Depot-Week", _format_int(avg_units) if avg_units is not None else "—")

with k5:
    st.info(
        f"**Horizon dates:**\n"
        f"- Week 1: {horizon_date_labels[1]}\n"
        f"- Week 2: {horizon_date_labels[2]}\n"
        f"- Week 3: {horizon_date_labels[3]}"
    )

# ----------------------------
# Tabs
# ----------------------------
tab_detail, tab_depot, tab_trend, tab_export = st.tabs(
    ["📋 Detail", "🏬 Depot Summary", "📈 Trend", "⬇️ Export"]
)

# ----------------------------
# Detail tab
# ----------------------------
with tab_detail:
    st.markdown("#### Forecast Detail (Depot × Item × Week)")

    if df.empty:
        st.info("No forecast data available for this selection.")
    else:
        display_df = df.copy()
        display_df["Week Start Date"] = display_df["ds"].dt.date
        display_df = display_df.rename(columns={"yhat": "Forecast Units"})
        display_df = display_df.sort_values(["DEPOT", "ITEM", "Week Start Date"])

        show_cols = [c for c in ["DEPOT", "ITEM", "Week Start Date", "horizon", "Forecast Units"] if c in display_df.columns]
        st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

        # Horizon totals chart (within current filters)
        if "horizon" in display_df.columns:
            hz = (
                display_df.groupby("horizon", as_index=False)["Forecast Units"]
                .sum()
                .rename(columns={"Forecast Units": "Total Units"})
                .sort_values("horizon")
            )
            hz["Horizon Label"] = hz["horizon"].map(
                {
                    1: f"Week 1 ({horizon_date_labels[1]})",
                    2: f"Week 2 ({horizon_date_labels[2]})",
                    3: f"Week 3 ({horizon_date_labels[3]})",
                }
            )

            st.markdown("#### Totals by Horizon (current filters)")
            chart = (
                alt.Chart(hz)
                .mark_bar()
                .encode(
                    x=alt.X("Horizon Label:N", sort=None, title=None),
                    y=alt.Y("Total Units:Q", title="Units"),
                    tooltip=["Horizon Label:N", alt.Tooltip("Total Units:Q", format=",.0f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

# ----------------------------
# Depot Summary tab
# ----------------------------
with tab_depot:
    st.markdown("#### Depot Totals (3-week or selected horizon)")

    if df.empty or "DEPOT" not in df.columns or "yhat" not in df.columns:
        st.info("No data available to build depot summaries.")
    else:
        depot_totals = (
            df.groupby("DEPOT", as_index=False)["yhat"]
            .sum()
            .rename(columns={"yhat": "Total Forecast Units"})
            .sort_values("Total Forecast Units", ascending=False)
        )

        top_n = st.slider("Show top N depots", min_value=5, max_value=15, value=10, step=1)

        depot_top = depot_totals.head(top_n)
        chart = (
            alt.Chart(depot_top)
            .mark_bar()
            .encode(
                x=alt.X("Total Forecast Units:Q", title="Units"),
                y=alt.Y("DEPOT:N", sort="-x", title="Depot"),
                tooltip=["DEPOT:N", alt.Tooltip("Total Forecast Units:Q", format=",.0f")],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(depot_totals, use_container_width=True, hide_index=True)

# ----------------------------
# Trend tab (by horizon over dates)
# ----------------------------
with tab_trend:
    st.markdown("#### Horizon Trend (Total units by week start date)")

    if df.empty or "ds" not in df.columns or "yhat" not in df.columns:
        st.info("No data available to build trends.")
    else:
        # Total units by ds (and horizon if available)
        trend = df.copy()
        trend["Week Start Date"] = trend["ds"].dt.date

        if "horizon" in trend.columns:
            agg = (
                trend.groupby(["Week Start Date", "horizon"], as_index=False)["yhat"]
                .sum()
                .rename(columns={"yhat": "Total Units"})
                .sort_values(["Week Start Date", "horizon"])
            )
            agg["Horizon"] = agg["horizon"].map({1: "Week 1", 2: "Week 2", 3: "Week 3"})
            line = (
                alt.Chart(agg)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Week Start Date:T", title="Week Start Date"),
                    y=alt.Y("Total Units:Q", title="Units"),
                    color="Horizon:N",
                    tooltip=[
                        "Horizon:N",
                        "Week Start Date:T",
                        alt.Tooltip("Total Units:Q", format=",.0f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(line, use_container_width=True)
        else:
            agg = (
                trend.groupby(["Week Start Date"], as_index=False)["yhat"]
                .sum()
                .rename(columns={"yhat": "Total Units"})
                .sort_values(["Week Start Date"])
            )
            line = (
                alt.Chart(agg)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Week Start Date:T", title="Week Start Date"),
                    y=alt.Y("Total Units:Q", title="Units"),
                    tooltip=["Week Start Date:T", alt.Tooltip("Total Units:Q", format=",.0f")],
                )
                .properties(height=320)
            )
            st.altair_chart(line, use_container_width=True)

# ----------------------------
# Export tab
# ----------------------------
with tab_export:
    st.markdown("#### Export")

    if df.empty:
        st.info("No data available to export.")
    else:
        export_df = df.copy()
        if "ds" in export_df.columns:
            export_df["ds"] = export_df["ds"].dt.date.astype(str)

        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecast Detail CSV",
            data=csv,
            file_name=f"forecast_detail_{run_id}.csv",
            mime="text/csv",
        )

        # Also export depot totals
        if "DEPOT" in df.columns and "yhat" in df.columns:
            depot_totals = (
                df.groupby("DEPOT", as_index=False)["yhat"]
                .sum()
                .rename(columns={"yhat": "Total Forecast Units"})
                .sort_values("Total Forecast Units", ascending=False)
            )
            depot_csv = depot_totals.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Depot Totals CSV",
                data=depot_csv,
                file_name=f"forecast_depot_totals_{run_id}.csv",
                mime="text/csv",
            )

# ----------------------------
# Usage Notes
# ----------------------------
with st.expander("How to use this page"):
    st.markdown(
        """
- Choose a **run date** first.
- Use **Item** and **Horizon** to match ordering needs (Week 1 is most common).
- Use **Depot Summary** to see where volume is concentrated.
- Use **Export** for sharing or further analysis.
"""
    )
