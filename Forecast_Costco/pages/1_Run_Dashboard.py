# pages/1_Run_Dashboard.py
# ------------------------------------------------------------
# Run Dashboard (Buyer-Facing, Production-Grade POC)
#
# What’s improved vs. the minimal version:
# ✅ Shows actual forecast week start dates (Week 1/2/3)
# ✅ Tabs: Snapshot / Depot Breakdown / Item Breakdown / Export
# ✅ Charts: horizon totals + top depots
# ✅ Strong empty/offline behavior (renders even with no DB connection)
# ✅ “Buyer friendly” language + clear guardrails
#
# Depends on QueryService methods (from services/queries.py):
# - forecast_run_ids(limit: int)
# - forecast_totals(run_id: str, item: str)
# - forecast_detail(run_id: str, item: str, horizon: int|str)
# ------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import altair as alt

from services.validators import validate_run_id, validate_item
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


def _empty_df(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def _safe_df(obj) -> pd.DataFrame:
    """QueryService returns objects with `.df` in this POC. Fall back safely."""
    if obj is None:
        return pd.DataFrame()
    df = getattr(obj, "df", None)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


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
st.set_page_config(page_title="Run Dashboard", layout="wide")

st.header("📊 Run Dashboard")
st.caption("Weekly forecast snapshot for buyers (15 depots × 2 items × 3-week horizon).")

qs = _get_qs()

# ----------------------------
# Controls
# ----------------------------
top_left, top_mid, top_right = st.columns([2, 2, 6])

with top_left:
    runs_df = _safe_df(qs.forecast_run_ids(limit=30))
    if runs_df.empty or "run_id" not in runs_df.columns:
        st.warning("No forecast runs available (offline mode or empty table).")
        selected_run = None
    else:
        selected_run = st.selectbox(
            "Run date (Monday)",
            options=runs_df["run_id"].tolist(),
            index=0,
            help="Select a weekly run date to view the snapshot.",
        )

with top_mid:
    item = st.selectbox(
        "Item filter",
        options=["All", "Conventional", "Organic", "Combined"],
        index=3,
        help="Combined is stored as a first-class item from the model output.",
    )

with top_right:
    st.markdown(
        """
**What this page answers**
- What are the Week 1 / Week 2 / Week 3 forecast totals?
- What are the actual week start dates for the forecast horizon?
- Which depots drive the most volume this run?

**Guardrails**
- Forecast values come directly from Databricks model outputs
- No AI-generated forecast numbers
"""
    )

# Validate inputs
run_id, run_err = validate_run_id(selected_run, allow_none=True)
if run_err:
    st.error(run_err)
    st.stop()

item, item_err = validate_item(item)
if item_err:
    st.error(item_err)
    st.stop()

if not run_id:
    st.info("Select a run date to view totals.")
    st.stop()

# ----------------------------
# Load core data (totals + detail for dates)
# ----------------------------
totals_df = _safe_df(qs.forecast_totals(run_id=run_id, item=item))
if totals_df.empty:
    totals_df = _empty_df(["ITEM", "horizon", "total_forecast_units"])

# We fetch detail (Combined preferred) to compute actual horizon week start dates.
# If Combined isn't available, fall back to whatever item filter is selected.
detail_item_for_dates = "Combined" if item in ("All", "Combined") else item
detail_df = _safe_df(qs.forecast_detail(run_id=run_id, item=detail_item_for_dates, horizon="All"))
if detail_df.empty:
    detail_df = _empty_df(["DEPOT", "ITEM", "ds", "horizon", "yhat"])

# Compute horizon dates (Week 1/2/3) from detail ds values
horizon_dates = []
if "ds" in detail_df.columns and not detail_df.empty:
    horizon_dates = sorted(pd.to_datetime(detail_df["ds"]).dropna().unique())[:3]
horizon_date_labels = {i + 1: (_as_date_str(d) if i < len(horizon_dates) else "—") for i, d in enumerate(horizon_dates)}
# Ensure keys exist
for k in (1, 2, 3):
    horizon_date_labels.setdefault(k, "—")

# Normalize totals df expected columns
if "horizon" in totals_df.columns:
    try:
        totals_df["horizon"] = totals_df["horizon"].astype(int)
    except Exception:
        pass

# ----------------------------
# KPI Row
# ----------------------------
st.subheader("Snapshot")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns([1.2, 1.2, 1.2, 1.4, 3.0])

def _total_for_h(h: int):
    if totals_df.empty or "horizon" not in totals_df.columns:
        return None
    row = totals_df[totals_df["horizon"] == h]
    if row.empty or "total_forecast_units" not in row.columns:
        return None
    return float(row["total_forecast_units"].iloc[0])

w1 = _total_for_h(1)
w2 = _total_for_h(2)
w3 = _total_for_h(3)

with kpi1:
    st.metric(f"Week 1 ({horizon_date_labels[1]})", _format_int(w1) if w1 is not None else "—")
with kpi2:
    st.metric(f"Week 2 ({horizon_date_labels[2]})", _format_int(w2) if w2 is not None else "—")
with kpi3:
    st.metric(f"Week 3 ({horizon_date_labels[3]})", _format_int(w3) if w3 is not None else "—")
with kpi4:
    total_3w = None
    try:
        if "total_forecast_units" in totals_df.columns:
            total_3w = float(totals_df["total_forecast_units"].sum())
    except Exception:
        total_3w = None
    st.metric("3-Week Total", _format_int(total_3w) if total_3w is not None else "—")
with kpi5:
    st.info(
        f"**Selected run:** {run_id}\n\n"
        f"**Item filter:** {item}\n\n"
        f"**Horizon weeks:** {horizon_date_labels[1]}, {horizon_date_labels[2]}, {horizon_date_labels[3]}"
    )

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Snapshot", "🏬 Depot Breakdown", "🧺 Item Breakdown", "⬇️ Export"])

# ----------------------------
# Tab 1: Snapshot charts + table
# ----------------------------
with tab1:
    left, right = st.columns([1.2, 1.0], gap="large")

    # Horizon totals chart
    with left:
        st.markdown("#### Totals by Horizon")
        if totals_df.empty or "horizon" not in totals_df.columns:
            st.info("No totals available for this selection.")
        else:
            chart_df = totals_df.copy()
            # Friendly labels
            chart_df["Horizon"] = chart_df["horizon"].map(
                {
                    1: f"Week 1 ({horizon_date_labels[1]})",
                    2: f"Week 2 ({horizon_date_labels[2]})",
                    3: f"Week 3 ({horizon_date_labels[3]})",
                }
            )
            chart_df = chart_df.rename(columns={"total_forecast_units": "Forecast Units"})

            c = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("Horizon:N", sort=None, title=None),
                    y=alt.Y("Forecast Units:Q", title="Units"),
                    tooltip=["Horizon:N", alt.Tooltip("Forecast Units:Q", format=",.0f")],
                )
                .properties(height=260)
            )
            st.altair_chart(c, use_container_width=True)

    # Totals table
    with right:
        st.markdown("#### Totals Table")
        if totals_df.empty:
            st.info("No data available.")
        else:
            display_df = (
                totals_df.copy()
                .sort_values("horizon")
                .assign(
                    Horizon=lambda d: d["horizon"].map(
                        {
                            1: f"Week 1 ({horizon_date_labels[1]})",
                            2: f"Week 2 ({horizon_date_labels[2]})",
                            3: f"Week 3 ({horizon_date_labels[3]})",
                        }
                    )
                )
                .rename(columns={"total_forecast_units": "Forecast Units"})
            )
            show_cols = [c for c in ["ITEM", "Horizon", "Forecast Units"] if c in display_df.columns]
            st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("#### Quick Notes")
    st.write(
        "- Use **Forecast Explorer** for depot-item-week detail.\n"
        "- Use **Model Performance** for last week’s error metrics.\n"
        "- Use **Price Input** to update Week 1–3 prices in the Google Sheet that the model parses."
    )

# ----------------------------
# Tab 2: Depot Breakdown
# ----------------------------
with tab2:
    st.markdown("#### Top Depots by Forecast Units (3-week total)")

    if detail_df.empty or "DEPOT" not in detail_df.columns or "yhat" not in detail_df.columns:
        st.info("No detail rows available to build depot breakdown.")
    else:
        dep = (
            detail_df.groupby("DEPOT", as_index=False)["yhat"]
            .sum()
            .rename(columns={"yhat": "Total Forecast Units"})
            .sort_values("Total Forecast Units", ascending=False)
        )

        top_n = st.slider("Show top N depots", min_value=5, max_value=15, value=10, step=1)
        dep_top = dep.head(top_n)

        bar = (
            alt.Chart(dep_top)
            .mark_bar()
            .encode(
                x=alt.X("Total Forecast Units:Q", title="Units"),
                y=alt.Y("DEPOT:N", sort="-x", title="Depot"),
                tooltip=["DEPOT:N", alt.Tooltip("Total Forecast Units:Q", format=",.0f")],
            )
            .properties(height=360)
        )
        st.altair_chart(bar, use_container_width=True)

        st.dataframe(dep, use_container_width=True, hide_index=True)

# ----------------------------
# Tab 3: Item Breakdown
# ----------------------------
with tab3:
    st.markdown("#### Item Mix (3-week total)")

    if detail_df.empty or "ITEM" not in detail_df.columns or "yhat" not in detail_df.columns:
        st.info("No detail rows available to build item mix.")
    else:
        item_mix = (
            detail_df.groupby("ITEM", as_index=False)["yhat"]
            .sum()
            .rename(columns={"yhat": "Total Forecast Units"})
            .sort_values("Total Forecast Units", ascending=False)
        )

        pie = (
            alt.Chart(item_mix)
            .mark_arc()
            .encode(
                theta=alt.Theta("Total Forecast Units:Q"),
                color=alt.Color("ITEM:N"),
                tooltip=["ITEM:N", alt.Tooltip("Total Forecast Units:Q", format=",.0f")],
            )
            .properties(height=280)
        )
        st.altair_chart(pie, use_container_width=True)

        st.dataframe(item_mix, use_container_width=True, hide_index=True)

# ----------------------------
# Tab 4: Export
# ----------------------------
with tab4:
    st.markdown("#### Export this run")

    # Export totals
    if not totals_df.empty:
        export_totals = totals_df.copy()
        export_totals_csv = export_totals.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Totals CSV",
            data=export_totals_csv,
            file_name=f"run_dashboard_totals_{run_id}.csv",
            mime="text/csv",
        )
    else:
        st.info("No totals available to export.")

    # Export detail sample (not always needed, but helpful)
    if not detail_df.empty:
        export_detail = detail_df.copy()
        export_detail_csv = export_detail.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Detail CSV",
            data=export_detail_csv,
            file_name=f"run_dashboard_detail_{run_id}.csv",
            mime="text/csv",
        )
    else:
        st.info("No detail available to export.")

# ----------------------------
# Footer guidance
# ----------------------------
with st.expander("How to use this page"):
    st.markdown(
        """
- **Run date**: choose the weekly Monday run you want to review.
- **Item filter**: choose Combined for an overall view, or narrow to Conventional/Organic.
- **Week labels** show the *actual week start dates* for the forecast horizon.
- Use **Depot Breakdown** to quickly see what drives volume this run.
"""
    )
