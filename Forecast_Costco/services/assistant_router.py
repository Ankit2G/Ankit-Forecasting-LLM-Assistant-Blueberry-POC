# services/assistant_router.py
# ------------------------------------------------------------
# Assistant Router (deterministic-first, production-ready POC)
# - Safe, testable routing + execution layer for Streamlit pages
# - No free-form SQL execution; only whitelisted tools with parameterized queries
# - Optional LLM router (OpenAI) for intent classification (OFF by default)
#
# Usage:
#   from services.assistant_router import build_default_router, RouterContext
#   router = build_default_router(query_fn=run_query, tables=RouterTables(...))
#   result = router.handle("What are total forecast units for week 1?", ctx)
#
# This module expects a callable:
#   query_fn(sql: str, params: tuple|None) -> list[dict]
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# -----------------------------
# Types
# -----------------------------
QueryFn = Callable[[str, Optional[Tuple[Any, ...]]], List[Dict[str, Any]]]


@dataclass(frozen=True)
class RouterTables:
    forecast: str
    backtest: str
    actuals: str
    dnd_company: str
    dnd_depot_current: str
    dnd_depot_hist: str


@dataclass(frozen=True)
class RouterConfig:
    """
    Configuration for routing and safety limits.
    """
    max_rows: int = 5000
    default_rows_limit: int = 2000

    # Optional OpenAI routing (intent classification). OFF by default.
    enable_llm_router: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    request_timeout_s: int = 15


@dataclass(frozen=True)
class RouterContext:
    """
    Defaults used when user doesn't specify details.
    """
    run_id: str = ""
    item: str = "Combined"
    depot: Optional[int] = None
    horizon: Optional[int] = None
    eval_week_start: str = ""
    dnd_week_start: str = ""


@dataclass
class ToolResult:
    title: str
    summary_md: str
    df: Optional[pd.DataFrame] = None
    sql: Optional[str] = None
    params: Optional[Tuple[Any, ...]] = None
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    # If LLM routing is used, you can optionally expose raw router output (for debugging)
    router_raw: Optional[str] = None


# -----------------------------
# Small utilities
# -----------------------------
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_DEPOT_ANY_NUM_RE = re.compile(r"\b(\d{2,6})\b")
_HORIZON_RE = re.compile(r"\b(?:horizon|week)\s*([1-3])\b", re.IGNORECASE)


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
    m = _DATE_RE.search(text or "")
    return _parse_iso_date_or_none(m.group(1)) if m else None


def _extract_number(text: str) -> Optional[int]:
    m = _DEPOT_ANY_NUM_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_horizon(text: str) -> Optional[int]:
    m = _HORIZON_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _contains_any(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    return any(term.lower() in t for term in terms)


def _safe_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


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


def _cap_int(v: Any, default: int, min_v: int, max_v: int) -> int:
    try:
        iv = int(v)
    except Exception:
        iv = int(default)
    if iv < min_v:
        return int(min_v)
    if iv > max_v:
        return int(max_v)
    return int(iv)


# -----------------------------
# Router
# -----------------------------
class AssistantRouter:
    def __init__(self, config: RouterConfig, tables: RouterTables, query_fn: QueryFn):
        self.config = config
        self.tables = tables
        self.query_fn = query_fn

    # -------- Tools (whitelisted, parameterized) --------
    def _tool_forecast_total(
        self,
        run_id: str,
        item: str = "All",
        depot: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> ToolResult:
        where = "WHERE run_id = ?"
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
            FROM {self.tables.forecast}
            {where}
            GROUP BY run_id, ITEM, horizon
            ORDER BY ITEM, horizon
        """
        rows = self.query_fn(sql, tuple(params))
        df = _safe_df(rows)

        if df.empty:
            return ToolResult(
                title="Forecast totals",
                summary_md="No forecast rows found for the selected filters.",
                df=df,
                sql=sql,
                params=tuple(params),
                tool="forecast_total",
                args={"run_id": run_id, "item": item, "depot": depot, "horizon": horizon},
            )

        df["total_forecast_units"] = pd.to_numeric(df.get("total_forecast_units"), errors="coerce")
        total = float(df["total_forecast_units"].sum() or 0.0)

        scope = [f"**Run date:** `{run_id}`"]
        if item and item != "All":
            scope.append(f"**Item:** `{item}`")
        if depot is not None:
            scope.append(f"**Depot:** `{int(depot)}`")
        if horizon is not None:
            scope.append(f"**Horizon:** `{int(horizon)}`")

        md = f"{' • '.join(scope)}\n\n**Total forecast units:** **{_fmt_int(total)}**\n\nBreakdown is shown below."
        return ToolResult(
            title="Forecast totals",
            summary_md=md,
            df=df,
            sql=sql,
            params=tuple(params),
            tool="forecast_total",
            args={"run_id": run_id, "item": item, "depot": depot, "horizon": horizon},
        )

    def _tool_forecast_rows(
        self,
        run_id: str,
        item: str = "All",
        depot: Optional[int] = None,
        horizon: Optional[int] = None,
        limit: int = 2000,
    ) -> ToolResult:
        limit = _cap_int(limit, self.config.default_rows_limit, 1, self.config.max_rows)

        where = "WHERE run_id = ?"
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
        df = _safe_df(rows)
        if not df.empty:
            if "DEPOT" in df.columns:
                df["DEPOT"] = pd.to_numeric(df["DEPOT"], errors="coerce").astype("Int64")
            if "horizon" in df.columns:
                df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
            if "yhat" in df.columns:
                df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")

        md = f"Showing up to **{limit:,}** forecast rows for **run `{run_id}`**."
        return ToolResult(
            title="Forecast rows",
            summary_md=md,
            df=df,
            sql=sql,
            params=tuple(params),
            tool="forecast_rows",
            args={"run_id": run_id, "item": item, "depot": depot, "horizon": horizon, "limit": limit},
        )

    def _tool_backtest_summary(
        self,
        eval_week_start: str,
        item: str = "All",
        depot: Optional[int] = None,
    ) -> ToolResult:
        where = "WHERE eval_week_start = ?"
        params: List[Any] = [eval_week_start]

        if item and item != "All":
            where += " AND ITEM = ?"
            params.append(item)
        if depot is not None:
            where += " AND DEPOT = ?"
            params.append(int(depot))

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
        df = _safe_df(rows)

        if df.empty or df.iloc[0].isna().all():
            return ToolResult(
                title="Model performance summary",
                summary_md="No backtest rows found for the selected filters.",
                df=None,
                sql=sql,
                params=tuple(params),
                tool="backtest_summary",
                args={"eval_week_start": eval_week_start, "item": item, "depot": depot},
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
            scope.append(f"**Depot:** `{int(depot)}`")

        md = (
            f"{' • '.join(scope)}\n\n"
            f"- **Actual units (sum):** {_fmt_int(sum_act)}\n"
            f"- **Forecast units (sum):** {_fmt_int(sum_fc)}\n"
            f"- **WAPE:** {_fmt_rate(wape, 1)}\n"
            f"- **Bias:** {_fmt_rate(bias, 1)}\n"
            f"- **Avg |% error|:** {_fmt_rate(r.get('avg_abs_pct_error'), 1)}\n"
        )

        return ToolResult(
            title="Model performance summary",
            summary_md=md,
            df=df,
            sql=sql,
            params=tuple(params),
            tool="backtest_summary",
            args={"eval_week_start": eval_week_start, "item": item, "depot": depot},
        )

    def _tool_dnd_company_latest(self, weeks_back: int = 52) -> ToolResult:
        weeks_back = _cap_int(weeks_back, 52, 1, 260)

        sql = f"""
            SELECT
                CAST(FSCL_WK_START_DATE AS STRING) AS FSCL_WK_START_DATE,
                Total_DND_QTY,
                Total_Units,
                Total_DND_$_loss,
                (SUM(Total_DND_QTY) / NULLIF(SUM(Total_Units), 0)) AS dnd_rate
            FROM {self.tables.dnd_company}
            GROUP BY FSCL_WK_START_DATE, Total_DND_QTY, Total_Units, Total_DND_$_loss
            ORDER BY FSCL_WK_START_DATE DESC
            LIMIT {weeks_back}
        """
        rows = self.query_fn(sql, None)
        df = _safe_df(rows)

        if df.empty:
            return ToolResult(
                title="Company D&D",
                summary_md="No company D&D rows found.",
                df=df,
                sql=sql,
                params=None,
                tool="dnd_company_latest",
                args={"weeks_back": weeks_back},
            )

        df["dnd_rate"] = pd.to_numeric(df.get("dnd_rate"), errors="coerce")
        df["Total_DND_$_loss"] = pd.to_numeric(df.get("Total_DND_$_loss"), errors="coerce")

        latest = df.iloc[0]
        md = (
            f"**Latest company D&D week:** `{latest.get('FSCL_WK_START_DATE')}`\n\n"
            f"- **D&D rate:** **{_fmt_rate(latest.get('dnd_rate'), 2)}**\n"
            f"- **D&D $ loss:** **{_fmt_money(latest.get('Total_DND_$_loss'))}**\n"
            f"- **DND qty:** {_fmt_int(latest.get('Total_DND_QTY'))}\n"
            f"- **Total units:** {_fmt_int(latest.get('Total_Units'))}\n"
        )

        return ToolResult(
            title="Company D&D (latest)",
            summary_md=md,
            df=df,
            sql=sql,
            params=None,
            tool="dnd_company_latest",
            args={"weeks_back": weeks_back},
        )

    def _tool_dnd_depot_snapshot(
        self,
        week_start: Optional[str] = None,
        rank_by: str = "rate",
        top_n: int = 10,
    ) -> ToolResult:
        top_n = _cap_int(top_n, 10, 1, 50)
        rank_by = (rank_by or "rate").strip().lower()
        if rank_by not in {"rate", "loss", "qty"}:
            rank_by = "rate"

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
                Total_DND_$_loss,
                (SUM(Total_DND_QTY) / NULLIF(SUM(Total_Units), 0)) AS dnd_rate
            FROM {self.tables.dnd_depot_current}
            {where}
            GROUP BY DEPOT, FSCL_WK_START_DATE, Volume, Total_DND_QTY, Total_Units, Total_DND_$_loss
        """
        rows = self.query_fn(sql, tuple(params) if params else None)
        df = _safe_df(rows)

        if df.empty:
            return ToolResult(
                title="Depot D&D snapshot",
                summary_md="No depot snapshot rows found.",
                df=df,
                sql=sql,
                params=tuple(params) if params else None,
                tool="dnd_depot_snapshot",
                args={"week_start": week_start, "rank_by": rank_by, "top_n": top_n},
            )

        for c in ["DEPOT", "Volume", "Total_DND_QTY", "Total_Units", "Total_DND_$_loss", "dnd_rate"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if rank_by == "loss":
            df = df.sort_values("Total_DND_$_loss", ascending=False)
            label = "D&D $ loss"
        elif rank_by == "qty":
            df = df.sort_values("Total_DND_QTY", ascending=False)
            label = "DND qty"
        else:
            df = df.sort_values("dnd_rate", ascending=False)
            label = "D&D rate"

        df_top = df.head(top_n).copy()
        md = f"Top **{top_n}** depots by **{label}**" + (f" for week `{week_start}`." if week_start else ".")
        return ToolResult(
            title="Depot D&D snapshot",
            summary_md=md,
            df=df_top,
            sql=sql,
            params=tuple(params) if params else None,
            tool="dnd_depot_snapshot",
            args={"week_start": week_start, "rank_by": rank_by, "top_n": top_n},
        )

    # -------- Routing --------
    def heuristic_route(self, user_text: str, ctx: RouterContext) -> Tuple[str, Dict[str, Any]]:
        text = (user_text or "").strip()
        t = text.lower()

        date_in_text = _extract_date(text)

        # D&D intent
        if _contains_any(t, ["d&d", "dnd", "dump", "d and d"]):
            if _contains_any(t, ["top", "worst", "rank", "highest"]):
                if _contains_any(t, ["loss", "$"]):
                    return "dnd_depot_snapshot", {"week_start": date_in_text or ctx.dnd_week_start or None, "rank_by": "loss", "top_n": 10}
                if _contains_any(t, ["qty", "quantity", "units"]):
                    return "dnd_depot_snapshot", {"week_start": date_in_text or ctx.dnd_week_start or None, "rank_by": "qty", "top_n": 10}
                return "dnd_depot_snapshot", {"week_start": date_in_text or ctx.dnd_week_start or None, "rank_by": "rate", "top_n": 10}
            return "dnd_company_latest", {"weeks_back": 52}

        # Performance intent
        if _contains_any(t, ["backtest", "performance", "error", "wape", "mape", "bias", "accuracy"]):
            return "backtest_summary", {
                "eval_week_start": date_in_text or ctx.eval_week_start,
                "item": ctx.item if ctx.item else "All",
                "depot": ctx.depot,
            }

        # Forecast intent
        if _contains_any(t, ["forecast", "yhat", "predict", "prediction", "demand"]):
            # detail rows
            if _contains_any(t, ["show", "list", "table", "rows", "detail"]):
                h = _extract_horizon(t) or ctx.horizon
                return "forecast_rows", {
                    "run_id": ctx.run_id,
                    "item": ctx.item if ctx.item else "All",
                    "depot": ctx.depot,
                    "horizon": h,
                    "limit": self.config.default_rows_limit,
                }

            # totals
            h = _extract_horizon(t) or ctx.horizon
            return "forecast_total", {
                "run_id": ctx.run_id,
                "item": ctx.item if ctx.item else "All",
                "depot": ctx.depot,
                "horizon": h,
            }

        # Default: forecast totals
        return "forecast_total", {"run_id": ctx.run_id, "item": ctx.item or "All", "depot": ctx.depot, "horizon": ctx.horizon}

    def llm_route(self, user_text: str, ctx: RouterContext) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """
        Optional OpenAI router using Responses API.
        Returns (tool, args, raw_text). If unavailable, returns ("", {}, None).
        """
        if not (self.config.enable_llm_router and self.config.openai_api_key):
            return "", {}, None

        # Keep it optional; if requests isn't installed, fall back.
        try:
            import requests  # type: ignore
        except Exception:
            return "", {}, None

        system = (
            "You are a routing assistant for a forecasting analytics app. "
            "Choose exactly one tool and arguments in strict JSON.\n\n"
            "TOOLS:\n"
            "1) forecast_total(run_id, item, depot, horizon)\n"
            "2) forecast_rows(run_id, item, depot, horizon, limit)\n"
            "3) backtest_summary(eval_week_start, item, depot)\n"
            "4) dnd_company_latest(weeks_back)\n"
            "5) dnd_depot_snapshot(week_start, rank_by['rate'|'loss'|'qty'], top_n)\n\n"
            "RULES:\n"
            "- Output ONLY JSON with keys: tool, args\n"
            "- Do not invent dates. If missing, use ctx defaults.\n"
            "- Keep forecast_rows.limit <= 5000.\n"
            "- Keep dnd_depot_snapshot.top_n <= 50.\n"
        )

        payload = {
            "model": self.config.openai_model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"question": user_text, "ctx": ctx.__dict__})},
            ],
            "temperature": 0,
        }

        try:
            resp = requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.config.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.config.request_timeout_s,
            )
            if resp.status_code >= 300:
                return "", {}, None

            data = resp.json()

            raw = ""
            try:
                for o in data.get("output", []):
                    for c in o.get("content", []):
                        if c.get("type") == "output_text":
                            raw += c.get("text", "")
            except Exception:
                raw = ""
            raw = (raw or "").strip()
            if not raw:
                return "", {}, None

            j = json.loads(raw)
            tool = str(j.get("tool", "") or "").strip()
            args = j.get("args", {}) or {}

            # Enforce caps
            if tool == "forecast_rows":
                args["limit"] = _cap_int(args.get("limit", self.config.default_rows_limit), self.config.default_rows_limit, 1, self.config.max_rows)
            if tool == "dnd_depot_snapshot":
                args["top_n"] = _cap_int(args.get("top_n", 10), 10, 1, 50)

            return tool, args, raw
        except Exception:
            return "", {}, None

    # -------- Execution --------
    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        if tool_name == "forecast_total":
            return self._tool_forecast_total(**args)
        if tool_name == "forecast_rows":
            return self._tool_forecast_rows(**args)
        if tool_name == "backtest_summary":
            return self._tool_backtest_summary(**args)
        if tool_name == "dnd_company_latest":
            return self._tool_dnd_company_latest(**args)
        if tool_name == "dnd_depot_snapshot":
            return self._tool_dnd_depot_snapshot(**args)

        return ToolResult(
            title="Unsupported request",
            summary_md="I couldn’t map that request to a supported analytics action yet. Try asking about forecasts, performance, or D&D.",
            df=None,
            tool=tool_name,
            args=args,
        )

    def handle(self, user_text: str, ctx: RouterContext) -> ToolResult:
        """
        Main entrypoint: route + execute with safety and normalization.
        """
        text = user_text or ""

        # Light normalization: if user says "depot 172" capture depot (only if ctx doesn't already set depot)
        if ctx.depot is None and re.search(r"\bdepot\b", text.lower()):
            depot_in_text = _extract_number(text)
            if depot_in_text is not None:
                ctx = RouterContext(**{**ctx.__dict__, "depot": depot_in_text})

        # Try LLM routing first (optional)
        tool, args, raw = self.llm_route(text, ctx)
        if not tool:
            tool, args = self.heuristic_route(text, ctx)
            raw = None

        # Fill missing args from ctx defaults
        if tool.startswith("forecast_"):
            args.setdefault("run_id", ctx.run_id)
            args.setdefault("item", ctx.item or "All")
            args.setdefault("depot", ctx.depot)
            # If user explicitly typed horizon/week, prefer it
            args.setdefault("horizon", _extract_horizon(text) or ctx.horizon)

        if tool == "backtest_summary":
            args.setdefault("eval_week_start", ctx.eval_week_start or _extract_date(text) or "")
            args.setdefault("item", ctx.item or "All")
            args.setdefault("depot", ctx.depot)

        if tool == "dnd_depot_snapshot":
            args.setdefault("week_start", _extract_date(text) or ctx.dnd_week_start or None)

        # Basic required checks
        if tool in {"forecast_total", "forecast_rows"} and not args.get("run_id"):
            return ToolResult(
                title="Missing run date",
                summary_md="Please provide a forecast **run date** (Monday) in `YYYY-MM-DD` format (or set a default run_id).",
                df=None,
                tool=tool,
                args=args,
                router_raw=raw,
            )

        if tool == "backtest_summary" and not args.get("eval_week_start"):
            return ToolResult(
                title="Missing evaluation week",
                summary_md="Please provide an **eval_week_start** in `YYYY-MM-DD` format (or set a default eval_week_start).",
                df=None,
                tool=tool,
                args=args,
                router_raw=raw,
            )

        result = self.execute(tool, args)
        result.router_raw = raw
        return result


# -----------------------------
# Convenience factory
# -----------------------------
def build_default_router(query_fn: QueryFn, tables: RouterTables) -> AssistantRouter:
    cfg = RouterConfig(
        max_rows=_cap_int(os.getenv("ASSISTANT_MAX_ROWS", "5000"), 5000, 1, 200000),
        default_rows_limit=_cap_int(os.getenv("ASSISTANT_DEFAULT_LIMIT", "2000"), 2000, 1, 200000),
        enable_llm_router=str(os.getenv("USE_LLM_ROUTER", "false")).strip().lower() in {"1", "true", "yes", "y"},
        openai_api_key=(os.getenv("OPENAI_API_KEY", "") or "").strip(),
        openai_model=(os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip(),
        request_timeout_s=_cap_int(os.getenv("OPENAI_TIMEOUT_S", "15"), 15, 1, 120),
    )
    return AssistantRouter(cfg, tables, query_fn)
