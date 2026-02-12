# 📈 Forecasting & D&D Analytics Assistant  
**Production-Ready Proof of Concept (POC)**

---

## Executive Summary
This application is a **buyer-facing forecasting analytics platform** that consolidates weekly demand forecasts, model performance, and Dump & Destroy (D&D) monitoring into a single, easy-to-use interface. It also includes a guided analytics assistant that answers questions using **deterministic, auditable calculations** sourced directly from Databricks tables.

The solution is intentionally designed as a **production-quality POC**: clean UI, strong guardrails, low operational cost, and clear separation between modeling and analytics.

---

## Business Value

**What this enables immediately**
- Faster, clearer decisions for buyers
- Reduced manual spreadsheet work
- Transparent view of forecast accuracy and bias
- Early visibility into D&D risk and financial impact
- Consistent interpretation of model outputs across teams

**What this does NOT do**
- ❌ It does not generate forecasts
- ❌ It does not modify model logic
- ❌ It does not invent or approximate numbers using AI

All numbers come from **existing Databricks pipelines** and stored Delta tables.

---

## What’s Included

### 1️⃣ Weekly Run Dashboard
- Snapshot of the latest forecast run
- Totals by item and forecast horizon
- Clear view of what buyers are expected to order

### 2️⃣ Forecast Explorer
- Drill-down by:
  - Run date
  - Depot
  - Item (Conventional, Organic, Combined)
  - Forecast horizon (1–3 weeks)
- Trend analysis across prior runs
- Exportable tables for offline review

### 3️⃣ Model Performance
- Backtesting results vs. actuals
- Metrics:
  - WAPE
  - Bias
  - Absolute and percent error
- Identification of worst-performing depot-items
- Historical performance trends over time

### 4️⃣ D&D Monitoring
- Company-level D&D trends and YoY comparison
- Depot-level D&D snapshot for the current week
- Ranking depots by:
  - D&D rate
  - D&D quantity
  - D&D financial impact

### 5️⃣ Ask Assistant (Guided Analytics)
- Natural-language questions such as:
  - “What are total forecast units for next week?”
  - “How did the model perform last week?”
  - “Which depots had the highest D&D rate?”
- **Deterministic math only**
- Optional AI intent routing (off by default)
- Every answer is traceable to underlying tables

---

## Architecture (High Level)

```text
Databricks Jobs (Weekly)
   │
   ├── Forecast Outputs
   ├── Backtesting Results
   └── D&D Monitoring Tables
           │
           ▼
Streamlit Analytics App (GCP)
   │
   ├── Dashboards
   ├── Explorers
   ├── Assistant (guard-railed)
   └── Exportable Views

**Key principle:**  
Models generate numbers.  
The application explains, analyzes, and presents them.

---

## Controls & Risk Management

### 🔒 Deterministic by Design
- No AI-generated forecasts
- No AI-generated math
- All calculations are computed from stored tables

### 🔍 Fully Auditable
- SQL can be displayed on demand
- Exact rows used for every answer are visible
- No hidden transformations or approximations

### 💰 Cost-Controlled
- Read-only Databricks SQL access
- Row limits enforced on queries
- AI usage is optional and minimal

### 🧠 AI Safety
- AI (if enabled) is used only for **intent routing**
- No free-form SQL generation
- No write access
- Can be disabled entirely with a single configuration flag

---

## Data Scope
- ~7 years of historical sales
- All weekly forecasts retained
- All backtesting results retained
- Full D&D history at company and depot levels

This enables:
- Long-term trend analysis
- Run-to-run comparison
- Post-mortem reviews and retrospectives

---

## Deployment Status
- Designed for **GCP Cloud Run**
- Compatible with Databricks SQL Warehouses
- Secure via environment variables
- Production-ready folder and service structure

---

## What This POC Proves
1. Forecast outputs can be operationalized without changing models  
2. Buyers can self-serve analytics safely and confidently  
3. AI can be used responsibly without introducing risk  
4. Forecasting, performance, and D&D can live in one unified system

