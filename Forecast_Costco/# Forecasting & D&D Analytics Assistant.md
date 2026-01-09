aa# Forecasting & D&D Analytics Assistant  
**Production-Ready Streamlit POC — Startup & Operations Guide**

---

## Overview

This Streamlit application provides a **buyer-facing interface** for:

- Entering **DEPOT × ITEM × WEEK (1–3)** prices  
- Reviewing **weekly forecasts**  
- Evaluating **model performance**  
- Monitoring **Dump & Destroy (D&D)** metrics  
- Querying data through a **deterministic analytics assistant**

The app is **offline-safe by design** and can run with or without Databricks and Google Sheets configured.

---

## 1. Environment Setup

Create and activate a Python virtual environment from the project root:

python -m venv .venv

Activate Python Virtual Environment (Windows):

.venv\Scripts\activate # Windows

Activate Python Virtual Environment (macOS / Linux)
source .venv/bin/activate # macOS / Linux

You should see `(.venv)` in your terminal once activated.

---

## 2. Install Dependencies

Install all required libraries:

pip install -r requirements.txt

This installs Streamlit, the Databricks SQL connector, Google Sheets libraries, and all supporting utilities required for the POC.

---

## 3. Choose Your Run Mode


The application supports **two run modes** without any code changes.

---

### Option A — Offline UI Mode (Recommended First Run)

This is the fastest way to verify the UI renders correctly.

DBX_SQL_OFFLINE_OK=true

**Behavior**
- Full UI renders
- Tables appear empty
- Clear banners indicate offline mode
- No Databricks or Google credentials required

---

### Option B — Online Mode (Databricks SQL Connected)

Enable this mode to load real forecasts, backtests, and D&D data.

Set the following environment variables:

DBX_SQL_HOST=adb-xxxx.azuredatabricks.net
DBX_SQL_HTTP_PATH=/sql/1.0/warehouses/xxxxxxxxxxxxxxxx
DBX_SQL_TOKEN=dapi-xxxxxxxx
DBX_SQL_CATALOG=ppe_dscoe
DBX_SQL_SCHEMA=default

**Behavior**
- Live Databricks data loads
- Forecast, performance, and D&D pages populate
- All queries are parameterized, read-only, and row-limited

---

## 4. Start the Streamlit App

Launch the application:

streamlit run app.py

The UI will be available at:

http://localhost:8501

---

## 5. Google Sheets Price Read & Write (Optional)

The **Price Input** page allows buyers to enter **DEPOT × ITEM × WEEK (1–3)** prices.

To enable price writes, set:

GOOGLE_SHEETS_PRICE_URL=<full_google_sheet_url>

And **one** of the following authentication options:

GOOGLE_SERVICE_ACCOUNT_JSON=<service_account_json_string>

or

GOOGLE_SERVICE_ACCOUNT_FILE=/absolute/path/to/service_account.json

**If not configured**
- The Price Input page still renders
- Write actions are disabled
- The UI displays a clear status banner

---

## 6. Price Fallback & Resilience

- Buyer-entered prices are written to Google Sheets
- Databricks parses the sheet during scheduled runs
- A **Delta fallback table** stores the most recent valid prices
- If the Google Sheet is unavailable, prices automatically revert to the last known values

This guarantees **pricing continuity with zero downtime**.

---

## 7. Assistant LLM Routing (Optional)

The assistant runs in **deterministic-only mode by default**.

To enable OpenAI-based routing:

USE_LLM_ROUTER=true
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_MODEL=gpt-4.1-mini

If not set, the assistant uses deterministic SQL + math only (no external calls).

---

## 8. What “Production-Ready POC” Means

- Offline-safe UI
- Strictly read-only Databricks access
- Centralized query and validation layers
- Buyer-facing inputs with controlled write paths
- Clear status banners and guardrails
- Clean upgrade path from POC → production

---

## 9. Recommended First-Time Flow

1. Run in **Offline UI Mode** to validate layout and navigation  
2. Connect **Databricks SQL** for live analytics  
3. Enable **Google Sheets** for buyer price entry  
4. (Optional) Enable **LLM routing** for enhanced assistant behavior  

---
