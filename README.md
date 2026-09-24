# Fund Report Parser

A Streamlit-based PDF extraction app that reads a fund report, uses Google Gemini to identify and structure the relevant financial data, maps the results to a Morningstar-style Excel output, and allows the user to download the generated workbook.

---

## 1. Executive Overview

This script is a single-file financial document processing pipeline designed for a very specific task:

- A user enters a Morningstar Document ID; the app fetches the PDF in memory.
- The user enters the target fund name, optional fund ID, and portfolio date in the sidebar.
- The app extracts PDF text and tables locally, selects relevant sections, and sends one compact request with the highly structured extraction prompt to Gemini.
- Gemini returns raw JSON describing:
  - financial position line items,
  - portfolio investments/holdings,
  - total net assets,
  - base currency.
- The script normalizes the extracted data, converts it into a pandas DataFrame, aligns it to a Morningstar 19-column schema, and displays a preview.
- Finally, it exports the result to an Excel file and offers it for download.

In practical terms, this is a document-to-spreadsheet automation tool for transforming unstructured PDF fund literature into a Morningstar-friendly template.

The script is intentionally opinionated and highly specialized:

- It filters to a single target fund and date.
- It standardizes holding IDs and dates.
- It converts financial statement values into a Morningstar-style tabular format.
- It exports a workbook ready for downstream use.

---

## 2. Technical Architecture Summary

This application is built as a single Streamlit page without a formal backend or API server. It follows a straightforward data flow:

1. Collect user inputs and validation checks.
2. Fetch the Morningstar PDF from its Document ID.
3. Extract page text and tables locally with PyMuPDF.
4. Detect text-based versus scanned PDFs and select relevant financial pages.
5. Send only compact local content and, when needed, selected scanned pages to Gemini.
6. Validate the single JSON response.
7. Normalize rows for balance sheet items and holdings.
8. Construct a DataFrame aligned to the Morningstar schema.
9. Render a preview table in Streamlit.
10. Download the Excel workbook as an in-memory file.

### Notable design characteristics

- Uses Streamlit for interface and interaction.
- Uses pandas for table shaping and export.
- Uses the OpenAI Python SDK with Google's Gemini OpenAI-compatible API for PDF ingestion and structured extraction.
- Extracts text and tables locally and renders only selected image-only pages for vision fallback.
- Produces an in-memory Excel workbook without writing to disk.

---

## 3. Script Components and Detailed Breakdown

### 3.1 Imports

The script imports the following modules:

```python
import streamlit as st
import pandas as pd
import io
import json
import base64
import pymupdf
from openai import OpenAI, OpenAIError
```

Purpose of each import:

- `streamlit`: UI framework for interactive app rendering, forms, sidebar, table display, and file download.
- `pandas`: DataFrame creation, date normalization, transformation, and Excel export.
- `io`: in-memory binary buffer for Excel download generation.
- `json`: parsing the Gemini response JSON.
- `base64`: encode selected rendered PDF pages for inline image input.
- `pymupdf`: extract PDF text/tables and render selected pages to PNG images.
- `from openai import OpenAI, OpenAIError`: OpenAI-compatible Gemini client and API error type.

### 3.2 Page configuration and UI shell

```python
st.set_page_config(
    page_title="Fund Report Parser",
    page_icon="📄",
    layout="wide"
)
```

This configures the Streamlit browser page title, icon, and wide layout.

```python
st.title("📄 Fund Report -> Morningstar Template")
st.write("Enter a Morningstar Document ID, specify the target fund details in the sidebar, and extract structured financial data into the Morningstar template.")
```

These calls set the main heading and a top-level descriptive message.

### 3.3 Authentication section

```python
st.sidebar.header("🔑 Gemini authentication")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
st.sidebar.button("Refresh Gemini Models")
selected_model = st.sidebar.selectbox("Gemini extraction model", available_gemini_models)
```

This creates a provider selector and password input for the selected provider. Provider keys are only used in the app session and are not displayed in diagnostics.

Important note: the script checks for `API_KEY` in the app UI, but the error message mentions `.streamlit/secrets.toml` as a legacy or alternate pattern. The actual implementation uses the text input instead of storing secrets locally.

### 3.4 Target fund input fields

The user enters the target metadata in the sidebar:

```python
fund_name = st.sidebar.text_input(
    "Fund Name *",
    value="",
    placeholder="e.g. United ASEAN Discovery Fund",
    help="Enter the exact or partial name of the fund as printed in the report."
)

fund_id = st.sidebar.text_input(
    "Fund ID",
    value="",
    placeholder="e.g. FS0000B37I",
    help="Morningstar Fund ID or internal system code."
)

portfolio_date = st.sidebar.text_input(
    "Portfolio Date (M/D/YYYY) *",
    value="",
    placeholder="e.g. 9/30/2023",
    help="Target reporting date formatted as M/D/YYYY (no leading zeros)."
)
```

#### Parameters

- `fund_name`: required string, used to identify the correct fund in the PDF.
- `fund_id`: optional string used in the final filename and the `Fund Id` column.
- `portfolio_date`: required date string formatted as `M/D/YYYY`.

#### Validation logic

```python
inputs_valid = bool(fund_name.strip() and portfolio_date.strip())
```

This ensures that both the fund name and the portfolio date are non-empty before processing can proceed.

### 3.5 Morningstar schema definition

```python
TEMPLATE_COLUMNS = [
    'Portfolio Date', 'Fund Id', 'Fund Name', 'Holding Id', 'Holding Name', 
    'Number of Share', 'Market Value', 'Coupon Rate', 'Maturity Date', 
    'Portfolio Currency (Base)', 'Local MValue', 'Currency (Local)', 
    'Cost (Base)', 'Country', 'Fund TNA', 'Unnamed: 15', '% TNA', 
    'Unnamed: 17', 'AssetType Reference'
]
```

This list defines the exact output schema expected by the Morningstar-style workbook. The script later ensures the final DataFrame matches these columns and uses the same ordering.

### 3.6 Morningstar document entry point

```python
document_id = st.text_input("Morningstar Document ID", placeholder="e.g. 670134091")
document_url, pdf_bytes = fetch_morningstar_pdf(document_id)
```

The app validates that the ID contains only digits, constructs the configurable Morningstar URL, and keeps the downloaded PDF bytes in memory.

### 3.7 Main processing gate

```python
if st.button("Process Report", type="primary"):
    if not API_KEY:
        st.error("⚠️ API Key not found! Please add `GEMINI_API_KEY` to your `.streamlit/secrets.toml` file.")
    elif not inputs_valid:
        st.warning("⚠️ Please enter both **Fund Name** and **Portfolio Date** in the sidebar before generating.")
    elif st.button("🚀 Generate Morningstar Template", type="primary"):
```

This is the main validation-and-trigger sequence:

6. The user enters a numeric Morningstar Document ID.
7. The app displays the generated URL and fetches the PDF only after `Process Report` is clicked.
8. The app verifies that the API key exists and all required fund metadata is filled in.
9. The downloaded bytes are passed directly to PyMuPDF without writing a PDF to disk.
4. Once the button is pressed, the script executes the extraction pipeline.

### 3.8 Date normalization for portfolio date

Inside the action block:

```python
try:
    dt_port = pd.to_datetime(portfolio_date)
    formatted_portfolio_date = f"{dt_port.month}/{dt_port.day}/{dt_port.year}"
except Exception:
    formatted_portfolio_date = portfolio_date
5. Enter the Morningstar Document ID, for example `670134091`.
6. Click `Process Report`.
This ensures the portfolio date is standardized to `M/D/YYYY` without leading zeros. If parsing fails, it falls back to the original text string.

### 3.9 Local PDF parsing and Gemini input

```python
    pdf_bytes, pages, relevant_page_indexes
)
```

Text and table content are extracted locally. Only relevant pages without usable text are rendered to images for Gemini vision input. The request is bounded to one extraction call, with limited retry for transient failures or malformed JSON.

### 3.10 Extraction prompt and JSON validation

The script builds a long prompt instructing the model to extract structured financial and investment data from the PDF.

Key points in the prompt:

  - `investments`
  - `total_net_assets`
  - `base_currency`
  - `DERIVATIVES` for derivative instruments
- It requires liability values to be stored as negative numbers.
- It requires `maturity_date` to be formatted as `M/D/YYYY` without leading zeros.

The actual prompt is sent as:

The existing extraction prompt is sent with the compact local document parts through the OpenAI-compatible Gemini endpoint using the stable `gemini-3.8-flash` model. The response is validated for the expected `financial_position`, `investments`, `total_net_assets`, and `base_currency` fields before mapping.

### 3.11 Parsing and output generation

`validate_extraction_json()` verifies the response is an object containing the four expected top-level fields and list-shaped financial position and investment entries. `build_morningstar_output()` then applies the existing mapping, date, coupon, and 19-column Excel logic.

### 3.12 Mapping financial position rows

```python
mapped_rows = []

for item in data.get('financial_position', []):
    holding_name = item.get('item', '')
    holding_name_lower = holding_name.lower()

    if "fair value" in holding_name_lower and ("invest" in holding_name_lower or "asset" in holding_name_lower):
        continue

    if "cash" in holding_name_lower or "bank" in holding_name_lower:
        holding_id = "CASH"
    else:
        holding_id = "N/A"

    mapped_rows.append({
        'Holding Id': holding_id,
        'Holding Name': holding_name,
        'Number of Share': 0,
        'Market Value': item.get('value')
    })
```

#### Inner logic

- Filters out investment-related “fair value” asset lines that are not meaningful at the raw balance-sheet level.
- Converts cash or bank lines to `CASH`.
- Everything else gets `N/A` as a placeholder.
- The data is appended into a row dictionary with the following fields:
  - `Holding Id`
  - `Holding Name`
  - `Number of Share`
  - `Market Value`

### 3.13 Mapping investment holdings

```python
for inv in data.get('investments', []):
    maturity_val = inv.get('maturity_date')
    try:
        if maturity_val and pd.notna(maturity_val):
            dt_mat = pd.to_datetime(maturity_val)
            formatted_maturity = f"{dt_mat.month}/{dt_mat.day}/{dt_mat.year}"
        else:
            formatted_maturity = pd.NaT
    except Exception:
        formatted_maturity = maturity_val

    raw_coupon = inv.get('coupon_rate')
    formatted_coupon = pd.NA
    if raw_coupon is not None and str(raw_coupon).strip().lower() not in ['', 'null', 'n/a', 'none']:
        try:
            formatted_coupon = f"{float(raw_coupon):.3f}"
        except (ValueError, TypeError):
            formatted_coupon = raw_coupon

    raw_id = inv.get('holding_id', 'N/A')
    mapped_rows.append({
        'Holding Id': raw_id,
        'Holding Name': inv.get('name'),
        'Number of Share': inv.get('quantity'),
        'Market Value': inv.get('market_value'),
        'Coupon Rate': formatted_coupon,
        'Maturity Date': formatted_maturity
    })
```

#### Inner logic

- Normalizes bond maturity dates to `M/D/YYYY` format without leading zeros.
- Converts coupon rate to a string with exactly three decimal places when it can be parsed as a number.
- Preserves a raw string if conversion fails.
- Adds the extracted holding ID, holding name, quantity, market value, coupon, and maturity date to the row list.

### 3.14 DataFrame assembly and metadata population

```python
df = pd.DataFrame(mapped_rows)
df['Portfolio Date'] = formatted_portfolio_date
df['Fund Id'] = fund_id
df['Fund Name'] = fund_name
df['Portfolio Currency (Base)'] = extracted_currency
df['Fund TNA'] = fund_tna
```

This adds the global metadata columns shared across all rows.

### 3.15 Standardization cleanup

```python
df['Holding Id'] = df['Holding Id'].fillna('N/A').astype(str).str.upper()
df['Holding Id'] = df['Holding Id'].replace({'NA': 'N/A'})
```

This ensures all Holding IDs are uppercase, null values are replaced with `N/A`, and the special `NA` string is normalized to `N/A`.

### 3.16 Re-alignment to the Morningstar template

```python
df_template = pd.DataFrame(columns=TEMPLATE_COLUMNS)
df_final = pd.concat([df_template, df], ignore_index=True)[TEMPLATE_COLUMNS]
```

This step ensures that the final output contains the exact 19-column schema in the required order, even if some rows are missing values.

### 3.17 Display and Excel export

```python
st.success(f"✅ Extracted data successfully for **{fund_name}** ({formatted_portfolio_date})")
st.dataframe(df_final, use_container_width=True)
```

The app displays a success message and a preview table to the user.

The workbook is created entirely in memory by `build_morningstar_output()`, which lets the script avoid writing local Excel files.

```python
safe_date = formatted_portfolio_date.replace('/', '')
filename = f"{fund_id if fund_id else 'Fund'}_{safe_date}_Morningstar.xlsx"
```

This builds the download file name. The date is flattened to remove the forward slashes (for example `9302023`).

```python
st.download_button(
    label="📥 Download Morningstar Excel Output",
    data=excel_data,
    file_name=filename,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary"
)
```

This provides a Streamlit file download widget for exporting the workbook.

### 3.18 Exception handling

```python
except Exception as e:
    st.error(f"❌ Extraction Error: {e}")
```

The entire generation process is wrapped in a `try/except` block to catch failures from PDF upload, model generation, JSON parsing, or DataFrame shaping and display them to the user in the UI.

---

## 4. End-to-End Execution Flow

### Startup

1. The user launches the application with Streamlit.
2. The app initializes the page title, icon, and wide layout.
3. The sidebar renders the authentication and metadata inputs.

### User input phase

4. The user enters:
   - Gemini API key,
   - fund name,
   - optional fund ID,
   - portfolio date.
5. The app validates that the fund name and portfolio date are present.

### File upload phase

6. The user uploads a PDF report.
7. The app verifies that the API key exists and all required metadata is filled in.
8. The user clicks the primary action button to generate the Morningstar template.

### Extraction phase

9. The script normalizes the date to `M/D/YYYY`.
10. It extracts text and tables locally and detects relevant pages.
11. It renders only relevant image-only pages when vision is needed.
12. It sends one compact request to Gemini with bounded transient-error retries.
13. It validates the JSON response against the expected extraction shape.
14. Gemini returns structured financial and investment data for mapping.

### Transformation phase

15. The script parses the JSON into Python dictionaries.
16. It maps financial statement items into `Holding Id`, `Holding Name`, and `Market Value` rows.
17. It converts investment holdings into row dictionaries with quantity, market value, coupon, maturity, and ID.
18. It creates a pandas DataFrame and adds the common metadata fields: portfolio date, fund ID, fund name, base currency, and fund TNA.
19. It normalizes `Holding Id` values and rebuilds the output table to match the Morningstar template exactly.

### Presentation and export phase

20. The app displays the generated table in the UI with a success message.
21. It creates an Excel workbook in memory.
22. It offers the workbook for download via a Streamlit download button.

### Completion

23. The file is downloaded to the user’s machine and the app resets to the ready state.

---

## 5. Functions, Classes, and Routes

The script defines reusable helpers for:

- local PDF text extraction and table extraction,
- scanned-PDF detection,
- relevant-page and section selection,
- selective vision payload construction,
- Gemini request/retry handling,
- JSON validation,
- Morningstar mapping and Excel generation.
- Built-in Streamlit methods such as:
  - `st.set_page_config`
  - `st.title`
  - `st.write`
  - `st.sidebar.text_input`
    - `st.selectbox`
  - `st.button`
  - `st.spinner`
  - `st.success`
  - `st.dataframe`
  - `st.download_button`
  - `st.error`
  - `st.warning`

The Streamlit UI remains a direct script, while the expensive and provider-specific operations are isolated behind these functions.

---

## 6. Prerequisites

### Required software

- Python 3.9+ recommended (3.10/3.11 are common working choices)
- A modern browser for the Streamlit UI
- A valid Google Gemini API key from Google AI Studio

### Python dependencies

The project dependencies are listed in `requirements.txt`:

```txt
streamlit
pandas
openpyxl
openai
PyMuPDF
```

### Environment assumptions

- Internet access is required for Gemini API calls.
- The uploaded PDF must be readable and well-structured enough for the model to extract the necessary fields.
- The app expects the target fund and date to be clearly identifiable in the source document.

---

## 7. Setup Instructions

### Option A: Standard Python installation

From the project root:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

Configure the Gemini API key:

```text
Copy .env.example to .env
Add your Gemini API key
pip install -r requirements.txt
streamlit run app.py
```

The app checks `st.secrets["GEMINI_API_KEY"]` first and falls back to `GEMINI_API_KEY` from `.env`.

### Option B: Direct install without a virtual environment

```bash
pip install streamlit pandas openpyxl openai PyMuPDF python-dotenv
```

---

## 8. Usage Commands

### Launch the app

```bash
streamlit run app.py
```

On Windows PowerShell, this is the usual command in the activated environment.

### In the browser

1. Open the local Streamlit URL shown in the terminal.
2. Select a provider and enter its API key/token in the sidebar.
3. Enter the target fund name and portfolio date.
4. Enter an optional fund ID.
5. Enter the Morningstar Document ID.
6. Click `Process Report`.
7. Review the parsed output preview.
8. Download the Morningstar Excel workbook.

---

## 9. Expected Output

The script creates an Excel workbook with the following characteristics:

- One worksheet named `Sheet1`
- A single row for each extracted financial position line item or investment holding
- Columns matching the Morningstar template order
- Included metadata such as:
  - `Portfolio Date`
  - `Fund Id`
  - `Fund Name`
  - `Portfolio Currency (Base)`
  - `Fund TNA`

The generated workbook is returned to the user as a file download rather than saved to disk inside the project.

---

## 10. Operational Notes and Limitations

### Strengths

- Very effective for a narrow, structured document-processing workflow
- Automatically handles key value mapping to a consistent schema
- Corrects common formatting issues in dates and coupon rates
- Exports directly to Excel with a Morningstar-style column order

### Limitations

- This app depends on the quality of the PDF and the model’s interpretation of it.
- It works best when the PDF clearly contains a target fund and a specific reporting date.
- There is no formal reconciliation step to validate whether the extracted totals match the original report.
- The extraction prompt is strict and specialized; it is not a general-purpose financial parser.
- The script does not persist or encrypt the Gemini API key.

---

## 11. Example Workflow

A typical use case looks like this:

```text
User opens app
Inputs:
  Fund Name = United ASEAN Discovery Fund
  Fund ID = FS0000B37I
  Portfolio Date = 9/30/2023
Uploads PDF report
Clicks Generate Morningstar Template
Gemini extracts fund-specific values
Script standardizes IDs and dates
Preview table is shown in Streamlit
User downloads Morningstar Excel file
```

---

## 12. Summary

This project is a focused automation workflow for turning a fund factsheet or PDF report into a spreadsheet-ready Morningstar template using AI-assisted extraction. It combines Streamlit’s user interface with Google Gemini’s document understanding and pandas’ spreadsheet formatting to provide a functional end-to-end data extraction tool.

It is best understood as a single-page, document-intelligence workflow rather than a full application framework. The value comes from its narrow specialization: converting PDF fund data into structured, exportable Morningstar-style tables.
