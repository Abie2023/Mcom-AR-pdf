import streamlit as st
import pandas as pd
import os
import io
import json
import base64
import re
import time
import requests
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Fund Report Parser",
    page_icon="📄",
    layout="wide"
)

MORNINGSTAR_DOCUMENT_BASE_URL = "https://doc.morningstar.com/Document"
MORNINGSTAR_DOCUMENT_SUFFIX = ".msdoc/original"
MORNINGSTAR_CLIENT_ID = "globaldocuments"
MORNINGSTAR_ACCESS_KEY = "52dbc583e1012395"
MORNINGSTAR_REQUEST_TIMEOUT_SECONDS = 30
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
PREFERRED_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

st.title("AR Report Portfolio holdings")
st.write("Enter a Morningstar Document ID, specify the target fund details in the sidebar, and extract structured financial data into the Morningstar template.")

# Gemini authentication and model selection
st.sidebar.header("🔑 Gemini authentication")
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY")
st.sidebar.markdown("---")

# --- 2. Dynamic User Inputs (Sidebar) ---
st.sidebar.header("⚙️ Target Fund Parameters")

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

# Mandatory field check helper
inputs_valid = bool(fund_name.strip() and portfolio_date.strip())

# Morningstar 19-Column Schema Definition
TEMPLATE_COLUMNS = [
    'Portfolio Date', 'Fund Id', 'Fund Name', 'Holding Id', 'Holding Name', 
    'Number of Share', 'Market Value', 'Coupon Rate', 'Maturity Date', 
    'Portfolio Currency (Base)', 'Local MValue', 'Currency (Local)', 
    'Cost (Base)', 'Country', 'Fund TNA', 'Unnamed: 15', '% TNA', 
    'Unnamed: 17', 'AssetType Reference'
]

RELEVANT_SECTION_KEYWORDS = (
    "statement of assets and liabilities",
    "statement of investments",
    "schedule of investments",
    "portfolio of investments",
    "net assets",
    "total investments",
    "cash and cash equivalents",
    "derivatives",
    "securities portfolio",
    "statement of net assets",
    "portfolio breakdown",
    "top ten holdings",
)
MAX_RELEVANT_PAGES = 24
MAX_LOCAL_CONTEXT_CHARS = 120_000


class AIExtractionError(Exception):
    """Raised when Gemini cannot return valid extraction data."""

    def __init__(self, message, model=None, status_code=None, retryable=False, attempts=0):
        super().__init__(message)
        self.model = model
        self.status_code = status_code
        self.retryable = retryable
        self.attempts = attempts


class MorningstarDownloadError(Exception):
    """Raised when a Morningstar PDF cannot be fetched."""


def discover_gemini_models(api_key):
    """Return preferred Gemini models that pass an access/capability check."""
    if not api_key:
        raise AIExtractionError("Gemini API key is not configured.")
    try:
        response = requests.get(
            GEMINI_MODELS_URL,
            params={"key": api_key, "pageSize": 100},
            timeout=MORNINGSTAR_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        models = response.json().get("models", [])
    except requests.Timeout as exc:
        raise AIExtractionError("Gemini model discovery timed out.", retryable=True) from exc
    except requests.ConnectionError as exc:
        raise AIExtractionError("Gemini model discovery network error.", retryable=True) from exc
    except requests.HTTPError as exc:
        status_code = response.status_code
        raise AIExtractionError(
            f"Gemini model discovery failed ({status_code}): {response.text[:240]}",
            status_code=status_code,
        ) from exc
    except (requests.RequestException, ValueError) as exc:
        raise AIExtractionError(f"Gemini model discovery failed: {exc}") from exc

    discovered = []
    listed_models = []
    for model in models:
        name = model.get("name", "")
        methods = model.get("supportedGenerationMethods", [])
        model_id = name.removeprefix("models/")
        if model_id in PREFERRED_GEMINI_MODELS and "generateContent" in methods:
            listed_models.append(model_id)

    usable_models = []
    for model_id in PREFERRED_GEMINI_MODELS:
        if model_id not in listed_models:
            continue
        try:
            model_response = requests.get(
                f"{GEMINI_MODELS_URL}/{model_id}",
                params={"key": api_key},
                timeout=MORNINGSTAR_REQUEST_TIMEOUT_SECONDS,
            )
            if model_response.status_code != 200:
                continue
            model_details = model_response.json()
            if "generateContent" in model_details.get("supportedGenerationMethods", []):
                usable_models.append(model_id)
        except (requests.RequestException, ValueError):
            continue
    return usable_models

if "gemini_models" not in st.session_state:
    st.session_state.gemini_models = []
if st.sidebar.button("Refresh Gemini Models"):
    if not API_KEY:
        st.sidebar.warning("Enter a Gemini API key before refreshing models.")
    else:
        try:
            st.session_state.gemini_models = discover_gemini_models(API_KEY)
            if st.session_state.gemini_models:
                st.sidebar.success(f"Found {len(st.session_state.gemini_models)} usable Gemini models.")
            else:
                st.sidebar.warning("No compatible Gemini models were returned; using the known model list.")
        except AIExtractionError as exc:
            st.session_state.gemini_models = []
            st.sidebar.warning(f"Model discovery failed: {exc}. Using the known model list.")

available_gemini_models = [
    model for model in PREFERRED_GEMINI_MODELS
    if model in st.session_state.gemini_models
]
if not available_gemini_models:
    available_gemini_models = PREFERRED_GEMINI_MODELS
if st.session_state.get("selected_gemini_model") not in available_gemini_models:
    st.session_state.selected_gemini_model = available_gemini_models[0]
selected_model = st.sidebar.selectbox(
    "Gemini extraction model",
    available_gemini_models,
    index=0,
    key="selected_gemini_model",
)
st.sidebar.caption(f"Selected model: {selected_model}")


def build_morningstar_url(document_id):
    """Build the Morningstar document URL from a numeric document ID."""
    if not re.fullmatch(r"\d+", document_id.strip()):
        raise ValueError("Morningstar Document ID must contain only digits.")
    return (
        f"{MORNINGSTAR_DOCUMENT_BASE_URL}/{document_id.strip()}"
        f"{MORNINGSTAR_DOCUMENT_SUFFIX}?clientid={MORNINGSTAR_CLIENT_ID}"
        f"&key={MORNINGSTAR_ACCESS_KEY}"
    )


def fetch_morningstar_pdf(document_id):
    """Fetch a Morningstar PDF into memory without writing it to disk."""
    document_url = build_morningstar_url(document_id)
    try:
        response = requests.get(
            document_url,
            timeout=MORNINGSTAR_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise MorningstarDownloadError("Morningstar request timed out.") from exc
    except requests.ConnectionError as exc:
        raise MorningstarDownloadError("Could not connect to Morningstar.") from exc
    except requests.RequestException as exc:
        raise MorningstarDownloadError(f"Morningstar network error: {exc}") from exc

    status_messages = {
        403: "Morningstar access denied for this document.",
        404: "Morningstar document not found.",
        429: "Morningstar rate limit reached. Please try again later.",
    }
    if response.status_code in status_messages:
        raise MorningstarDownloadError(status_messages[response.status_code])
    if 500 <= response.status_code <= 599:
        raise MorningstarDownloadError(
            f"Morningstar server error ({response.status_code}). Please try again later."
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise MorningstarDownloadError(
            f"Morningstar request failed ({response.status_code})."
        ) from exc

    if not response.content:
        raise MorningstarDownloadError("Morningstar returned an empty response.")
    return document_url, response.content


def extract_page_tables(page):
    """Extract simple table rows locally when the installed PyMuPDF supports it."""
    if not hasattr(page, "find_tables"):
        return []

    try:
        tables = page.find_tables().tables
    except Exception:
        return []

    extracted_tables = []
    for table in tables:
        rows = table.extract()
        if rows:
            extracted_tables.append("\n".join(
                "\t".join(str(cell or "").strip() for cell in row)
                for row in rows
            ))
    return extracted_tables


def extract_pdf_content(pdf_bytes):
    """Extract page text and locally detected table rows from a PDF in memory."""
    try:
        pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise ValueError("The Morningstar response is not a readable PDF.") from exc

    pages = []
    try:
        for page_number, page in enumerate(pdf_document, start=1):
            text = page.get_text("text").strip()
            pages.append({
                "number": page_number,
                "text": text,
                "tables": extract_page_tables(page),
                "has_images": bool(page.get_images(full=True)),
            })
    finally:
        pdf_document.close()

    if not pages:
        raise ValueError("The uploaded PDF contains no pages.")
    return pages


def detect_scanned_pdf(pages):
    """Classify a PDF as scanned when most pages lack meaningful text."""
    text_page_count = sum(len(page["text"]) >= 40 for page in pages)
    return text_page_count / len(pages) < 0.5


def detect_relevant_pages(pages, target_fund_name="", reporting_date=""):
    """Find keyword-matching pages and include adjacent continuation pages."""
    page_scores = []
    exact_fund_name = " ".join(target_fund_name.lower().split())
    exact_fund_pattern = re.compile(
        rf"(?<!\w){re.escape(exact_fund_name)}(?!\s+side\s*-\s*pocket)(?!\w)"
    ) if exact_fund_name else None
    fund_terms = [term for term in exact_fund_name.split() if len(term) > 2]
    date_terms = [term.lower() for term in reporting_date.replace("/", " ").split()]
    target_section_matches = set()
    for page in pages:
        page_text = page["text"].lower()
        score = sum(page_text.count(keyword) for keyword in RELEVANT_SECTION_KEYWORDS)
        if exact_fund_pattern:
            exact_matches = len(exact_fund_pattern.findall(page_text))
            score += 100 * exact_matches
            if exact_matches and any(
                keyword in page_text for keyword in RELEVANT_SECTION_KEYWORDS
            ) and "table of contents" not in page_text and "notes to the financial statements" not in page_text:
                target_section_matches.add(page["number"] - 1)
        score += 5 * sum(page_text.count(term) for term in fund_terms)
        score += 2 * sum(page_text.count(term) for term in date_terms)
        if score:
            page_scores.append((score, page["number"] - 1))

    if target_section_matches:
        matches = target_section_matches
    elif page_scores:
        page_scores.sort(reverse=True)
        matches = {page_index for _, page_index in page_scores[:MAX_RELEVANT_PAGES]}
    else:
        matches = {page["number"] - 1 for page in pages[:min(3, len(pages))]}

    relevant = set(matches)
    if not target_section_matches:
        for page_index in list(matches):
            if page_index > 0:
                relevant.add(page_index - 1)
            if page_index + 1 < len(pages):
                relevant.add(page_index + 1)
    return sorted(relevant)[:MAX_RELEVANT_PAGES]


def extract_relevant_sections(pages, relevant_page_indexes):
    """Build compact, page-labelled text and table context for Gemini."""
    sections = []
    current_length = 0
    for page_index in relevant_page_indexes:
        page = pages[page_index]
        content = [f"--- PDF page {page['number']} ---"]
        if page["text"]:
            content.append(page["text"])
        for table_number, table in enumerate(page["tables"], start=1):
            content.append(f"[Local table {table_number}]\n{table}")
        section = "\n".join(content)
        if current_length + len(section) > MAX_LOCAL_CONTEXT_CHARS:
            break
        sections.append(section)
        current_length += len(section)
    return "\n\n".join(sections)


def build_gemini_document_parts(pdf_bytes, pages, relevant_page_indexes):
    """Use local text by default and render only relevant pages needing vision."""
    local_context = extract_relevant_sections(pages, relevant_page_indexes)
    document_parts = []
    if local_context:
        document_parts.append({"type": "text", "text": local_context})

    pages_needing_vision = [
        page_index for page_index in relevant_page_indexes
        if len(pages[page_index]["text"]) < 40
    ]

    if pages_needing_vision:
        try:
            pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            try:
                for page_index in pages_needing_vision:
                    page_image = pdf_document[page_index].get_pixmap(
                        matrix=pymupdf.Matrix(2, 2),
                        alpha=False,
                    )
                    image_data = base64.b64encode(page_image.tobytes("png")).decode("utf-8")
                    document_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    })
            finally:
                pdf_document.close()
        except pymupdf.FileDataError as exc:
            raise ValueError("The Morningstar response is not a readable PDF.") from exc

    if not document_parts:
        raise ValueError("No relevant PDF content could be extracted.")
    return document_parts, len(pages_needing_vision)


def validate_extraction_json(response_text):
    """Validate the model response before it reaches Morningstar mapping."""
    if not response_text:
        raise AIExtractionError("Gemini returned an empty response.")
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise AIExtractionError("Gemini returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise AIExtractionError("Gemini returned JSON, but it was not a JSON object.")
    for key in ("financial_position", "investments", "total_net_assets", "base_currency"):
        if key not in data:
            raise AIExtractionError(f"Gemini JSON is missing required field: {key}.")
    if not isinstance(data["financial_position"], list):
        raise AIExtractionError("Gemini JSON field 'financial_position' must be a list.")
    if not isinstance(data["investments"], list):
        raise AIExtractionError("Gemini JSON field 'investments' must be a list.")
    if any(not isinstance(item, dict) for item in data["financial_position"]):
        raise AIExtractionError("Gemini financial position entries must be JSON objects.")
    if any(not isinstance(item, dict) for item in data["investments"]):
        raise AIExtractionError("Gemini investment entries must be JSON objects.")
    return data


def _document_parts_characters(document_parts):
    """Estimate the text size sent to an OpenAI-compatible provider."""
    return sum(
        len(part.get("text", ""))
        for part in document_parts
        if part.get("type") == "text"
    )


def _reduce_document_parts(document_parts):
    """Reduce only local text context once when a provider reports a context limit."""
    reduced_parts = []
    for part in document_parts:
        if part.get("type") == "text":
            reduced_parts.append({
                **part,
                "text": part.get("text", "")[:MAX_LOCAL_CONTEXT_CHARS // 2],
            })
        else:
            reduced_parts.append(part)
    return reduced_parts


def _is_context_limit_error(error):
    message = str(error).lower()
    return any(term in message for term in (
        "context length",
        "context window",
        "maximum context",
        "too many tokens",
        "token limit",
    ))


def generate_gemini_extraction(api_key, model, document_parts, extraction_prompt, status_callback=None):
    """Send one Gemini extraction request with bounded, status-aware retries."""
    if not api_key:
        raise AIExtractionError("Gemini API key is not configured.", model=model)

    current_parts = document_parts
    context_reduced = False
    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        if status_callback:
            status_callback(
                f"Selected model: {model} | "
                f"Input characters: {_document_parts_characters(current_parts):,} | "
                f"Estimated tokens: ~{_document_parts_characters(current_parts) // 4:,} | "
                f"API request attempt: {attempt} of {max_attempts}"
            )
        started_at = time.perf_counter()
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=GEMINI_BASE_URL,
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": current_parts + [
                            {"type": "text", "text": extraction_prompt}
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
            if status_callback:
                status_callback(
                    f"Selected model: {getattr(response, 'model', model)} | "
                    "API response status: success | "
                    f"Response time: {time.perf_counter() - started_at:.2f}s"
                )
            if not response.choices:
                raise AIExtractionError("Gemini returned no response choices.", model=model)
            data = validate_extraction_json(response.choices[0].message.content)
            return {
                "data": data,
                "model": getattr(response, "model", model),
                "attempts": attempt,
                "status": "success",
                "elapsed_seconds": time.perf_counter() - started_at,
                "input_characters": _document_parts_characters(current_parts),
                "estimated_tokens": _document_parts_characters(current_parts) // 4,
            }
        except AIExtractionError as exc:
            if exc.model is None:
                exc.model = model
            exc.attempts = attempt
            raise
        except OpenAIError as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 404:
                message = "This Gemini model is not available for your API key. Please select another available model."
            elif status_code == 429:
                message = "Gemini quota/rate limit reached."
            elif status_code == 503:
                message = "Gemini temporarily unavailable."
            elif status_code in {400, 401, 403}:
                message = f"Gemini request failed ({status_code}). Check the key, model, and request."
            else:
                message = f"Gemini API request failed: {exc}"

            if status_code in {400, 413} and not context_reduced and _is_context_limit_error(exc):
                current_parts = _reduce_document_parts(current_parts)
                context_reduced = True
                continue
            if status_code == 503 and attempt == 1:
                time.sleep(2 ** (attempt - 1))
                continue
            raise AIExtractionError(
                message,
                model=model,
                status_code=status_code,
                retryable=status_code == 503,
                attempts=attempt,
            ) from exc
        except (TimeoutError, ConnectionError) as exc:
            if attempt == 1:
                time.sleep(2 ** (attempt - 1))
                continue
            raise AIExtractionError(
                f"Gemini connection/network error: {exc}",
                model=model,
                retryable=True,
                attempts=attempt,
            ) from exc
        except json.JSONDecodeError as exc:
            raise AIExtractionError(
                "Gemini returned malformed JSON.",
                model=model,
                attempts=attempt,
            ) from exc

    raise AIExtractionError("Gemini request failed after the allowed retry.", model=model)


def build_morningstar_output(data, formatted_portfolio_date, fund_id, fund_name):
    """Apply the existing extraction mapping and build the Morningstar workbook."""
    fund_tna = data.get('total_net_assets', 0)
    extracted_currency = data.get('base_currency', '')

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

    df = pd.DataFrame(mapped_rows)
    df['Portfolio Date'] = formatted_portfolio_date
    df['Fund Id'] = fund_id
    df['Fund Name'] = fund_name
    df['Portfolio Currency (Base)'] = extracted_currency
    df['Fund TNA'] = fund_tna

    df['Holding Id'] = df['Holding Id'].fillna('N/A').astype(str).str.upper()
    df['Holding Id'] = df['Holding Id'].replace({'NA': 'N/A'})

    df_template = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    df_final = pd.concat([df_template, df], ignore_index=True)[TEMPLATE_COLUMNS]

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_final.to_excel(writer, index=False, sheet_name='Sheet1')
    return df_final, excel_buffer.getvalue()


# --- 3. Morningstar Document Fetch & Processing Pipeline ---
document_id = st.text_input(
    "Morningstar Document ID",
    value="",
    placeholder="e.g. 670134091",
    help="Enter the numeric Morningstar Document ID only.",
)
document_url = None
if document_id.strip():
    try:
        document_url = build_morningstar_url(document_id)
        st.caption(f"Morningstar URL: {document_url}")
    except ValueError as exc:
        st.warning(str(exc))

st.info(f"Selected Gemini model: {selected_model}")

if st.button("Process Report", type="primary"):
    if not API_KEY:
        st.error(
            "⚠️ Gemini API key is not configured. Copy `.env.example` to `.env`, "
            "put your key in `.env`, and restart Streamlit."
        )
    elif not inputs_valid:
        st.warning("⚠️ Please enter both **Fund Name** and **Portfolio Date** in the sidebar before processing.")
    elif not document_url:
        st.error("⚠️ Enter a valid numeric Morningstar Document ID before processing.")
    else:
        st.info(f"Fetching Morningstar document: {document_url}")
        with st.spinner(f"Extracting data for '{fund_name}' as of {portfolio_date}..."):
            try:
                fetched_url, pdf_bytes = fetch_morningstar_pdf(document_id)
                st.success(f"Morningstar PDF fetched successfully from {fetched_url}")

                # Force format portfolio date to M/D/YYYY (no leading zeros)
                try:
                    dt_port = pd.to_datetime(portfolio_date)
                    formatted_portfolio_date = f"{dt_port.month}/{dt_port.day}/{dt_port.year}"
                except Exception:
                    formatted_portfolio_date = portfolio_date

                # Dynamic extraction prompt with uppercase Holding ID rules
                extraction_prompt = f"""
You are an expert financial data extractor. I have attached a fund report/factsheet PDF.

TARGET PARAMETERS:
- Target Fund Name: "{fund_name}"
- Target Reporting Date: "{formatted_portfolio_date}"

CRITICAL INSTRUCTION:
Extract data ONLY for the exact target fund "{fund_name}" as of date "{formatted_portfolio_date}".
Ignore all other funds, tables, and dates present in the document.

Extract the following into a valid JSON object:
1. 'financial_position': List of non-investment asset and liability line items from Statement of Financial Position for "{fund_name}" as of "{formatted_portfolio_date}".
   - Fields: "item" (string), "value" (numeric integer/float).
   - Convert liability values to NEGATIVE numbers.

2. 'investments': Detailed portfolio list/breakdown of holdings for "{fund_name}" as of "{formatted_portfolio_date}".
   - Fields: 
     - "holding_id" (string). You MUST assign this ID strictly based on the following standardization rules (use EXACT uppercase strings):
        - '*E*': Common Stock, Real Estate Investment Trusts, Preferred Stock
        - '*B*': Corporate Bonds, Government/Treasury Bonds, Municipal Bonds, Convertible Bond, Floating/Variable rate notes, Mortgage-Backed Security, Asset-Backed Security, Convertible Preferred
        - 'CASH': Cash Equivalents, Cash - CD/Time Deposit, Cash - Commercial Paper, Cash - Repurchase Agreement
        - '*QQ*': Other Assets and Liabilities, Commodity, Property
        - 'DERIVATIVES': Derivatives, Swaps, Futures, Forwards, Options, Warrants, Rights, Units
        - 'FUND': Mutual Fund, Money Market Fund, Closed End Fund, Open End Fund, Exchange Traded Funds (ETF), Separate Account, CIT/Custom Fund
     - "name" (string)
     - "quantity" (numeric or null)
     - "market_value" (numeric)
     - "coupon_rate" (numeric/string or null, extract clean number if it is a bond)
     - "maturity_date" (string formatted M/D/YYYY without leading zeros, or null)
   - For Corporate Bonds: Include Credit Rating in the holding name if available.

3. 'total_net_assets': Single numeric value for Net Asset Value / Total Net Assets for "{fund_name}" on "{formatted_portfolio_date}".

4. 'base_currency': 3-letter ISO 4217 currency code (e.g., MYR, USD, SGD).

Return ONLY raw JSON without markdown code fences. Remove currency symbols and formatting commas from numbers.
"""

                pages = extract_pdf_content(pdf_bytes)
                scanned_pdf = detect_scanned_pdf(pages)
                relevant_page_indexes = detect_relevant_pages(
                    pages,
                    target_fund_name=fund_name,
                    reporting_date=formatted_portfolio_date,
                )
                document_parts, vision_page_count = build_gemini_document_parts(
                    pdf_bytes,
                    pages,
                    relevant_page_indexes,
                )
                local_text_page_count = sum(bool(page["text"]) for page in pages)
                st.info(
                    "PDF analysis: "
                    f"{len(pages)} total pages; "
                    f"{local_text_page_count} pages extracted locally; "
                    f"{len(relevant_page_indexes)} relevant pages selected; "
                    f"{vision_page_count} pages require vision; "
                    f"classification: {'scanned/image-based' if scanned_pdf else 'text-based'}."
                )
                st.caption(
                    f"PDF pages: {len(pages)} | Locally parsed pages: {local_text_page_count} | "
                    f"Relevant pages: {len(relevant_page_indexes)} | Vision pages: {vision_page_count}"
                )
                request_status = st.empty()
                provider_result = generate_gemini_extraction(
                    API_KEY,
                    selected_model,
                    document_parts,
                    extraction_prompt,
                    status_callback=request_status.info,
                )
                response_time = provider_result["elapsed_seconds"]
                data = provider_result["data"]
                st.session_state.last_gemini_failure = None
                st.success(
                    f"Gemini request succeeded with **{provider_result['model']}**. "
                    f"Requests: {provider_result['attempts']} | "
                    f"Input characters: {provider_result['input_characters']:,} | "
                    f"Estimated input tokens: ~{provider_result['estimated_tokens']:,} | "
                    f"Response time: {response_time:.2f}s"
                )

                df_final, excel_data = build_morningstar_output(
                    data,
                    formatted_portfolio_date,
                    fund_id,
                    fund_name,
                )
                
                # Display success & preview table
                st.success(f"✅ Extracted data successfully for **{fund_name}** ({formatted_portfolio_date})")
                st.dataframe(df_final, use_container_width=True)

                # Ensure filename uses safe date format
                safe_date = formatted_portfolio_date.replace('/', '')
                filename = f"{fund_id if fund_id else 'Fund'}_{safe_date}_Morningstar.xlsx"
                
                st.download_button(
                    label="📥 Download Morningstar Excel Output",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
                
            except AIExtractionError as e:
                st.session_state.last_gemini_failure = {
                    "model": e.model or selected_model,
                    "status_code": e.status_code,
                    "message": str(e),
                }
                status_text = f"HTTP/status {e.status_code}" if e.status_code else "status unavailable"
                retry_text = "Retrying is appropriate only for a single transient 503/network failure." if e.retryable else "Automatic retry is not recommended for this failure."
                st.error(
                    f"❌ Gemini API Error | Selected model: {e.model or selected_model} | "
                    f"{status_text} | Requests: {e.attempts} | {e}\n\n{retry_text}"
                )
            except MorningstarDownloadError as e:
                st.error(f"❌ Morningstar download error: {e}")
            except ValueError as e:
                st.error(f"❌ Input/PDF/JSON Error: {e}")
            except Exception as e:
                st.error(f"❌ Extraction Error: {e}")

if st.session_state.get("last_gemini_failure"):
    failure = st.session_state.last_gemini_failure
    model_position = available_gemini_models.index(failure["model"]) if failure["model"] in available_gemini_models else -1
    next_position = (model_position + 1) % len(available_gemini_models)
    if st.button("Try another Gemini model"):
        st.session_state.selected_gemini_model = available_gemini_models[next_position]
        st.session_state.last_gemini_failure = None
        st.rerun()