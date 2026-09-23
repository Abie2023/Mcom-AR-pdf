import streamlit as st
import pandas as pd
import io
import json
import base64
import re
import time
import pymupdf
from openai import OpenAI, OpenAIError

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Fund Report Parser",
    page_icon="📄",
    layout="wide"
)

st.title("AR Report Portfolio holdings")
st.write("Upload a PDF fund report, specify the target fund details in the sidebar, and extract structured financial data into the Morningstar template.")

# Let the user input their own API key
st.sidebar.header("🔑 Authentication")
API_KEY = st.sidebar.text_input(
    "Gemini API Key", 
    type="password", 
    help="Get your free key from Google AI Studio (starts with AQ.). This key is not saved."
)
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
    """Raised when the configured AI provider cannot return valid extraction data."""


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
    """Extract page text and locally detected table rows from an uploaded PDF."""
    try:
        pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc

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
            raise ValueError("The uploaded file is not a readable PDF.") from exc

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


def extract_json_with_ai(api_key, document_parts, extraction_prompt, model="gemini-3.8-flash", status_callback=None):
    """Send one compact request to Gemini with bounded retry/backoff."""
    max_attempts = 3
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        for attempt in range(1, max_attempts + 1):
            if status_callback:
                status_callback(f"Gemini request attempt {attempt} of {max_attempts}...")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": document_parts + [
                                {"type": "text", "text": extraction_prompt}
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                )
                if not response.choices:
                    raise AIExtractionError("Gemini returned no response choices.")
                try:
                    return validate_extraction_json(response.choices[0].message.content)
                except AIExtractionError:
                    if attempt == 1:
                        time.sleep(1)
                        continue
                    raise
            except OpenAIError as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = status_code in {429, 503} or isinstance(exc, (TimeoutError, ConnectionError))
                if not retryable or attempt == max_attempts:
                    if status_code == 429:
                        message = "Gemini rate limit reached after retries."
                    elif status_code == 503:
                        message = "Gemini is temporarily unavailable after retries."
                    elif status_code in {400, 401, 403, 404}:
                        message = f"Gemini request failed ({status_code}). Check the API key, model, and request."
                    else:
                        message = f"Gemini API request failed: {exc}"
                    raise AIExtractionError(message) from exc
                time.sleep(2 ** (attempt - 1))
        raise AIExtractionError("Gemini request failed after retries.")
    except AIExtractionError:
        raise


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


# --- 3. Main File Upload & Processing Pipeline ---
uploaded_file = st.file_uploader("Upload Fund Report (PDF)", type=["pdf"])

if uploaded_file is not None:
    if not API_KEY:
        st.error("⚠️ API Key not found! Please add `GEMINI_API_KEY` to your `.streamlit/secrets.toml` file.")
    elif not inputs_valid:
        st.warning("⚠️ Please enter both **Fund Name** and **Portfolio Date** in the sidebar before generating.")
    elif st.button("🚀 Generate Morningstar Template", type="primary"):
        with st.spinner(f"Extracting data for '{fund_name}' as of {portfolio_date}..."):
            try:
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

                pdf_bytes = uploaded_file.getvalue()
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
                request_status = st.empty()

                data = extract_json_with_ai(
                    API_KEY,
                    document_parts,
                    extraction_prompt,
                    status_callback=request_status.info,
                )
                request_status.success("Gemini request completed successfully.")

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
                st.error(f"❌ Gemini API Error: {e}")
            except ValueError as e:
                st.error(f"❌ PDF/JSON Error: {e}")
            except Exception as e:
                st.error(f"❌ Extraction Error: {e}")