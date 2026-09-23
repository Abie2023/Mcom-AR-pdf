import streamlit as st
import pandas as pd
import io
import json
import base64
import pymupdf
from openai import OpenAI, OpenAIError

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Fund Report Parser",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Fund Report -> Morningstar Template")
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

                # The OpenAI-compatible Gemini endpoint accepts image_url parts, not PDF file parts.
                pdf_bytes = uploaded_file.getvalue()
                try:
                    pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                    if pdf_document.page_count == 0:
                        raise ValueError("The uploaded PDF contains no pages.")

                    document_parts = []
                    for page in pdf_document:
                        page_image = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                        image_data = base64.b64encode(page_image.tobytes("png")).decode("utf-8")
                        document_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        })
                    pdf_document.close()
                except pymupdf.FileDataError as exc:
                    raise ValueError("The uploaded file is not a readable PDF.") from exc

                client = OpenAI(
                    api_key=API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                response = client.chat.completions.create(
                    model="gemini-3.8-flash",
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
                    raise ValueError("Gemini returned no response choices.")

                response_text = response.choices[0].message.content
                if not response_text:
                    raise ValueError("Gemini returned an empty response.")

                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    raise ValueError("Gemini returned invalid JSON.") from exc
                if not isinstance(data, dict):
                    raise ValueError("Gemini returned JSON, but it was not a JSON object.")

                fund_tna = data.get('total_net_assets', 0)
                extracted_currency = data.get('base_currency', '')
                
                mapped_rows = []
                
                # Process Financial Position Items
                for item in data.get('financial_position', []):
                    holding_name = item.get('item', '')
                    holding_name_lower = holding_name.lower()

                    # Filter out "Investment at Fair Value" balance sheet lines 
                    if "fair value" in holding_name_lower and ("invest" in holding_name_lower or "asset" in holding_name_lower):
                        continue

                    # Assign CASH for cash/bank, otherwise strictly "N/A"
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
                
                # Process Investment Holdings
                for inv in data.get('investments', []):
                    maturity_val = inv.get('maturity_date')
                    # Force M/D/YYYY format (no leading zeros) for maturity dates
                    try:
                        if maturity_val and pd.notna(maturity_val):
                            dt_mat = pd.to_datetime(maturity_val)
                            formatted_maturity = f"{dt_mat.month}/{dt_mat.day}/{dt_mat.year}"
                        else:
                            formatted_maturity = pd.NaT
                    except Exception:
                        formatted_maturity = maturity_val

                    # Force Coupon Rate to exactly 3 decimal points
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
                
                # Build Dataframe and populate shared user metadata
                df = pd.DataFrame(mapped_rows)
                df['Portfolio Date'] = formatted_portfolio_date
                df['Fund Id'] = fund_id
                df['Fund Name'] = fund_name
                df['Portfolio Currency (Base)'] = extracted_currency
                df['Fund TNA'] = fund_tna
                
                # Final cleanup: Ensure all Holding Ids are uppercase and 'NA' becomes 'N/A'
                df['Holding Id'] = df['Holding Id'].fillna('N/A').astype(str).str.upper()
                df['Holding Id'] = df['Holding Id'].replace({'NA': 'N/A'})

                # Re-align with exact 19-column schema
                df_template = pd.DataFrame(columns=TEMPLATE_COLUMNS)
                df_final = pd.concat([df_template, df], ignore_index=True)[TEMPLATE_COLUMNS]
                
                # Display success & preview table
                st.success(f"✅ Extracted data successfully for **{fund_name}** ({formatted_portfolio_date})")
                st.dataframe(df_final, use_container_width=True)

                # Export to Excel in-memory
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_final.to_excel(writer, index=False, sheet_name='Sheet1')
                
                # Ensure filename uses safe date format
                safe_date = formatted_portfolio_date.replace('/', '')
                filename = f"{fund_id if fund_id else 'Fund'}_{safe_date}_Morningstar.xlsx"
                
                st.download_button(
                    label="📥 Download Morningstar Excel Output",
                    data=excel_buffer.getvalue(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
                
            except OpenAIError as e:
                st.error(f"❌ Gemini API Error: {e}")
            except ValueError as e:
                st.error(f"❌ PDF/JSON Error: {e}")
            except Exception as e:
                st.error(f"❌ Extraction Error: {e}")