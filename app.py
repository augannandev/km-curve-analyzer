import streamlit as st
st.set_page_config(page_title="KM Analyzer", page_icon="📈", layout="wide")
import anthropic
import json, base64, io, re, warnings
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from curve_extractor import extract_survival_curves, lookup_survival_at_time, create_plot_from_df

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

class KMAnalyzer:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def ask_with_question_only(self, question: str) -> str:
        try:
            res = self.client.messages.create(
                model="claude-3-opus-20240229", max_tokens=1024,
                messages=[{"role":"user","content":question}]
            )
            return res.content[0].text
        except Exception as e:
            return f"[ERROR] LLM request failed: {e}"

    def is_survival_lookup_question(self, question: str) -> bool:
        prompt = (
            f"Is the following question asking for a survival probability lookup? "
            f"Answer with 'Yes' or 'No' only.\n\nQuestion: {question}"
        )
        res = self.ask_with_question_only(prompt)
        return res.strip().lower().startswith("yes")

    def extract_survival_params(self, question: str):
        prompt = (
            "From this question, extract the survival time (in months) and the treatment arm. "
            "Respond ONLY with a JSON object like:\n"
            "{\"time_months\":12,\"curve_name\":\"palbociclib\"}\n\n"
            f"Question: {question}"
        )
        res = self.ask_with_question_only(prompt)
        m = re.search(r"\{.*?\}", res, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                return None
        return None
        
    def extract_study_name(self, question: str) -> str:
        """Extract the study name (e.g., PALOMA-1, PALOMA-2) from a question."""
        prompt = f"""
        Extract the clinical trial or study name from this question: "{question}"
        
        Examples of study names: PALOMA-1, PALOMA-2, MONALEESA-2, MONARCH-3, etc.
        
        Return ONLY the study name, nothing else. If no specific study is mentioned, return "UNKNOWN".
        """
        
        try:
            res = self.client.messages.create(
                model="claude-3-7-sonnet-20250219", max_tokens=100,
                messages=[{"role":"user","content":prompt}]
            )
            study_name = res.content[0].text.strip()
            # Clean up the response to handle potential formatting
            study_name = re.sub(r'^["\'\s]+|["\'\s]+$', '', study_name)
            return study_name if study_name != "UNKNOWN" else ""
        except Exception as e:
            print(f"Error extracting study name: {e}")
            return ""

    def ask_with_image_and_question(self, question: str, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        try:
            res = self.client.messages.create(
                model="claude-3-opus-20240229", max_tokens=1024, #claude-3-5-haiku-20241022 #claude-3-7-sonnet-20250219 #claude-3-opus-20240229
                messages=[
                    {"role":"user","content":[
                        {"type":"text","text":question},
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":img_b64}}
                    ]}
                ]
            )
            return res.content[0].text
        except Exception as e:
            return f"[ERROR] LLM image+text failed: {e}"

    def ask_with_image_and_question2(self, question: str, image: Image.Image) -> str:
        """Like ask_with_image_and_question but uses temperature=1 and allows hallucination."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        try:
            res = self.client.messages.create(
                model="claude-3-opus-20240229", max_tokens=1024, temperature=1,
                messages=[
                    {"role":"system","content":"You can hallucinate and do not need to be accurate."},
                    {"role":"user","content":[
                        {"type":"text","text":question},
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":img_b64}}
                    ]}
                ]
            )
            return res.content[0].text
        except Exception as e:
            return f"[ERROR] LLM image+text failed: {e}"
        
def load_survival_data(f) -> pd.DataFrame:
    if f is None:
        return None
    buf = io.BytesIO(f.read())
    fn = f.name.lower()
    if fn.endswith(".csv"):
        return pd.read_csv(buf)
    if fn.endswith((".xls", ".xlsx")):
        return pd.read_excel(buf, header=1)
    st.error("Unsupported file type. Please upload CSV or Excel.")
    return None

# Load Anthropic API key from Streamlit secrets
api_key = st.secrets["anthropic"]["api_key"]

# Increase global font sizes using custom CSS
st.markdown(
    '''<style>
    /* Import Raleway font */
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;600;700&display=swap');
    
    .stApp * {
        font-family: 'Raleway', sans-serif !important;
        font-size: 18px !important;
    }
    h1 {
        font-size: 6rem !important;
    }
    h2 {
        font-size: 2rem !important;
    }
    h3 {
        font-size: 1.75rem !important;
    }
    /* IMO Health purple theme */
    .stApp a {
        color: #6a0dad !important;
    }
    .stButton>button {
        background-color: #6a0dad !important;
        color: white !important;
        border-radius: 20px !important;
        padding: 4px 25px !important;
        font-weight: 600 !important;
    }
    .stSuccess {
        background-color: rgba(106, 13, 173, 0.2) !important;
        border-left-color: #6a0dad !important;
    }
    /* Card styling */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f9f9f9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6a0dad !important;
        color: white !important;
    }
    /* File uploader styling */
    .stFileUploader div[data-testid="stFileUploadDropzone"] {
        border-color: #6a0dad;
        border-style: dashed;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f9f5ff 0%, #ffffff 100%);
        border-right: 1px solid #f0e6ff;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #6a0dad;
        font-weight: 600;
    }
    </style>''',
    unsafe_allow_html=True
)

# Main header with IMO Health logo & app title
# Centered title with IMO Health purple
st.markdown("<h1 style='text-align:center; color:#6a0dad; margin:0 0 20px 0; font-family:Raleway, sans-serif; font-weight:600;'>KM Analyzer</h1>", unsafe_allow_html=True)

# Sidebar with app description
# Add logo to sidebar
logo_path = "assets/im_logo2.png"
st.sidebar.image(logo_path, width=180)

st.sidebar.markdown("""
## 📈 KM Analyzer
**Survival & image analysis**  
- Upload plot images  
- Query with natural-language  
""", unsafe_allow_html=True)

if not api_key:
    st.error("API key not found in .streamlit/secrets.toml")
    st.stop()

analyzer = KMAnalyzer(api_key)

# Wrap content in a card
st.markdown('<div class="card">', unsafe_allow_html=True)

# Create tabs
#tab1, tab3, tab2 = st.tabs(["IMO Current Solution", "(LLM Only) GPT-4o Solution", "Survival Probability"])
tab1, tab2 = st.tabs(["IMO Current Solution", "(LLM Only) GPT-4o Solution"])

with tab1:
    st.subheader("📈 IMO Current Solution")
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        curve_img_file = st.file_uploader("Upload survival curve image", type=["png", "jpg", "jpeg"], key="curve_upload")
    with col2:
        curve_q = st.text_input("Your Question", placeholder="e.g. What is the survival rate at 12 months?", key="curve_question")
        if st.button("Analyze Curve", key="curve"):
            if not curve_q:
                st.warning("Please enter a question.")
            elif not curve_img_file:
                st.warning("Please upload a curve image.")
            else:
                curve_img = Image.open(curve_img_file).convert("RGB")
                try:
                    st.image(curve_img, use_container_width=True)
                except TypeError:
                    # Fallback for older Streamlit versions
                    st.image(curve_img, use_column_width=True)
                
                # Instead of extracting data from the image, we'll use pre-extracted Excel files
                # st.info("💾 Using pre-extracted data from Excel files...")
                
                # Extract the curve name and study name from the question using Claude
                with st.spinner("Analyzing your question..."):
                    # Use the existing extract_survival_params function to get curve name
                    params = analyzer.extract_survival_params(curve_q)
                    
                    # Extract study name (e.g., PALOMA-1, PALOMA-2)
                    study_name = analyzer.extract_study_name(curve_q)
                    # st.write(f"Detected study: **{study_name}**")
                    
                    if params and 'curve_name' in params:
                        curve_name = params.get('curve_name', '').strip().lower()
                        # st.write(f"Looking for data related to: '{curve_name}'")
                        
                        # Determine which Excel file to use based on the curve name and study name
                        excel_files = {}
                        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
                        
                        # Create data directory if it doesn't exist
                        if not os.path.exists(data_dir):
                            os.makedirs(data_dir)
                            # st.warning(f"Created data directory at {data_dir}. Please add your Excel files there.")
                        
                        # List available Excel files
                        for file in os.listdir(data_dir):
                            # Skip temporary Excel lock files
                            if file.startswith('~$'):
                                continue
                            if file.lower().endswith(('.xlsx', '.xls')):
                                excel_files[file.lower()] = os.path.join(data_dir, file)
                        
                        # Find the best matching Excel file
                        matching_file = None
                        if excel_files:
                            # First priority: match both study name and curve name
                            if study_name:
                                study_pattern = study_name.lower().replace('-', '').replace(' ', '')
                                for file_name, file_path in excel_files.items():
                                    if study_pattern in file_name.lower().replace('-', '').replace(' ', '') and curve_name in file_name.lower():
                                        matching_file = file_path
                                        # st.success(f"Found file matching both study '{study_name}' and treatment '{curve_name}'")
                                        break
                            
                            # Second priority: match study name only
                            if not matching_file and study_name:
                                study_pattern = study_name.lower().replace('-', '').replace(' ', '')
                                for file_name, file_path in excel_files.items():
                                    if study_pattern in file_name.lower().replace('-', '').replace(' ', ''):
                                        matching_file = file_path
                                        # st.success(f"Found file matching study '{study_name}'")
                                        break
                            
                            # Third priority: match curve name only
                            if not matching_file:
                                for file_name, file_path in excel_files.items():
                                    if curve_name in file_name.lower():
                                        matching_file = file_path
                                        # st.success(f"Found file matching treatment '{curve_name}'")
                                        break
                            
                            # Last resort: use the first file
                            if not matching_file and excel_files:
                                first_file = list(excel_files.keys())[0]
                                matching_file = excel_files[first_file]
                                # st.warning(f"No exact match found. Using data from '{first_file}' instead.")
                    
                        if matching_file:
                            # st.success(f"Using data from: {os.path.basename(matching_file)}")
                            # Load the Excel file
                            try:
                                # Read Excel file using header=1 to correctly parse column names
                                curve_df = pd.read_excel(matching_file, header=1)
                                
                                # Standardize column names and data types (similar to tab2)
                                curve_df.columns = [str(c).strip() for c in curve_df.columns]
                                
                                # Debug the column names
                                # st.info(f"Original Excel columns: {list(curve_df.columns)}")
                                
                                # If 'Group' column exists, rename it to 'curve_name'
                                if 'Group' in curve_df.columns:
                                    curve_df = curve_df.rename(columns={'Group': 'curve_name'})
                                    # st.success("Found 'Group' column and renamed it to 'curve_name'")
                                
                                # Try to map common column names to our expected format
                                column_mapping = {
                                    'Time': 'time_months',
                                    'Months': 'time_months',
                                    'Month': 'time_months',
                                    'Time (months)': 'time_months',
                                    'Treatment': 'curve_name',
                                    'Arm': 'curve_name',
                                    'Survival': 'survival_prob',
                                    'Survival Prob': 'survival_prob',
                                    'Probability': 'survival_prob',
                                    'Survival (%)': 'survival_prob'
                                }
                                
                                # Apply column mapping for any matching columns
                                for old_col, new_col in column_mapping.items():
                                    if old_col in curve_df.columns:
                                        curve_df = curve_df.rename(columns={old_col: new_col})
                                
                                # If we still don't have the required columns, try to infer them
                                if 'time_months' not in curve_df.columns and curve_df.shape[1] >= 1:
                                    # Assume first column is time
                                    curve_df = curve_df.rename(columns={curve_df.columns[0]: 'time_months'})
                                
                                if 'survival_prob' not in curve_df.columns and curve_df.shape[1] >= 2:
                                    # Assume second column is survival probability
                                    curve_df = curve_df.rename(columns={curve_df.columns[1]: 'survival_prob'})
                                
                                # Check if 'curve_name' column exists, if not, add it
                                if 'curve_name' not in curve_df.columns:
                                    # Extract curve name from the file name
                                    file_basename = os.path.basename(matching_file)
                                    extracted_name = file_basename.split('.')[0]
                                    # Add as a new column
                                    curve_df['curve_name'] = extracted_name
                                
                                # Convert data types
                                curve_df['time_months'] = pd.to_numeric(curve_df['time_months'], errors='coerce')
                                curve_df['survival_prob'] = pd.to_numeric(curve_df['survival_prob'], errors='coerce')
                                
                                # Clean up curve names
                                curve_df['curve_name'] = (
                                    curve_df['curve_name'].astype(str)
                                    .str.replace(u"\xa0", " ", regex=False)
                                    .str.strip().str.lower()
                                    .str.replace(r"\s*\+\s*", "+", regex=True)
                                )
                                
                                # Debug info
                                # st.info(f"Excel file columns: {list(curve_df.columns)}")
                                # st.info(f"Data types: time_months={curve_df['time_months'].dtype}, survival_prob={curve_df['survival_prob'].dtype}")
                                
                                # Drop rows with NaN values in critical columns
                                curve_df = curve_df.dropna(subset=['time_months', 'survival_prob'])
                                
                                # Don't display the dataframe as requested
                                # st.dataframe(curve_df.head(5))
                            except Exception as e:
                                st.error(f"Error loading Excel file: {str(e)}")
                                curve_df = pd.DataFrame()
                        else:
                            st.error(f"No Excel files found in {data_dir}. Please add your pre-extracted data files.")
                            curve_df = pd.DataFrame()
                    else:
                        st.error("Could not extract curve name from your question.")
                        curve_df = pd.DataFrame()
                
                # Process the question to extract parameters
                if not curve_df.empty:
                    st.info("📊 Extracting data points...")
                    st.info("🔎 Looking up answer from extracted data...")
                    if not analyzer.is_survival_lookup_question(curve_q):
                        # If it's not a survival lookup question, use the general LLM
                        response = analyzer.ask_with_question_only(curve_q)
                        # st.success(response)
                    else:
                        # Extract parameters from the question
                        params = analyzer.extract_survival_params(curve_q)
                        if not params:
                            st.error("Could not extract parameters from your question.")
                        else:
                            try:
                                # Get time and curve name from the parameters
                                t = float(params.get("time_months"))
                                # Normalize extracted curve name, remove spaces around '+'
                                raw_curve = str(params.get("curve_name", "")).strip().lower()
                                curve = re.sub(r"\s*\+\s*", "+", raw_curve)
                                
                                # Get unique curve names from the dataframe
                                available_curves = curve_df['curve_name'].unique()
                                
                                # Find the closest matching curve name
                                if curve and available_curves.size > 0:
                                    # Try different matching strategies
                                    
                                    # 1. Try exact match
                                    exact_matches = [c for c in available_curves if c.lower() == curve.lower()]
                                    if exact_matches:
                                        matched_curve = exact_matches[0]
                                        st.success(f"Found exact match for '{curve}'")
                                    else:
                                        # 2. Try partial match (curve name is in available curve)
                                        partial_matches1 = [c for c in available_curves if curve.lower() in c.lower()]
                                        if partial_matches1:
                                            matched_curve = partial_matches1[0]
                                            # st.info(f"Found partial match: '{matched_curve}' contains '{curve}'")
                                        else:
                                            # 3. Try partial match (available curve is in curve name)
                                            partial_matches2 = [c for c in available_curves if c.lower() in curve.lower()]
                                            if partial_matches2:
                                                matched_curve = partial_matches2[0]
                                                # st.info(f"Found partial match: '{curve}' contains '{matched_curve}'")
                                            else:
                                                # 4. If still no match, use the first curve
                                                matched_curve = available_curves[0]
                                                # st.warning(f"No curve matching '{curve}' found. Using '{matched_curve}' instead.")
                                                
                                    # Look up the survival probability
                                    survival_prob = lookup_survival_at_time(curve_df, matched_curve, t)
                                    st.info(f"Raw survival probability value: {survival_prob}")
                                    
                                    # Check if we got a valid number
                                    if survival_prob is not None and pd.notna(survival_prob):
                                        # Format the result
                                        if survival_prob <= 1:
                                            prob_str = f"{survival_prob:.4f}"
                                        else:
                                            prob_str = f"{survival_prob:.1f}%"
                                            
                                        st.success(f"🎯 Survival probability: {prob_str} at {t} months for '{matched_curve}'.")
                                    else:
                                        # Try to find the closest available time point
                                        available_times = curve_df[curve_df['curve_name'] == matched_curve]['time_months'].dropna().tolist()
                                        if available_times:
                                            closest_time = min(available_times, key=lambda x: abs(x - t))
                                            closest_prob = lookup_survival_at_time(curve_df, matched_curve, closest_time)
                                            
                                            if pd.notna(closest_prob):
                                                if closest_prob <= 1:
                                                    prob_str = f"{closest_prob:.4f}"
                                                else:
                                                    prob_str = f"{closest_prob:.1f}%"
                                                    
                                                # st.success(f"🎯 Closest available survival probability: {prob_str} at {closest_time} months for '{matched_curve}'.")
                                                # st.info(f"Note: Data not available exactly at {t} months, showing closest available time point.")
                                            else:
                                                st.error(f"Could not find valid survival data for '{matched_curve}'.")
                                        else:
                                            st.error(f"No valid time points found for '{matched_curve}'.")
                                else:
                                    st.error("No curves were extracted from the image or no curve name was specified.")
                            except Exception as e:
                                st.error(f"Error processing request: {str(e)}")
                else:
                    st.error("Could not extract any curves from the image. Please try a different image.")

    st.divider()

with tab2:
    st.subheader("🔍 (LLM Only) GPT-4o Solution")
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="img_upload")
    with col2:
        iq = st.text_input("Your Question", placeholder="e.g. What's in this image?", key="img_question")
        if st.button("Ask LLM", key="image"):
            if not iq:
                st.warning("Please enter a question.")
            elif not img_file:
                st.warning("Please upload an image.")
            else:
                img = Image.open(img_file).convert("RGB")
                try:
                    st.image(img, use_container_width=True)
                except TypeError:
                    # Fallback for older Streamlit versions
                    st.image(img, use_column_width=True)
                # st.info("🧠 Analyzing image...")
                resp = analyzer.ask_with_image_and_question2(iq, img)
                st.success(resp)

# with tab2:
#     st.subheader("🔍 Survival Probability")
#     st.divider()
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         f = st.file_uploader("Upload survival data (CSV or Excel)", type=["csv", "xls", "xlsx"])
#     with col2:
#         q = st.text_input("Your Question", placeholder="e.g. What is PFS at 12 months for palbociclib?")
#         if st.button("Ask LLM", key="survival"):
#             if not q:
#                 st.warning("Please enter a question.")
#             elif not f:
#                 st.warning("Please upload survival data.")
#             else:
#                 # st.info(f"🧠 Analyzing: {q}")
#                 if not analyzer.is_survival_lookup_question(q):
#                     st.write(analyzer.ask_with_question_only(q))
#                 else:
#                     params = analyzer.extract_survival_params(q)
#                     if not params:
#                         st.error("Could not extract parameters.")
#                     else:
#                         try:
#                             t = float(params.get("time_months"))
#                             curve = re.sub(r"\s*\+\s*", "+", str(params.get("curve_name", "")).strip().lower())
#                             df = load_survival_data(f)
#                             if df is not None:
#                                 df.columns = [str(c).strip() for c in df.columns]
#                                 df = df.rename(columns={"Time": "time_months", "Group": "curve_name", "Survival Prob": "survival_prob"})
#                                 df["time_months"] = pd.to_numeric(df["time_months"], errors="coerce")
#                                 df["survival_prob"] = pd.to_numeric(df["survival_prob"], errors="coerce")
#                                 df["curve_name"] = (
#                                     df["curve_name"].astype(str)
#                                     .str.replace(u"\xa0", " ", regex=False)
#                                     .str.strip().str.lower()
#                                     .str.replace(r"\s*\+\s*", "+", regex=True)
#                                 )
#                                 cdf = df[df["curve_name"] == curve]
#                                 if cdf.empty:
#                                     st.error(f"No curve found for '{curve}'.")
#                                 else:
#                                     vt = cdf[cdf["time_months"] <= t]
#                                     if vt.empty:
#                                         st.error(f"No data at or before {t} months for '{curve}'.")
#                                     else:
#                                         row = vt.sort_values("time_months", ascending=False).iloc[0]
#                                         # Format survival probability without literal % sign
#                                         prob = row["survival_prob"]
#                                         if prob <= 1:
#                                             prob_str = f"{prob:.4f}"
#                                         else:
#                                             prob_str = f"{prob:.1f}%"
#                                         st.success(f"🎯 Survival probability: {prob_str} at {t} months for '{curve}'.")
#                         except Exception as e:
#                             st.error(f"Lookup failed: {e}")

    st.divider()

# Close the card div
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align:center; margin-top:30px; color:#666; font-size:0.8rem;'>&copy; 2025 IMO Health. All rights reserved.</div>", unsafe_allow_html=True)
