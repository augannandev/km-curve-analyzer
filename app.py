import streamlit as st
st.set_page_config(page_title="KM Analyzer", page_icon="📈", layout="wide")
import anthropic
import json, base64, io, re, warnings
import pandas as pd
from PIL import Image

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

    def ask_with_image_and_question(self, question: str, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        try:
            res = self.client.messages.create(
                model="claude-3-7-sonnet-20250219", max_tokens=1024, #claude-3-5-haiku-20241022 #claude-3-7-sonnet-20250219 #claude-3-opus-20240229
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
    .stApp * {
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
    </style>''',
    unsafe_allow_html=True
)

# Sidebar with app description
st.sidebar.markdown("""
## 📈 KM Analyzer
**Survival & image analysis**  
- Upload clinical data or plot images  
- Query with natural-language  
""", unsafe_allow_html=True)

# Centered, styled main title
st.markdown(
    "<h1 style='text-align:center; color:#2E86AB; margin-top:0;'>KM Analyzer</h1>",
    unsafe_allow_html=True
)

#st.title("KM Analyzer App")

if not api_key:
    st.error("API key not found in .streamlit/secrets.toml")
    st.stop()

analyzer = KMAnalyzer(api_key)

tab1, tab2 = st.tabs(["Survival Probability", "Image Analysis"])

with tab1:
    st.subheader("🔍 Survival Probability")
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        f = st.file_uploader("Upload survival data (CSV or Excel)", type=["csv", "xls", "xlsx"])
    with col2:
        q = st.text_input("Your Question", placeholder="e.g. What is PFS at 12 months for palbociclib?")
        if st.button("Ask LLM", key="survival"):
            if not q:
                st.warning("Please enter a question.")
            elif not f:
                st.warning("Please upload survival data.")
            else:
                st.info(f"🧠 Analyzing: {q}")
                if not analyzer.is_survival_lookup_question(q):
                    st.write(analyzer.ask_with_question_only(q))
                else:
                    params = analyzer.extract_survival_params(q)
                    if not params:
                        st.error("Could not extract parameters.")
                    else:
                        try:
                            t = float(params.get("time_months"))
                            curve = re.sub(r"\s*\+\s*", "+", str(params.get("curve_name", "")).strip().lower())
                            df = load_survival_data(f)
                            if df is not None:
                                df.columns = [str(c).strip() for c in df.columns]
                                df = df.rename(columns={"Time": "time_months", "Group": "curve_name", "Survival Prob": "survival_prob"})
                                df["time_months"] = pd.to_numeric(df["time_months"], errors="coerce")
                                df["survival_prob"] = pd.to_numeric(df["survival_prob"], errors="coerce")
                                df["curve_name"] = (
                                    df["curve_name"].astype(str)
                                    .str.replace(u"\xa0", " ", regex=False)
                                    .str.strip().str.lower()
                                    .str.replace(r"\s*\+\s*", "+", regex=True)
                                )
                                cdf = df[df["curve_name"] == curve]
                                if cdf.empty:
                                    st.error(f"No curve found for '{curve}'.")
                                else:
                                    vt = cdf[cdf["time_months"] <= t]
                                    if vt.empty:
                                        st.error(f"No data at or before {t} months for '{curve}'.")
                                    else:
                                        row = vt.sort_values("time_months", ascending=False).iloc[0]
                                        # Format survival probability without literal % sign
                                        prob = row["survival_prob"]
                                        if prob <= 1:
                                            prob_str = f"{prob:.4f}"
                                        else:
                                            prob_str = f"{prob:.1f}%"
                                        st.success(f"🎯 Survival probability: {prob_str} at {t} months for '{curve}'.")
                        except Exception as e:
                            st.error(f"Lookup failed: {e}")

    st.divider()

with tab2:
    st.subheader("🖼️ Image Analysis")
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        img_file = st.file_uploader("Upload KM plot image", type=["png", "jpg", "jpeg", "bmp", "gif"])
    with col2:
        iq = st.text_input("Image Question", placeholder="e.g. What is the median survival?")
        if st.button("Analyze Image", key="image"):
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
                st.info("🧠 Analyzing image...")
                resp = analyzer.ask_with_image_and_question(iq, img)
                st.success(resp)

    st.divider()
