import streamlit as st
from fastai.vision.all import *
from PIL import Image
import json
from openai import OpenAI

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_ecg_model():
    """Loads the FastAI model. Cached for speed."""
    # Fix for PosixPath issues if moving between Windows/Linux
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
    try:
        learn = load_learner('ecg_model_v1.pkl')
    except:
        # If running on Linux (Streamlit Cloud), revert to Posix
        pathlib.PosixPath = temp
        learn = load_learner('ecg_model_v1.pkl')
        
    return learn

try:
    learn = load_ecg_model()
    model_status = "✅ Model Loaded"
except Exception as e:
    model_status = f"❌ Error: {e}"

# --- 3. UI LAYOUT ---
st.title("🫀 AI-Powered ECG Interpreter")
st.markdown("Upload a 12-lead ECG image (or rhythm strip) for instant analysis.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload ECG")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG", use_column_width=True)
        
        # PREDICTION BUTTON
        if st.button("Analyze Tracing"):
            with st.spinner("Analyzing rhythm patterns..."):
                # FastAI Inference
                pred_class, pred_idx, probs = learn.predict(image)
                confidence = float(probs[pred_idx]) * 100
                
                # Store results in session state to persist
                st.session_state.prediction = pred_class
                st.session_state.confidence = confidence
                st.session_state.run_llm = True

with col2:
    st.header("Clinical Report")
    
    if 'prediction' in st.session_state:
        # 1. DISPLAY VISION RESULTS
        pred = st.session_state.prediction
        conf = st.session_state.confidence
        
        # Color code based on severity (Simple logic for demo)
        color = "red" if pred in ["Inferior MI", "Anterior MI", "AFib"] else "green"
        
        st.markdown(f"### Diagnosis: <span style='color:{color}'>{pred.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Model Confidence:** {conf:.2f}%")
        st.progress(int(conf))
        
        if conf < 70:
            st.warning("⚠️ Low confidence. Clinical correlation required.")

        # 2. GENERATE LLM REPORT
        if st.session_state.get('run_llm', False):
            
            # CHECK FOR API KEY (Secrets or Sidebar)
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            
            if not api_key:
                st.warning("⚠️ No OpenAI API Key found in Secrets. Using Mock Report.")
                # Mock Response for Demo
                report = {
                    "rhythm_diagnosis": pred,
                    "clinical_significance": "Requires API Key for full analysis.",
                    "key_findings": ["Pattern matched visual features"],
                    "immediate_action": "Configure Secrets on Streamlit Cloud.",
                    "urgency": "High"
                }
            else:
                # REAL LLM CALL
                try:
                    client = OpenAI(api_key=api_key)
                    prompt = f"""
                    You are a cardiologist. Analyze this diagnosis: {pred} (Confidence: {conf:.1f}%).
                    Return valid JSON with fields: rhythm_diagnosis, clinical_significance, key_findings (list), immediate_action, urgency.
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}
                    )
                    report = json.loads(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"LLM Error: {e}")
                    report = None

            # 3. RENDER REPORT
            if report:
                st.subheader("📋 Physician's Note")
                st.info(f"**Significance:** {report.get('clinical_significance')}")
                
                st.write("**Key Findings:**")
                for item in report.get('key_findings', []):
                    st.write(f"- {item}")
                
                st.error(f"**Action:** {report.get('immediate_action')}")
                st.caption(f"Urgency Level: {report.get('urgency')}")
                
            st.session_state.run_llm = False # Don't re-run on every refresh