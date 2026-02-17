import streamlit as st
import pathlib
import platform
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 2. UNIVERSAL MODEL LOADER ---
@st.cache_resource
def load_ecg_model():
    """
    Robust loader that works on both Windows and Linux,
    regardless of where the model was trained.
    """
    from fastai.vision.all import load_learner
    
    model_path = 'ecg_model_v1.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Critical Error: Model file '{model_path}' not found!")
        st.stop()

    # STRATEGY 1: Try Standard Load (Best Case)
    try:
        learn = load_learner(model_path)
        return learn
    except Exception as e_standard:
        # If standard load fails, we try patching based on OS
        
        # STRATEGY 2: Windows Patch (Loading Linux model on Windows)
        if platform.system() == 'Windows':
            try:
                # Force PosixPath to be WindowsPath
                pathlib.PosixPath = pathlib.WindowsPath
                learn = load_learner(model_path)
                return learn
            except Exception as e_win:
                st.error(f"❌ Windows Patch Failed: {e_win}")
        
        # STRATEGY 3: Linux Patch (Loading Windows model on Cloud/Linux)
        else: # Linux/Mac
            try:
                # Force WindowsPath to be PosixPath
                pathlib.WindowsPath = pathlib.PosixPath
                learn = load_learner(model_path)
                return learn
            except Exception as e_linux:
                st.error(f"❌ Linux Patch Failed: {e_linux}")
        
        # If we reach here, nothing worked
        st.error("❌ All loading attempts failed.")
        st.error(f"Original Error: {e_standard}")
        st.stop()

# Load the model immediately
learn = load_ecg_model()

# --- 3. UI LAYOUT ---
# Only import FastAI libraries AFTER patching
from fastai.vision.all import *
from PIL import Image
import json

st.title("🫀 AI-Powered ECG Interpreter")
st.markdown("Upload a 12-lead ECG image (or rhythm strip) for instant analysis.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload ECG")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG", use_column_width=True)
        
        if st.button("Analyze Tracing"):
            with st.spinner("Analyzing rhythm patterns..."):
                try:
                    pred_class, pred_idx, probs = learn.predict(image)
                    confidence = float(probs[pred_idx]) * 100
                    
                    st.session_state.prediction = pred_class
                    st.session_state.confidence = confidence
                    st.session_state.run_llm = True
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

with col2:
    st.header("Clinical Report")
    
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        conf = st.session_state.confidence
        
        # Color coding
        color = "red" if pred in ["Inferior MI", "Anterior MI", "AFib"] else "green"
        
        st.markdown(f"### Diagnosis: <span style='color:{color}'>{pred.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Model Confidence:** {conf:.2f}%")
        st.progress(int(conf))
        
        if conf < 70:
            st.warning("⚠️ Low confidence. Clinical correlation required.")

        # LLM SECTION
        if st.session_state.get('run_llm', False):
            # Check for API Key in Secrets OR Environment Variable
            api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
            
            if not api_key:
                st.warning("⚠️ No API Key found. Using Mock Report.")
                report = {
                    "rhythm_diagnosis": pred,
                    "clinical_significance": "This is a SIMULATED report. Add OpenAI Key for real analysis.",
                    "key_findings": ["Visual pattern matches diagnosis", "QRS morphology consistent"],
                    "immediate_action": "Verify with cardiologist.",
                    "urgency": "High"
                }
            else:
                try:
                    from openai import OpenAI
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

            if report:
                st.subheader("📋 Physician's Note")
                st.info(f"**Significance:** {report.get('clinical_significance')}")
                st.write("**Key Findings:**")
                for item in report.get('key_findings', []):
                    st.write(f"- {item}")
                st.error(f"**Action:** {report.get('immediate_action')}")
