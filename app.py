import streamlit as st
import pathlib
import platform
import os

# --- 1. THE CRITICAL FIX ---
# We must apply this patch BEFORE importing fastai.
# This forces Linux to accept 'WindowsPath' objects from your model file.
posix_backup = pathlib.PosixPath
try:
    pathlib.WindowsPath = pathlib.PosixPath
except:
    pass

# --- 2. IMPORTS ---
# Now it is safe to import fastai
from fastai.vision.all import *
from PIL import Image
import json

# --- 3. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_ecg_model():
    model_path = 'ecg_model_v1.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Critical Error: Model file '{model_path}' not found!")
        st.stop()

    try:
        # Load the model
        learn = load_learner(model_path)
        return learn
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load immediately
learn = load_ecg_model()

# --- 5. UI LAYOUT ---
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
                st.warning("⚠️ No OpenAI API Key found. Using Mock Report.")
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
