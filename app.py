import streamlit as st
from fastai.vision.all import *
from PIL import Image
import json
import pathlib
import os
import platform

# --- 1. CROSS-PLATFORM COMPATIBILITY ---
# Even with a clean model, this small check prevents "Ghost" errors
# depending on which OS the server is running.
plt = platform.system()
if plt == 'Windows': 
    pathlib.PosixPath = pathlib.WindowsPath
else: 
    pathlib.WindowsPath = pathlib.PosixPath

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_ecg_model():
    # We look for the CLEAN file name
    model_name = 'ecg_model_clean.pkl'

    if not os.path.exists(model_name):
        st.error(f"❌ Error: '{model_name}' not found. Please upload the new file to GitHub.")
        st.stop()
        
    try:
        learn = load_learner(model_name)
        return learn
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        st.stop()

learn = load_ecg_model()

# --- 4. UI LAYOUT ---
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
                try:
                    # Vision Inference
                    pred_class, pred_idx, probs = learn.predict(image)
                    confidence = float(probs[pred_idx]) * 100
                    
                    # Store results in session state
                    st.session_state.prediction = pred_class
                    st.session_state.confidence = confidence
                    st.session_state.run_llm = True
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

with col2:
    st.header("Clinical Report")
    
    if 'prediction' in st.session_state:
        # 1. DISPLAY VISION RESULTS
        pred = st.session_state.prediction
        conf = st.session_state.confidence
        
        # Color code based on severity
        color = "red" if pred in ["Inferior MI", "Anterior MI", "AFib"] else "green"
        
        st.markdown(f"### Diagnosis: <span style='color:{color}'>{pred.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Model Confidence:** {conf:.2f}%")
        st.progress(int(conf))
        
        if conf < 70:
            st.warning("⚠️ Low confidence. Clinical correlation required.")

        # 2. GENERATE LLM REPORT
        if st.session_state.get('run_llm', False):
            
            # CHECK FOR API KEY (Secrets or Sidebar)
            try:
                # Import OpenAI inside the block to avoid start-up errors
                from openai import OpenAI
                api_key = st.secrets.get("OPENAI_API_KEY", None)
                
                if not api_key:
                    st.warning("⚠️ No OpenAI API Key found in Secrets. Using Mock Report.")
                    # Mock Response for Demo
                    report = {
                        "rhythm_diagnosis": pred,
                        "clinical_significance": "This is a SIMULATED report. Add OpenAI Key for real analysis.",
                        "key_findings": ["Visual pattern matches diagnosis", "QRS morphology consistent"],
                        "immediate_action": "Verify with cardiologist.",
                        "urgency": "High"
                    }
                else:
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
