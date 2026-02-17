import streamlit as st
import pathlib
import platform
import os

# --- 1. BRUTE FORCE PATH PATCH ---
# This must run before ANYTHING else.
# It forces the system to treat all paths as valid for the current OS.
if os.name == 'nt': # Windows
    pathlib.PosixPath = pathlib.WindowsPath
else: # Linux / Streamlit Cloud
    pathlib.WindowsPath = pathlib.PosixPath

# --- 2. NOW IMPORT FASTAI ---
from fastai.vision.all import *
from PIL import Image
import json

# --- 3. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
    model_name = 'ecg_model_final.pkl'

    if not os.path.exists(model_name):
        st.error(f"❌ Error: '{model_name}' not found. Please upload the 'lobotomized' file from Colab.")
        st.stop()
        
    try:
        # Load the model
        learn = load_learner(model_name)
        return learn
    except Exception as e:
        st.error(f"❌ CRITICAL LOAD ERROR: {e}")
        st.stop()

learn = load_model()

# --- 5. UI LAYOUT ---
st.title("🫀 AI-Powered ECG Interpreter")
st.markdown("Upload a 12-lead ECG image for instant analysis.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload ECG")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG", use_column_width=True)
        
        if st.button("Analyze Tracing"):
            with st.spinner("Analyzing..."):
                try:
                    pred_class, pred_idx, probs = learn.predict(image)
                    confidence = float(probs[pred_idx]) * 100
                    
                    st.session_state.prediction = pred_class
                    st.session_state.confidence = confidence
                    st.session_state.run_llm = True
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

with col2:
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        conf = st.session_state.confidence
        color = "red" if pred in ["Inferior MI", "Anterior MI", "AFib"] else "green"
        
        st.markdown(f"### Diagnosis: <span style='color:{color}'>{pred.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {conf:.1f}%")
        st.progress(int(conf))

        # LLM SECTION
        if st.session_state.get('run_llm', False):
            try:
                from openai import OpenAI
                api_key = st.secrets.get("OPENAI_API_KEY", None)
                
                if not api_key:
                    st.info("ℹ️ Add OpenAI API Key to Secrets for full report.")
                else:
                    client = OpenAI(api_key=api_key)
                    # (Simplified prompt for stability)
                    prompt = f"Explain this ECG diagnosis: {pred} (Confidence: {conf:.1f}%) in one paragraph."
                    
                    st.subheader("📋 Physician's Note")
                    with st.spinner("Generating report..."):
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"LLM Error: {e}")
            
            st.session_state.run_llm = False
