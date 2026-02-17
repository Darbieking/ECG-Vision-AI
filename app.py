import streamlit as st
from fastai.vision.all import *
import pathlib
import os
import platform

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 2. COMPATIBILITY PATCH ---
# Even with a clean model, this tiny patch prevents "Ghost" errors on different OSs.
# We apply it universally to be safe.
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    # We look for the CLEAN file name
    model_name = 'ecg_model_clean.pkl'

    if not os.path.exists(model_name):
        st.error(f"❌ Error: '{model_name}' not found. Please upload the new file from Colab.")
        st.stop()
        
    try:
        learn = load_learner(model_name)
        return learn
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        st.stop()

learn = load_model()

# --- 4. UI LAYOUT ---
st.title("🫀 AI-Powered ECG Interpreter")
st.markdown("Upload a 12-lead ECG image (or rhythm strip) for instant analysis.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded ECG", use_column_width=True)
    
    if st.button("Analyze Tracing"):
        with st.spinner("Analyzing..."):
            pred, idx, probs = learn.predict(image)
            confidence = float(probs[idx]) * 100
            
            st.success(f"Prediction: {pred.upper()} ({confidence:.1f}%)")
            
            # (Optional) LLM Integration Code goes here if you want the full report
