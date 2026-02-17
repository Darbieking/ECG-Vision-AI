import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import json
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 2. DEFINE THE CLASSES ---
# ⚠️ REPLACE THIS LIST with the one you copied from Colab!
# It must be in the exact same order.
vocab = ['AFib', 'Anterior MI', 'Inferior MI', 'LBBB', 'Left Vent. Hypertrophy', 'Normal', 'RBBB', 'Sinus Rhythm']
# (I included common ones above, but paste yours to be safe)

# --- 3. BUILD THE MODEL MANUALLY ---
@st.cache_resource
def load_model():
    # A. Create the empty architecture (EfficientNet-B0)
    # This matches exactly what we trained in FastAI
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(vocab))
    
    # B. Load the weights
    if not os.path.exists('ecg_weights.pth'):
        st.error("❌ Error: 'ecg_weights.pth' not found. Please upload it to GitHub.")
        st.stop()
        
    # Load weights to CPU (Safe for Streamlit Cloud)
    state_dict = torch.load('ecg_weights.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # C. Set to evaluation mode
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ critical error building model: {e}")
    st.stop()

# --- 4. IMAGE PREPROCESSING ---
# Since we aren't using FastAI's transform pipeline, we do a standard resize/normalize
def process_image(image):
    # Ensure 3 channels (RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), # EfficientNet standard size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0) # Add batch dimension

# --- 5. UI LAYOUT ---
st.title("🫀 AI-Powered ECG Interpreter")
st.markdown("Upload a 12-lead ECG image for instant analysis.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload ECG", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG", use_column_width=True)
        
        if st.button("Analyze Tracing"):
            with st.spinner("Analyzing pattern..."):
                try:
                    # 1. Preprocess
                    img_tensor = process_image(image)
                    
                    # 2. Inference
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # 3. Get Top Prediction
                    confidence, idx = torch.max(probs, 1)
                    pred_label = vocab[idx.item()]
                    conf_score = confidence.item() * 100
                    
                    st.session_state.prediction = pred_label
                    st.session_state.confidence = conf_score
                    st.session_state.run_llm = True
                    
                except Exception as e:
                    st.error(f"Inference Error: {e}")

with col2:
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        conf = st.session_state.confidence
        
        color = "red" if "MI" in pred or "AFib" in pred else "green"
        
        st.markdown(f"### Diagnosis: <span style='color:{color}'>{pred}</span>", unsafe_allow_html=True)
        st.progress(int(conf))
        st.caption(f"Confidence: {conf:.1f}%")

        # LLM Section
        if st.session_state.get('run_llm', False):
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    prompt = f"Explain this ECG diagnosis: {pred} (Confidence: {conf:.1f}%) in one paragraph."
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.info(response.choices[0].message.content)
                except:
                    st.warning("Could not generate report.")
            else:
                st.warning("Add OpenAI API Key to Secrets for full report.")
            
            st.session_state.run_llm = False
