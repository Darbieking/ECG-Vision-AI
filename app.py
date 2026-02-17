import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import os
import pandas as pd

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 2. EXACT CLASS LIST ---
vocab = ['AFib', 'Anterior MI', 'Inferior MI', 'LBBB', 'Left Vent. Hypertrophy', 'Normal', 'RBBB', 'Sinus Rhythm']

# --- 3. MODEL ARCHITECTURE ---
class ECGModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Body
        self.body = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        
        # Head (Reconstructed from FastAI)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2560),
            nn.Dropout(0.25),
            nn.Linear(2560, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes, bias=False) # Bias determined dynamically below
        )
        
    def forward(self, x):
        x = self.body.forward_features(x)
        # FastAI Pooling (Avg + Max)
        x = torch.cat([x.mean(dim=(2,3)), x.amax(dim=(2,3))], dim=1)
        x = self.head(x)
        return x

# --- 4. STRICT LOADER ---
@st.cache_resource
def load_model():
    if not os.path.exists('ecg_weights.pth'):
        st.error("❌ 'ecg_weights.pth' missing. Please upload to GitHub.")
        st.stop()
        
    state_dict = torch.load('ecg_weights.pth', map_location='cpu')
    
    # 1. Detect Bias
    has_bias = any(k.endswith('8.bias') for k in state_dict.keys())
    
    # 2. Build Model
    model = ECGModel(num_classes=len(vocab))
    if has_bias:
        model.head[6] = nn.Linear(512, len(vocab), bias=True)

    # 3. Clean & Remap Keys
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'num_batches_tracked' (FastAI noise)
        if 'num_batches_tracked' in k: continue
        
        # Remap Body
        if k.startswith('0.model.'):
            new_k = k.replace('0.model.', 'body.')
            new_state_dict[new_k] = v
            
        # Remap Head
        elif k.startswith('1.'):
            parts = k.split('.')
            idx, param = parts[1], parts[2]
            if idx == '2': new_idx = '0'
            elif idx == '4': new_idx = '2'
            elif idx == '6': new_idx = '4'
            elif idx == '8': new_idx = '6'
            else: continue
            new_state_dict[f"head.{new_idx}.{param}"] = v

    # 4. Strict Load (Critical Check)
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        # If strict loading fails, print the mismatch for debugging
        st.error(f"❌ Weight Mismatch: {e}")
        st.stop()
        
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model Init Error: {e}")
    st.stop()

# --- 5. IMAGE PREPROCESSING ---
def process_image(img):
    if img.mode != 'RGB': img = img.convert('RGB')
    
    # We use a padding resize to avoid squishing the ECG signals
    t = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return t(img).unsqueeze(0)

# --- 6. UI ---
st.title("🫀 ECG AI Doctor (Debug Mode)")
st.caption("Now showing top 3 predictions to diagnose mapping issues.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload 12-Lead ECG", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Trace", use_container_width=True)
        
        if st.button("Analyze Tracing"):
            with st.spinner("Analyzing..."):
                img_t = process_image(image)
                with torch.no_grad():
                    out = model(img_t)
                    probs = torch.softmax(out, dim=1)
                
                # Get Top 3 Predictions
                top3_prob, top3_idx = torch.topk(probs, 3)
                
                # Store in session
                st.session_state.top3_results = []
                for i in range(3):
                    label = vocab[top3_idx[0][i].item()]
                    score = top3_prob[0][i].item() * 100
                    st.session_state.top3_results.append((label, score))
                
                st.session_state.run_llm = True

with col2:
    if 'top3_results' in st.session_state:
        results = st.session_state.top3_results
        top_pred, top_score = results[0]
        
        # 1. Primary Diagnosis
        color = "green" if top_pred in ["Normal", "Sinus Rhythm"] else "red"
        st.markdown(f"### Diagnosis: <span style='color:{color}'>{top_pred}</span>", unsafe_allow_html=True)
        st.progress(int(top_score))
        
        # 2. Probability Table (Debug Info)
        st.markdown("#### 🔍 Probability Breakdown")
        df = pd.DataFrame(results, columns=["Condition", "Confidence (%)"])
        st.table(df)

        # 3. Warning on Input Type
        if top_score < 50:
            st.warning("⚠️ Low confidence. Ensure you are uploading a **12-lead grid**, not a rhythm strip.")

        # 4. LLM
        if st.session_state.get('run_llm', False):
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    prompt = f"Explain the ECG finding: {top_pred} (Confidence: {top_score:.1f}%)."
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.info(resp.choices[0].message.content)
                except: pass
            st.session_state.run_llm = False
