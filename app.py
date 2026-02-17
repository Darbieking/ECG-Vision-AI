import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
import os
import pandas as pd

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 2. CLASS LIST ---
vocab = ['AFib', 'Anterior MI', 'Inferior MI', 'LBBB', 'Left Vent. Hypertrophy', 'Normal', 'RBBB', 'Sinus Rhythm']

# --- 3. MODEL ARCHITECTURE ---
class ECGModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.body = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2560),
            nn.Dropout(0.25),
            nn.Linear(2560, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes, bias=False)
        )
        
    def forward(self, x):
        x = self.body.forward_features(x)
        x = torch.cat([x.mean(dim=(2,3)), x.amax(dim=(2,3))], dim=1)
        x = self.head(x)
        return x

@st.cache_resource
def load_model():
    if not os.path.exists('ecg_weights.pth'): return None
    state_dict = torch.load('ecg_weights.pth', map_location='cpu')
    
    # Auto-Bias Detection
    has_bias = any(k.endswith('8.bias') for k in state_dict.keys())
    model = ECGModel(num_classes=len(vocab))
    if has_bias: model.head[6] = nn.Linear(512, len(vocab), bias=True)

    # Key Remapping
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'num_batches_tracked' in k: continue
        if k.startswith('0.model.'):
            new_state_dict[k.replace('0.model.', 'body.')] = v
        elif k.startswith('1.'):
            parts = k.split('.')
            idx = parts[1]
            if idx == '2': n = '0'
            elif idx == '4': n = '2'
            elif idx == '6': n = '4'
            elif idx == '8': n = '6'
            else: continue
            new_state_dict[f"head.{n}.{parts[2]}"] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

model = load_model()

# --- 4. LIVE PREPROCESSING UI ---
st.title("🫀 ECG AI Doctor (Live Calibration)")

if model is None:
    st.error("❌ 'ecg_weights.pth' not found in GitHub.")
    st.stop()

# SIDEBAR CONTROLS
st.sidebar.header("⚙️ Image Settings")
resize_mode = st.sidebar.radio("Resize Mode", ["Squish (Standard)", "Pad (Black Bars)", "Crop (Center)"])
contrast_level = st.sidebar.slider("Contrast Boost", 0.5, 3.0, 1.5, help="Increase to make lines darker")
grayscale = st.sidebar.checkbox("Convert to Grayscale", value=False)
invert = st.sidebar.checkbox("Invert Colors", value=False)

def process_image_live(img):
    # 1. Color Corrections
    if grayscale: img = img.convert("L").convert("RGB")
    else: img = img.convert("RGB")
    
    if invert: img = ImageOps.invert(img)
    
    # Contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_level)

    # 2. Resizing Logic
    target = (224, 224)
    if resize_mode == "Squish (Standard)":
        img = img.resize(target)
    elif resize_mode == "Pad (Black Bars)":
        ratio = 224 / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size)
        new_img = Image.new("RGB", target, (0, 0, 0))
        new_img.paste(img, ((224-new_size[0])//2, (224-new_size[1])//2))
        img = new_img
    elif resize_mode == "Crop (Center)":
        # Resize shortest side to 224 then crop
        ratio = 224 / min(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size)
        # Center crop
        left = (img.width - 224)/2
        top = (img.height - 224)/2
        img = img.crop((left, top, left+224, top+224))

    # 3. Visualize what the AI sees
    st.sidebar.image(img, caption="What the AI sees", width=150)
    
    # 4. Normalize
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return t(img).unsqueeze(0)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload 12-Lead ECG", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Upload", use_container_width=True)
        
        if st.button("Analyze Tracing"):
            with st.spinner("Processing..."):
                img_t = process_image_live(image)
                with torch.no_grad():
                    out = model(img_t)
                    probs = torch.softmax(out, dim=1)
                
                top3_prob, top3_idx = torch.topk(probs, 3)
                
                st.session_state.results = []
                for i in range(3):
                    idx = top3_idx[0][i].item()
                    label = vocab[idx]
                    score = top3_prob[0][i].item() * 100
                    st.session_state.results.append((label, score))

with col2:
    if 'results' in st.session_state:
        results = st.session_state.results
        top_pred, top_score = results[0]
        
        # Color Logic
        color = "green" if top_pred in ["Normal", "Sinus Rhythm"] else "red"
        st.markdown(f"### Diagnosis: <span style='color:{color}'>{top_pred}</span>", unsafe_allow_html=True)
        st.progress(int(top_score))
        
        st.markdown("#### 📊 Confidence Breakdown")
        df = pd.DataFrame(results, columns=["Diagnosis", "Confidence %"])
        st.table(df)

        # Advice
        if top_score < 50:
             st.info("💡 **Tip:** Try increasing 'Contrast Boost' in the sidebar or switching Resize Mode.")
