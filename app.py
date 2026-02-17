import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import os

st.set_page_config(page_title="ECG Calibration", layout="wide")
st.title("🔧 ECG Model Calibration")
st.markdown("""
**Instructions:**
1. Upload a known **Normal** ECG. Note the **"Predicted Class Index"**.
2. Upload a known **AFib** ECG. Note the Index.
3. Upload a known **Inferior MI** ECG. Note the Index.
4. **Post these numbers in the chat.**
""")

# --- MODEL DEFINITION ---
class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2560),
            nn.Dropout(0.25),
            nn.Linear(2560, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 8, bias=False) # Assuming 8 classes
        )
        
    def forward(self, x):
        x = self.body.forward_features(x)
        x = torch.cat([x.mean(dim=(2,3)), x.amax(dim=(2,3))], dim=1)
        x = self.head(x)
        return x

@st.cache_resource
def load_model():
    if not os.path.exists('ecg_weights.pth'):
        st.error("Missing weights file.")
        st.stop()
    
    state_dict = torch.load('ecg_weights.pth', map_location='cpu')
    
    # Auto-detect bias
    has_bias = any(k.endswith('8.bias') for k in state_dict.keys())
    
    model = ECGModel()
    if has_bias:
        model.head[6] = nn.Linear(512, 8, bias=True)
        
    # Remap keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'num_batches_tracked' in k: continue
        if k.startswith('0.model.'):
            new_state_dict[k.replace('0.model.', 'body.')] = v
        elif k.startswith('1.'):
            parts = k.split('.')
            idx = parts[1]
            if idx == '2': new_idx = '0'
            elif idx == '4': new_idx = '2'
            elif idx == '6': new_idx = '4'
            elif idx == '8': new_idx = '6'
            else: continue
            new_state_dict[f"head.{new_idx}.{parts[2]}"] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

model = load_model()

# --- PREDICTION ---
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300)
    
    # Preprocessing
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = t(image).unsqueeze(0)
    
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, 1)
        
    st.info(f"### 🔢 Predicted Class Index: {idx.item()}")
    st.write(f"Confidence: {conf.item()*100:.2f}%")
