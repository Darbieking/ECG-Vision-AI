import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 2. YOUR EXACT CLASS LIST ---
# This matches the training data perfectly.
vocab = ['AFib', 'Anterior MI', 'Inferior MI', 'LBBB', 'Left Vent. Hypertrophy', 'Normal', 'RBBB', 'Sinus Rhythm']

# --- 3. FLEXIBLE MODEL ARCHITECTURE ---
class ECGModel(nn.Module):
    def __init__(self, num_classes, use_final_bias=True):
        super().__init__()
        # Body (EfficientNet-B0)
        # We load the base model without the classifier
        self.body = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        
        # Head (FastAI Custom Structure)
        # This matches the layers in your saved weights file exactly.
        self.head = nn.Sequential(
            nn.BatchNorm1d(2560),           # head.0
            nn.Dropout(0.25),
            nn.Linear(2560, 512, bias=False), # head.2
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),            # head.4
            nn.Dropout(0.5),
            nn.Linear(512, num_classes, bias=use_final_bias) # head.6 (Output)
        )
        
    def forward(self, x):
        # 1. Get features from body
        x = self.body.forward_features(x)
        
        # 2. FastAI ConcatPooling (Global Avg + Global Max)
        # [Batch, 1280, 7, 7] -> [Batch, 2560]
        avg_pool = torch.mean(x, dim=(2,3))
        max_pool = torch.amax(x, dim=(2,3))
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # 3. Pass through head
        x = self.head(x)
        return x

# --- 4. THE SMART LOADER ---
@st.cache_resource
def load_model():
    if not os.path.exists('ecg_weights.pth'):
        st.error("❌ Error: 'ecg_weights.pth' not found. Please upload it to GitHub.")
        st.stop()
        
    # A. Load raw weights
    state_dict = torch.load('ecg_weights.pth', map_location='cpu')
    
    # B. Auto-Detect Bias
    # FastAI models sometimes have a bias in the final layer, sometimes not.
    # We check the file to see if '1.8.bias' (the output bias) exists.
    has_bias = False
    for k in state_dict.keys():
        if k.endswith('8.bias'): 
            has_bias = True
            break
            
    # C. Build the correct skeleton
    model = ECGModel(num_classes=len(vocab), use_final_bias=has_bias)
    
    # D. Key Mapping (FastAI -> PyTorch)
    # We rename the keys in the file to match our manual model class
    new_state_dict = {}
    for key, value in state_dict.items():
        # Body Mapping: "0.model.layer" -> "body.layer"
        if key.startswith('0.model.'):
            new_key = key.replace('0.model.', 'body.')
            new_state_dict[new_key] = value
        
        # Head Mapping: "1.layer" -> "head.index"
        elif key.startswith('1.'):
            parts = key.split('.')
            idx = parts[1] # e.g., '2', '4', '8'
            param = parts[2] # weight or bias
            
            # Map FastAI indices to our Sequential indices
            if idx == '2': new_idx = '0'   # BN
            elif idx == '4': new_idx = '2' # Linear
            elif idx == '6': new_idx = '4' # BN
            elif idx == '8': new_idx = '6' # Linear (Output)
            else: continue
            
            new_key = f"head.{new_idx}.{param}"
            new_state_dict[new_key] = value
            
    # E. Load Weights (Strict=False ignores minor version mismatches like 'num_batches_tracked')
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Critical Error Building Model: {e}")
    st.stop()

# --- 5. IMAGE PREPROCESSING ---
def process_image(img):
    if img.mode != 'RGB': img = img.convert('RGB')
    
    # Standard ImageNet normalization
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return t(img).unsqueeze(0) # Add batch dimension

# --- 6. UI & INFERENCE ---
st.title("🫀 AI-Powered ECG Interpreter")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG", use_column_width=True)
        
        if st.button("Analyze Tracing"):
            with st.spinner("Analyzing rhythm patterns..."):
                try:
                    # Inference
                    img_t = process_image(image)
                    with torch.no_grad():
                        out = model(img_t)
                        probs = torch.softmax(out, dim=1)
                        conf, idx = torch.max(probs, 1)
                    
                    # Get Result
                    pred_label = vocab[idx.item()]
                    conf_score = conf.item() * 100
                    
                    # Store in Session State
                    st.session_state.prediction = pred_label
                    st.session_state.confidence = conf_score
                    st.session_state.run_llm = True
                    
                except Exception as e:
                    st.error(f"Prediction Failed: {e}")

with col2:
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        conf = st.session_state.confidence
        
        # Dynamic Color Logic
        if pred == "Normal" or pred == "Sinus Rhythm":
            color = "green"
        else:
            color = "red"
            
        st.markdown(f"### Diagnosis: <span style='color:{color}'>{pred}</span>", unsafe_allow_html=True)
        st.progress(int(conf))
        st.caption(f"Model Confidence: {conf:.1f}%")
        
        # LLM Report Generation
        if st.session_state.get('run_llm', False):
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            
            if api_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    
                    prompt = f"""
                    You are a cardiologist. 
                    Diagnosis: {pred} (Confidence: {conf:.1f}%).
                    Explain what this condition is, its clinical significance, and immediate next steps.
                    Keep it concise (3-4 sentences).
                    """
                    
                    with st.spinner("Generating clinical notes..."):
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.info(f"📋 **Physician's Note:**\n\n{response.choices[0].message.content}")
                except Exception as e:
                    st.warning(f"Could not generate report: {e}")
            else:
                st.info("ℹ️ Add `OPENAI_API_KEY` to Streamlit Secrets for full clinical reports.")
            
            st.session_state.run_llm = False
