import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import json
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ECG AI Diagnosis", page_icon="🫀", layout="wide")

# --- 2. DEFINE THE EXACT ARCHITECTURE ---
# This class reconstructs the exact FastAI model structure (Body + Head)
class ECGModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1. The Body (EfficientNet-B0)
        # We load it without the top classifier because we use a custom head
        self.body = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        
        # 2. The Custom FastAI Head
        # EfficientNet-B0 outputs 1280 features. FastAI uses ConcatPooling (x2) -> 2560.
        # The structure matches the keys in your error log (BN -> Linear -> BN -> Linear)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2560),
            nn.Dropout(0.25),
            nn.Linear(2560, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Pass through body
        x = self.body.forward_features(x)
        # FastAI Concat Pooling (Max + Avg)
        x = torch.cat([x.mean(dim=(2,3)), x.amax(dim=(2,3))], dim=1)
        # Pass through head
        x = self.head(x)
        return x

# --- 3. SMART LOADER ---
@st.cache_resource
def load_model():
    # ⚠️ REPLACE THIS LIST EXACTLY with the one from Colab
    vocab = ['AFib', 'Anterior MI', 'Inferior MI', 'LBBB', 'Left Vent. Hypertrophy', 'Normal', 'RBBB', 'Sinus Rhythm']
    
    model = ECGModel(num_classes=len(vocab))
    
    if not os.path.exists('ecg_weights.pth'):
        st.error("❌ Error: 'ecg_weights.pth' not found. Please upload it to GitHub.")
        st.stop()
        
    # Load raw weights
    state_dict = torch.load('ecg_weights.pth', map_location=torch.device('cpu'))
    
    # --- KEY MAPPING (The Magic Fix) ---
    # We rename the FastAI keys to match our PyTorch model
    new_state_dict = {}
    for key, value in state_dict.items():
        # Map Body: "0.model.layer" -> "body.layer"
        if key.startswith('0.model.'):
            new_key = key.replace('0.model.', 'body.')
            new_state_dict[new_key] = value
        # Map Head: "1.x.weight" -> "head.x.weight"
        elif key.startswith('1.'):
            # FastAI head layers correspond to our Sequential head indices
            # 1.2 -> 0 (BN), 1.4 -> 2 (Lin), 1.6 -> 4 (BN), 1.8 -> 6 (Lin)
            parts = key.split('.')
            idx = int(parts[1])
            
            # Map the indices explicitly to match our Sequential definition
            if idx == 2: new_idx = '0' # BN
            elif idx == 4: new_idx = '2' # Linear
            elif idx == 6: new_idx = '4' # BN
            elif idx == 8: new_idx = '6' # Linear
            else: continue
            
            new_key = f"head.{new_idx}.{parts[2]}" # e.g., head.0.weight
            new_state_dict[new_key] = value

    # Load the mapped weights
    try:
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        return model, vocab
    except RuntimeError as e:
        st.error(f"❌ Weight Mismatch: {e}")
        st.stop()

try:
    model, vocab = load_model()
except Exception as e:
    st.error(f"❌ Error initializing: {e}")
    st.stop()

# --- 4. IMAGE PREPROCESSING ---
def process_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

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
                    img_tensor = process_image(image)
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                    
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
