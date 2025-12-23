import streamlit as st
import torch
from PIL import Image
import timm
import torchvision.transforms as transforms
import numpy as np
import gdown
import os

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    st.info("Loading model...")
    
    # YOUR MODEL (confirmed working ID)
    model_url = "https://drive.google.com/uc?id=1WVvZOoMJemLzebQLC3OxZcuqLTOmqr4I"
    model_path = "/tmp/best_4view_anthrovision_model.pth"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            gdown.download(model_url, model_path, quiet=False, fuzzy=True)
    
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    st.success("✅ Model loaded!")
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def forward_four_views(model, imgs4):
    B, V, C, H, W = imgs4.shape
    imgs_flat = imgs4.view(B * V, C, H, W)
    logits_flat = model(imgs_flat)
    logits = logits_flat.view(B, V, 2)
    return logits.mean(dim=1)

st.set_page_config(page_title="Malnutrition Detector", page_icon="👶", layout="wide")
st.title("👶 Child Malnutrition Detector")

with st.sidebar:
    st.header("📋 Child Info")
    name = st.text_input("Child Name")
    age_months = st.number_input("Age (months)", 1, 120, 24)
    
    st.header("📸 Photo Tips")
    st.markdown("- Good lighting\n- Same distance\n- Face camera directly\n- Plain background")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

front_img = col1.file_uploader("📷 Front", type=['jpg','jpeg','png'])
right_img = col2.file_uploader("➡️ Right", type=['jpg','jpeg','png'])
left_img = col3.file_uploader("⬅️ Left", type=['jpg','jpeg','png']) 
back_img = col4.file_uploader("🔙 Back", type=['jpg','jpeg','png'])

images = [front_img, right_img, left_img, back_img]

if st.button("🚀 ANALYZE", type="primary", use_container_width=True):
    if all(images):
        with st.spinner("Analyzing..."):
            imgs = []
            for img_file in images:
                img = Image.open(img_file).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)
                imgs.append(img_tensor)
                st.image(img, width=150)
            
            imgs_tensor = torch.stack(imgs).to(device)
            
            with torch.no_grad():
                logits = forward_four_views(model, imgs_tensor)
                probs = torch.softmax(logits, dim=1)
                mal_prob = probs[0, 1].item()
            
            col1, col2 = st.columns(2)
            with col1:
                status = "🔴 MALNOURISHED" if mal_prob > 0.5 else "🟢 NORMAL"
                st.metric("Result", status)
                st.metric("Confidence", f"{mal_prob:.1%}")
            
            with col2:
                st.metric("Child", name or "Unknown")
                st.metric("Age", f"{age_months} months")
            
            if mal_prob > 0.5:
                st.error("⚠️ Seek medical help immediately")
            else:
                st.success("✅ Healthy appearance")
    else:
        st.error("Upload ALL 4 images")

st.markdown("*EfficientNet-B0 trained on AnthroVision (F1: 0.49)*")
