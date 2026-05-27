import streamlit as st
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from PIL import Image
from model import residual_cnn, transforms_val

# PAGE SETUP
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

#CS SYLES
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
        }
        
    .mian-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        color: #7dcea0;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #a8c9b5;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #a8c9b5;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .result-card {
        background: rgba(45,122, 79, 0.15);
        border-left: 5px solid #52c27a;
        border-radius: 12px;
        padding: 1.5re, 2rem;
        margin: 1rem 0;
    }    
    
    .result-card.warning {
        background: rgba(224, 123, 57, 0.15);
        border-left: 5px solid #e07b39;
    }
    
    .result-card.danger {
        background: rgba(192, 57, 43, 0.15);
        border-left: 5px solid #e74c3c;
    }
    
    .disease-name {
        font-family: 'Source Sans 3', serif;
        font-size: 1.5rem;
        color: #e8f5ee;
        margin-bottom: 0.3rem;
    }
    
    .confidence-bar-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        height: 10px;
        margin: 05rem 0 0.8rem 0;
        overflow: hidden;
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #2d7a4f, #52c27a);
    }
    
    .conf-label{
        color: #a8c9b5;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .conf-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #52c27a;
        margin-bottom: 1rem;
    }
    
    .top3-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.9);
    }
    
    .top3-row:last-child {
        border-bottom: none
    }
    
    .top3-rank {
        font-size: 0.8rem;
        font-weight: 600;
        color: #7dcea0;
        min-width: 24px;
        text-align: center;
    }
    
    .top3-name {
        felx: 1;
        font-size: 0.95rem;
        color: #d0e8d8;
    }
    
    .top3-pct {
        font-size: 0.9rem;
        font-weight: 600;
        color: #52c27a;
        min-width: 52px;
        text-align: right;
    }
    
    .top3-bar-bg {
        flex: 1;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 50px;
        height: 6px;
        overflow: hidden;
    }
    
    .top3-bar-fill {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #2d7a4f, #52c27a);
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(125, 206, 160, 0.25);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: #114623;
        line-height: 1.6;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2d7a4f, #1a5c39);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-family: 'Source Sans 3', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3a9e65, #2d7a4f);
        box-shadow: 0 4px 20px rgba(82, 194, 122, 0.35);
        transform: translateY(-1px);
    }
    
    .sidebar-info {
        font-size: 0.85rem;
        color: #8ACBAA;
        line-height: 1.7rem;
    }
    
    .sidebar-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #AEE8CB;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem; margin-top: 0.8rem;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3d2b 0%, #1e1e1e 100%);
    }
    
    [data-testid="stSidebarHeader"] {
        display: none;
    }
    
    .plant-emoji {
        font-size: 2,5rem;
        display: block;
        margin-bottom: 0.3rem;
    }
    
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(44, 84, 64, 0.2);
        margin: 1rem 0;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #2d7a4f, #52c27a);
    }
    
    .main-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);    
        margin: 1.5rem 0;
    }
    
    .footer-text {
        text-align: center;
        color: #7a9e8a;
        font-size: 0.82rem;
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# MODEL LOADING
@st.cache_resource
def load_model_and_classes():
    with open('leafnet_class_dict.json', 'r') as f:
        raw_mapping = json.load(f)

    # Reverse: {"Apple_scab": 0} -> {"0": "Apple_scab"}
    class_mapping = {str(value): key for key, value in raw_mapping.items()}
    num_classes = len(class_mapping)

    model = residual_cnn(num_classes=num_classes)
    state_dict = torch.load('custom_resnet_weights.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_mapping

try:
    model, class_mapping = load_model_and_classes()
    MODEL_READY = True
except Exception as e:
    model, class_mapping = None, {}
    MODEL_READY = False
    MODEL_ERROR = str(e)

# PREDICTION
def run_prediction(pil_image, top_k=3):
    input_tensor = transforms_val(pil_image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        # Convert raw logits to probabilities
        probabilities = torch.softmax(output, dim=1)
        topk_probs, topk_indices = torch.topk(probabilities, k=top_k, dim=1)

    probs = topk_probs.squeeze().tolist()
    indices = topk_indices.squeeze().tolist()
    if not isinstance(probs, list):
        probs, indices = [probs], [indices]
    return [(class_mapping[str(idx)], p) for idx, p in zip(indices, probs)]

def parse_class_name(raw_name):
    """
    Wyciąga roślinę i stan z opisu tekstowego klasy.
    Obsługuje formaty PlantVillage:
      "a image of Peach leaves diseased by Bacterial spot with symptoms..."
      "a image of Cucumber healthy leaves with..."
      "Tomato_Early_blight"
    Zwraca dict: plant, condition, is_healthy, short, full_desc
    """

    raw = raw_name.strip()
    result = {"full_desc": raw, "is_healthy": False, "plant": "", "condition": "", "short": raw}
    lower = raw.lower()

    # Format: "a image/photo of <Plant> ... (healthy | diseased by <Disease>) ..."
    m = re.search(
        r"a (?:image|photo) of ([A-Za-z ]+?) (?:leaves? )?"
        r"(healthy|diseased by ([A-Za-z ,\-]+?))"
        r"(?:\s+with|\s+leaves|\s*$)",
        raw, re.IGNORECASE
    )
    if m:
        plant = m.group(1).strip().title()
        if "healthy" in m.group(2).lower():
            condition, is_healthy = "Healthy", True
        else:
            condition = (m.group(3) or m.group(2)).strip().title()
            is_healthy = False
        result.update({"plant": plant, "condition": condition,
                       "is_healthy": is_healthy, "short": f"{plant} — {condition}"})
        return result

    # Format: "Tomato_Early_blight" / "Apple___Apple_scab"
    clean = re.sub(r"_+", " ", raw).strip()
    parts = clean.split()
    if len(parts) >= 2:
        plant = parts[0].title()
        condition = " ".join(parts[1:]).title()
        is_healthy = "healthy" in condition.lower()
        result.update({"plant": plant, "condition": condition,
                       "is_healthy": is_healthy, "short": f"{plant} — {condition}"})
        return result

    # Fallback
    is_healthy = "healthy" in lower
    short = raw if len(raw) <= 55 else raw[:52] + "..."
    result.update({"is_healthy": is_healthy, "short": short})
    return result

#SIDEBAR
with st.sidebar:
    st.markdown('<span class="plant-emoji">🌿</span>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info">Detects and distinguishes plant diseases based on a leaf photo. Model <b>Resnet</b> wytrenowany na zbiorze []. </div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label"> How to use it</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sidebar-info">
        1. Upload a photo of a plant leaf<br>
        2. Make sure that the leaf is clearly visible<br>
        3. Click the 'Analyze' button<br>
        4. Read the diagnosis and recommended treatment options
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label"> 🌱 Plants in database:</div>', unsafe_allow_html=True)
    plants = ["🍎 Apple", "🍇 Grape", "🍅 Tomato", "🥔 Potato",
              "🍑 Peach", "🌽 Corn", "🫑 Paprika", "🍓 Strawberry"]
    plants_html = "".join(f'<div class="sidebar-info">{p}</div>' for p in plants)
    st.markdown(plants_html, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    # Status modelu
    if MODEL_READY:
        st.success("✅ Model successfully loaded")
    else:
        st.warning("⚠️ Demo Mode\n\nModel is not connected - all results are simulated")

# Sidebar controls
st.sidebar.title("Settings")
show_top3 = st.sidebar.toggle("Show top-3 confidence chart", value=True)

# HEADER
st.markdown('<h1 class="main-title">🌿 Plant Disease<br>Detector</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detects and distinguishes plant diseases based on a leaf photo.</div>', unsafe_allow_html=True)

# Include demo-mode banner
if not MODEL_READY:
    st.error(f"Could not load model: '{MODEL_ERROR}' \n"
             f"Make sure that all files are in the same directory as this application.")
st.markdown('<div class="main-divider"></div>', unsafe_allow_html=True)

# MAIN SECTION
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📸 Upload leaf photo")
    st.markdown('<div class="color:#a8c9b5; font-size:0.85rem; margin-bottom:1rem;">Supported file formats: JPG, JPEG, PNG · Max. 10 MB</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Drag-and-drop or choose a file",
        type=["jpg", "jpeg", "png"],
        help="To ensure the best results the photo should contain one singular leaf on a uniform background."
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(pil_image, caption="Uploaded photo", use_container_width=True)

        analyze_btn = st.button("Analyze photo", disabled=not MODEL_READY)
    else:
        st.info(" Upload photo to start the analysis.")
        analyze_btn = False


# ANALYSIS AND RESULTS SECTION

with col_result:
    st.markdown("### 📋 Diagnosis results")

    if uploaded_file is None:
        st.markdown("""
        <div class="info-box" style="color:#8aaa98; font-style:italic;">
            The results will show here ^-.-^ after uploading a photo and clicking the analyze button.
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn:
        with st.spinner("Analyzing..."):
            progress = st.progress(0)
            for i in range(100):
                progress.progress(i + 1)
            progress.empty()
            top_k = 3 if show_top3 else 1
            results = run_prediction(pil_image, top_k=top_k)

        top_class, top_conf  = results[0]
        parsed = parse_class_name(top_class)

        is_healthy = parsed["is_healthy"]
        card_style = "" if is_healthy else ("danger" if top_conf > 0.7 else "warning")
        icon = "✅" if is_healthy else "🔬"

        plant_html = f'<span style="opacity:0.7;font-size:1rem;font-weight:400;">{parsed["plant"]}</span><br>' if parsed["plant"] else ""
        cond_html = parsed["condition"] if parsed["condition"] else parsed["short"]

        st.markdown(f"""
        <div class="result-card {card_style}">
            <div class="disease-name">Diagnosis</div>
            <div class="disease-name">{icon} {plant_html}{cond_html}</div>
            <div class="confidence-bar-container">
                <div class="confidence-bar" style="width:{top_conf * 100:.1f}%"></div>
            </div>
            <div class="conf-value">{top_conf * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        if parsed["full_desc"] != parsed["short"] and len(parsed["full_desc"]) > 60:
            with st.expander("Full class description"):
                st.caption(parsed["full_desc"])

        # Top-3
        if show_top3 and len(results) > 1:
            st.markdown('<div style="margin-top:1rem;margin-bottom:0.4rem;font-size:0.8rem;color:#a8c9b5;text-transform:uppercase;letter-spacing:0.08em;">Inne możliwe diagnozy</div>', unsafe_allow_html=True)
            rows_html = ""
            for rank, (cls_name, prob) in enumerate(results[1:], start=2):
                rows_html += f"""
                <div class="top3-row">
                    <div class="top3-rank">#{rank}</div>
                    <div class="top3-name">{cls_name.replace("_", " ")}</div>
                    <div class="top3-bar-bg">
                        <div class="top3-bar-fill" style="width:{prob * 100:.1f}%"></div>
                    </div>
                    <div class="top3-pct">{prob * 100:.1f}%</div>
                </div>"""
            st.markdown(f'<div class="info-box" style="padding:0.8rem 1.2rem;">{rows_html}</div>',
                        unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-box">
            ✅ Photo uploaded. Press <b>„Analyze photo"</b> to ask the AI model.
        </div>""", unsafe_allow_html=True)


#FOOTER
st.markdown('<div class = "main-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-text">
    Plant Disease Detector · Trained on [] ·
    Model distinguishes [] diseases and healthy plants
</div>
""", unsafe_allow_html=True)

