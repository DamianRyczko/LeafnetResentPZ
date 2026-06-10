import streamlit as st
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from PIL import Image
from model import residual_cnn, transforms_val
import numpy as np
import torch.nn.functional as F
import matplotlib.cm as cm
import os
import time
import uuid
from streamlit.connections import SQLConnection
import extra_streamlit_components as stx
from sqlalchemy import text

#GRAD-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._h1 = target_layer.register_forward_hook(self._save_act)
        self._h2 = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, out):
        self.activations = out.detach()

    def _save_grad(self, _m, _gi, go):
        self.gradients = go[0].detach()

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)          # forward WITH grad
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1,C,1,1]
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

    def close(self):
        self._h1.remove(); self._h2.remove()


def overlay_cam(pil_img, cam, size=336, alpha=0.5, colormap="jet"):
    img = np.array(pil_img.resize((size, size))) / 255.0
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    heat = cm.get_cmap(colormap)(np.array(cam_img) / 255.0)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)

# PAGE SETUP
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# COOKIE MANAGER AND DATABASE CONNECTION
cookie_manager = stx.CookieManager()
conn = stx.connections.SQLConnection if False else st.connection("history_db", type=SQLConnection)

# Memory containers for current prediciton and user_id
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "gradcam_overlay" not in st.session_state:
    st.session_state.gradcam_overlay = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "chosen_index" not in st.session_state:
    st.session_state.chosen_index = 0
if "show_gradcam_state" not in st.session_state:
    st.session_state.show_gradcam_state = False

# get cookies
cookie_user_id = cookie_manager.get(cookie="plant_detector_user_id")

if cookie_user_id:
    st.session_state.user_id = cookie_user_id
else:
    if st.session_state.user_id is None:
        time.sleep(0.2) 
        cookie_user_id = cookie_manager.get(cookie="plant_detector_user_id")
        
        if cookie_user_id:
            st.session_state.user_id = cookie_user_id
            st.rerun()
        else:
            new_uid = str(uuid.uuid4())
            st.session_state.user_id = new_uid
            cookie_manager.set(cookie="plant_detector_user_id", val=new_uid, max_age=365 * 24 * 3600)
            st.rerun()

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
        padding: 0.2rem 0;
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

    /* Styl domyślny dla przycisków-kafelków (nadpisywany dynamicznie przez st.html) */
    div[data-testid="stHorizontalBlock"] div.stButton > button {
        color: #e8f5ee !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        padding: 0.6rem 1rem !important;
        border-radius: 8px !important;
        text-align: left !important;
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
        font-size: 2.5rem;
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
            
    .history-tile {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.4rem 0.6rem;
        margin-bottom: 0.5rem;
    }
    .history-plant {
        font-size: 0.85rem;
        font-weight: 600;
        color: #AEE8CB;
    }
    .history-date {
        font-size: 0.7rem;
        color: #7a9e8a;
    }
    .history-diagnosis {
        font-size: 0.8rem;
        color: #e8f5ee;
        margin-top: 0.1rem;
    }
    .history-confidence {
        font-size: 0.8rem;
        font-weight: 600;
        color: #52c27a;
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
    return [(class_mapping[str(idx)], p, idx) for idx, p in zip(indices, probs)]

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

def save_to_history(parsed_result, confidence, pil_image):
    """Zapisuje zdjęcie w uploads/<user_id>/ oraz rekord do bazy danych SQLite"""
    current_user = st.session_state.get("user_id")
    if not current_user:
        return

    #Creating user's directory
    user_folder = os.path.join("uploads", current_user)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    #Unique name of the pic
    file_name = f"pred_{int(time.time())}.jpg"
    full_image_path = os.path.join(user_folder, file_name)

    #Save in the directory
    try:
        pil_image.save(full_image_path, "JPEG")
    except Exception as img_err:
        full_image_path = None

    raw_query = """
        INSERT INTO prediction_history (user_id, plant_name, condition_name, confidence, is_healthy, image_path)
        VALUES (:user_id, :plant, :condition, :confidence, :is_healthy, :image_path);
    """
    
    safe_query = text(raw_query)

    with conn.session as session:
        session.execute(
            safe_query,
            {
                "user_id": current_user,
                "plant": parsed_result["plant"],
                "condition": parsed_result["condition"],
                "confidence": float(confidence),
                "is_healthy": bool(parsed_result["is_healthy"]),
                "image_path": full_image_path
            }
        )
        session.commit()

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
    if MODEL_READY:
        st.success("✅ Model successfully loaded")
    else:
        st.warning("⚠️ Demo Mode\n\nModel is not connected - all results are simulated")

    # Sidebar controls
    st.sidebar.title("Settings")
    show_top3 = st.sidebar.toggle("Show top-3 confidence chart", value=True)
    has_results = st.session_state.get("analysis_results") is not None
    
    show_gradcam = st.sidebar.toggle(
        "Show Grad-CAM heatmap", 
        value=st.session_state.show_gradcam_state,
        key="gradcam_toggle_widget",
        disabled=not has_results,
        help="Grad-CAM is only available after the analysis. Upload the photo and click the 'Analyze' button." if not has_results else ""
    )
    st.session_state.show_gradcam_state = show_gradcam

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">📜 Your analysis history </div>', unsafe_allow_html=True)    
    with st.expander("Show last 5 analyses"):
        current_user = st.session_state.get("user_id")
        if current_user:
            try:
                history_df = conn.query(
                    f'SELECT "timestamp", plant_name, condition_name, confidence, image_path FROM prediction_history WHERE user_id = :uid ORDER BY "timestamp" DESC LIMIT 5;',
                    params={"uid": current_user},
                    ttl=0
                )
                
                if history_df is not None and not history_df.empty:
                    for _, row in history_df.iterrows():
                        raw_time = str(row['timestamp'])
                        short_date = raw_time[:10] if len(raw_time) >= 16 else raw_time
                        
                        st.markdown('<div class="history-tile">', unsafe_allow_html=True)
                        
                        col_img, col_txt = st.columns([1, 3], gap="small")
                        
                        with col_img:
                            if row['image_path'] and os.path.exists(row['image_path']):
                                try:
                                    img_thumb = Image.open(row['image_path'])
                                    st.image(img_thumb, use_container_width=True)
                                except:
                                    st.text("🖼️") 
                            else:
                                st.text("🍃") 
                                
                        with col_txt:
                            st.markdown(
                                f'<div class="history-plant">{row["plant_name"] or "Nieznana"} - '
                                f'<span class="history-date">{short_date}</span></div>', 
                                unsafe_allow_html=True
                            )

                            conf_pct = float(row['confidence']) * 100
                            st.markdown(
                                f'<div class="history-diagnosis">{row["condition_name"]} - '
                                f'<span class="history-confidence">{conf_pct:.1f}%</span></div>', 
                                unsafe_allow_html=True
                            )
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.caption("Brak wcześniejszych analiz.")
            except Exception as e:
                st.caption(f"Nie udało się pobrać historii: {e}")
        else:
            st.caption("Inicjalizacja profilu użytkownika...")

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

#Dynamic grad-CAM loading for chosen diagnosis
if uploaded_file := st.session_state.get("last_uploaded_file"):
    if st.session_state.get("analysis_results") is not None and st.session_state.show_gradcam_state:
        results = st.session_state.analysis_results
        current_idx = st.session_state.chosen_index
        _, _, top_idx = results[current_idx]
        
        pil_image_now = Image.open(uploaded_file).convert("RGB")
        input_tensor = transforms_val(pil_image_now).unsqueeze(0)
        cam_extractor = GradCAM(model, model.residual_block14)
        try:
            cam = cam_extractor(input_tensor.clone(), class_idx=top_idx)
            st.session_state.gradcam_overlay = overlay_cam(pil_image_now, cam, alpha=0.5)
        except Exception:
            st.session_state.gradcam_overlay = None
        finally:
            cam_extractor.close()

with col_upload:
    st.markdown("### 📸 Upload leaf photo")
    st.markdown('<div class="color:#a8c9b5; font-size:0.85rem; margin-bottom:1rem;">Supported file formats: JPG, JPEG, PNG · Max. 10 MB</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Drag-and-drop or choose a file",
        type=["jpg", "jpeg", "png"],
        help="To ensure the best results the photo should contain one singular leaf on a uniform background."
    )

    if uploaded_file != st.session_state.last_uploaded_file:
        st.session_state.analysis_results = None
        st.session_state.gradcam_overlay = None
        st.session_state.last_uploaded_file = uploaded_file
        st.session_state.chosen_index = 0
        st.session_state.show_gradcam_state = False

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        if st.session_state.show_gradcam_state and st.session_state.gradcam_overlay is not None:
            st.image(st.session_state.gradcam_overlay, caption="Uploaded photo with Grad-CAM heatmap", use_container_width=True)
        else:
            st.image(pil_image, caption="Uploaded photo", use_container_width=True)
        analyze_btn = st.button("Analyze photo", disabled=not MODEL_READY)
    else:
        st.info(" Upload photo to start the analysis.")
        analyze_btn = False


# ANALYSIS AND RESULTS SECTION
with col_result:
    st.markdown("### 📋 Diagnosis results")

    if analyze_btn and uploaded_file is not None:
        with st.spinner("Analyzing..."):
            progress = st.progress(0)
            for i in range(100):
                progress.progress(i + 1)
            progress.empty()

            results = run_prediction(pil_image, top_k=3)
            st.session_state.analysis_results = results
            st.session_state.chosen_index = 0
            st.session_state.show_gradcam_state = False

            top_class, top_conf, top_idx = results[0]

            #Save as history
            parsed_initial = parse_class_name(top_class)
            try:
                save_to_history(parsed_initial, top_conf, pil_image)
            except Exception as db_err:
                st.sidebar.error(f"Database error: {db_err}")

            st.rerun()

    if uploaded_file is None:
        st.markdown("""
        <div class="info-box" style="color:#8aaa98; font-style:italic;">
            The results will show here ^-.-^ after uploading a photo and clicking the analyze button.
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.get("analysis_results") is not None:
        results = st.session_state.analysis_results
        
        current_idx = st.session_state.chosen_index
        top_class, top_conf, _ = results[current_idx]
        
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
            with st.expander("Check for more information"):
                ref_image_path = os.path.join("references", f"{top_class}/1.png")
                ref_image_path2 = os.path.join("references", f"{top_class}/2.png")

                if os.path.exists(ref_image_path):
                    left_col, right_col = st.columns(2)
                    with left_col:
                        try:
                            ref_img = Image.open(ref_image_path)
                            ref_img_resized = ref_img.resize((300, 300))
                            st.image(ref_img_resized, use_container_width=True)
                        except Exception:
                            st.caption("Something went wrong when loading reference photo.")
                    
                    with right_col:
                        if os.path.exists(ref_image_path2):
                            try:
                                ref_img = Image.open(ref_image_path2)
                                ref_img_resized = ref_img.resize((300, 300))
                                st.image(ref_img_resized, use_container_width=True)
                            except Exception:
                                st.caption("Something went wrong when loading reference photo.")
                else:
                    st.info("Sorry, we don't have any reference photo for that.")   

                if parsed["full_desc"] != parsed["short"]:
                    st.markdown("**Pełny opis laboratoryjny klasy:**")
                    st.caption(parsed["full_desc"])

        # Top-3
        if show_top3 and len(results) > 1:
            st.markdown('<div style="margin-top:1rem;margin-bottom:0.4rem;font-size:0.8rem;color:#a8c9b5;text-transform:uppercase;letter-spacing:0.08em;">Pozostałe możliwe diagnozy (kliknij, aby zamienić)</div>', unsafe_allow_html=True)
            
            with st.container():
                for index, (cls_name, prob, _) in enumerate(results):
                    if index == current_idx:
                        continue
                    
                    display_name = cls_name.replace("_", " ")
                    button_label = f"#{index + 1} | {display_name} — {prob * 100:.1f}%"
                    
                    #Progress bar
                    pct = prob * 100
                    
                    if st.button(button_label, key=f"switch_tile_{index}"):
                        st.session_state.chosen_index = index
                        #Grad-CAM turns off when diagnosis is being changed
                        st.session_state.show_gradcam_state = False
                        st.rerun()

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