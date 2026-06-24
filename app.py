import streamlit as st
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from model import residual_cnn, transforms_val

import numpy as np
import torch.nn.functional as F
import matplotlib.cm as cm

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


model, class_mapping = load_model_and_classes()

# Sidebar controls
st.sidebar.title("Settings")
show_top3 = st.sidebar.toggle("Show top-3 confidence chart", value=True)
show_gradcam = st.sidebar.toggle("Show Grad-CAM heatmap", value=True)
gradcam_alpha = st.sidebar.slider("Heatmap opacity", 0.0, 1.0, 0.5, 0.05)
# Main UI
st.title("Image Classification Model")
st.write("Upload an image to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            input_tensor = transforms_val(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                # Convert raw logits to probabilities
                probabilities = torch.softmax(output, dim=1)
                # Grab top 3
                top3_probs, top3_indices = torch.topk(probabilities, k=3, dim=1)
                top3_probs = top3_probs.squeeze().tolist()
                top3_indices = top3_indices.squeeze().tolist()

            top_class = class_mapping[str(top3_indices[0])]
            top_confidence = top3_probs[0] * 100
            st.success(f"Prediction: **{top_class}**  ({top_confidence:.2f}%)")

            if show_gradcam:
                cam_extractor = GradCAM(model, model.residual_block14)
                try:
                    cam = cam_extractor(input_tensor.clone(), class_idx=top3_indices[0])
                finally:
                    cam_extractor.close()

                overlay = overlay_cam(image, cam, alpha=gradcam_alpha)
                st.subheader("Grad-CAM")
                st.image(overlay, caption=f"What the model looked at for '{top_class}'",
                         use_container_width=True)

            if show_top3:
                st.subheader("Top 3 Predictions")

                top3_classes = [class_mapping[str(idx)] for idx in top3_indices]
                top3_confidence = [p * 100 for p in top3_probs]
                rank_labels = [f"#{i + 1}" for i in range(len(top3_classes))]

                df = pd.DataFrame({
                    'Rank': rank_labels,
                    'Confidence (%)': top3_confidence,
                })

                sns.set_style("whitegrid")
                fig, ax = plt.subplots(figsize=(8, 2.5))
                sns.barplot(
                    data=df,
                    x='Confidence (%)',
                    y='Rank',
                    hue='Rank',
                    palette='viridis',
                    legend=False,
                    ax=ax,
                )
                ax.set_xlim(0, 100)
                ax.set_ylabel("")
                for i, v in enumerate(top3_confidence):
                    ax.text(v + 1, i, f'{v:.2f}%', va='center')
                plt.tight_layout()
                st.pyplot(fig)

                # Full class names underneath
                legend_df = pd.DataFrame({
                    'Rank': rank_labels,
                    'Class': top3_classes,
                    'Confidence': [f"{c:.2f}%" for c in top3_confidence],
                })
                st.table(legend_df.set_index('Rank'))