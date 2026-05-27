import streamlit as st
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from model import residual_cnn, transforms_val

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