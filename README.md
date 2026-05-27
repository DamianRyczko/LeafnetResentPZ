# 🌿 LeafNet Image Classifier

A web-based plant disease classification app built with **Streamlit** and a **custom ResNet** model trained from scratch on the **LeafNet** dataset. This project was developed as part of a university assignment.

Upload a photo of a leaf and the model will predict which class it belongs to, showing the top-3 predictions with confidence scores.

---

## ✨ Features

- 🖼️ Upload `.jpg`, `.jpeg`, or `.png` images directly through the browser
- 🧠 Inference powered by a custom-trained residual CNN (ResNet-style architecture)
- 📊 Top-3 predictions visualized as a horizontal bar chart (toggleable from the sidebar)

---

## 🧰 Tech Stack

- **PyTorch** — model definition and inference
- **Streamlit** — interactive web UI
- **Matplotlib + Seaborn** — confidence visualization
- **Pillow** — image handling
- **Docker / Docker Compose** — packaging and deployment

---
## - [See the architecture](https://netron.app/)
-Upload `.onnx` saved model on the site 
---
## 📁 Project Structure

```
.
├── app.py                       # Streamlit application
├── model.py                     # Custom ResNet architecture + image transforms
├── custom_resnet_weights.pth    # Trained model weights
├── leafnet_class_dict.json      # Class name ↔ index mapping
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image definition
└── docker-compose.yml           # Service orchestration
```

---

## 🚀 Setup Guide

### Prerequisites

Make sure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (bundled with Docker Desktop)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Build and run the container

```bash
docker compose up --build
```

This will:
- Build the Docker image from the `Dockerfile`
- Install all Python dependencies from `requirements.txt`
- Start the Streamlit server on port `8501`

To run it in the background (detached mode):

```bash
docker compose up --build -d
```

### 3. Open the app in your browser

Once the container is running, visit:

👉 **http://localhost:8501**

You should see the LeafNet classifier UI. Upload an image and click **Predict**.

---

## 🛑 Stopping the app

To stop the container:

```bash
docker compose down
```

To stop and remove all associated volumes/images:

```bash
docker compose down --rmi all --volumes
```

---

## 🔁 Useful commands

| Command | What it does |
|---|---|
| `docker compose up --build` | Build and start the app |
| `docker compose up -d` | Start in detached mode |
| `docker compose logs -f` | Follow container logs |
| `docker compose restart` | Restart the container |
| `docker compose down` | Stop and remove the container |
| `docker ps` | List running containers |

---

## 📝 Notes

- The container is named `my-pytorch-app` (defined in `docker-compose.yml`).
- `app.py` and `model.py` are mounted as volumes, so edits to those files are reflected without rebuilding the image — just restart the container with `docker compose restart`.
- Model weights and the class dictionary are baked into the image during build, so changes to them **do** require a rebuild (`docker compose up --build`).

---

## 🎓 About

This project was created as a university assignment. The classifier uses a custom residual CNN architecture trained on the LeafNet plant disease dataset.
