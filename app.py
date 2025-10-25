# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import timm
import torch.nn as nn
import random
import torch.nn.functional as F

# ----------------------------------
# MODEL DEFINITION
# ----------------------------------
class ResNet50Transformer(nn.Module):
    def _init_(self, num_classes=10, use_transformer=True, transformer_dim=256, num_heads=4, num_layers=2):
        super()._init_()
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.use_transformer = use_transformer
        if use_transformer:
            self.proj = nn.Linear(feat_dim, transformer_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Linear(transformer_dim, num_classes)
        else:
            self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if self.use_transformer:
            x_t = self.proj(feats).unsqueeze(0)
            x_t = self.transformer(x_t)
            x_t = x_t.squeeze(0)
            logits = self.classifier(x_t)
        else:
            logits = self.classifier(feats)
        return logits


# ----------------------------------
# TRANSFORMS
# ----------------------------------
def random_occlusion(image, max_h=16, max_w=16):
    """Randomly black out a region of the image for robustness testing."""
    h, w = image.shape[:2]
    occ_h = random.randint(6, max(8, max_h))
    occ_w = random.randint(6, max(8, max_w))
    x1 = random.randint(0, w - occ_w)
    y1 = random.randint(0, h - occ_h)
    image[y1:y1+occ_h, x1:x1+occ_w, :] = 0
    return image

val_transforms = A.Compose([
    A.Resize(32, 32),
    A.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.247, 0.243, 0.261)),
    ToTensorV2()
])

inv_normalize = torchvision.transforms.Normalize(
    mean=[-0.4914/0.247, -0.4822/0.243, -0.4465/0.261],
    std=[1/0.247, 1/0.243, 1/0.261]
)


# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50Transformer(use_transformer=True).to(device)
    model.load_state_dict(torch.load('best_model_cls.pth', map_location=device))
    model.eval()
    return model, device

model, device = load_model()
classes = torchvision.datasets.CIFAR10(root='./data', train=False).classes


# ----------------------------------
# STREAMLIT APP
# ----------------------------------
st.set_page_config(page_title="CIFAR-10 Robustness Classifier", page_icon="🧠", layout="centered")

st.title("🧠 CIFAR-10 Robust Classification App")
st.markdown("""
Upload an image to see how the trained ResNet50+Transformer model classifies it.  
You can also apply *random occlusion* to test model robustness.
""")

uploaded_file = st.file_uploader("📁 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)

    st.image(image, caption="Original Image", width=250)

    # --- Optional Occlusion ---
    apply_occ = st.checkbox("Apply Random Occlusion (simulate damage/noise)")
    if apply_occ:
        img_np = random_occlusion(img_np)
        st.image(img_np, caption="Occluded Image", width=250)

    # --- Preprocess for Model ---
    augmented = val_transforms(image=img_np)
    inp = augmented['image'].unsqueeze(0).to(device)

    # --- Prediction ---
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(1).item()
        confidence = probs[0][pred].item() * 100

    pred_class = classes[pred]

    st.success(f"🎯 *Predicted Label:* {pred_class}")
    st.metric(label="Confidence", value=f"{confidence:.2f}%")

    # --- Optional: Display Top-3 Predictions ---
    topk = torch.topk(probs, 3)
    top_probs = topk.values[0].cpu().numpy()
    top_labels = [classes[i] for i in topk.indices[0].cpu().numpy()]

    st.write("*Top 3 Predictions:*")
    for lbl, pr in zip(top_labels, top_probs):
        st.write(f"• {lbl} — {pr*100:.2f}%")

else:
    st.info("👆 Upload an image to get started!")


st.markdown("---")
st.caption("Model: ResNet50 + Transformer | Dataset: CIFAR-10 | Augmentation: Albumentations (Random Occlusion)")