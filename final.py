import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from model import SegNet
from bts.model import DynamicUNet
from bts.classifier import BrainTumorClassifier

# -------------------------------
# 🎯 Set Device (CPU/GPU)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🚀 Load U-Net Model
# -------------------------------
FILTER_LIST = [16, 32, 64, 128, 256]
MODEL_PATH = "UNet-[16, 32, 64, 128, 256].pt"

# Load U-Net Architecture
unet_model = DynamicUNet(FILTER_LIST).to(device)

# Load Model Weights
state_dict = torch.load(MODEL_PATH, map_location=device)
unet_model.load_state_dict(state_dict)
unet_model.to(device)

# Create Classifier
unet_classifier = BrainTumorClassifier(unet_model, device)

# -------------------------------
# 🔥 Load SegNet Model
# -------------------------------
segnet = SegNet().to(device)  
segnet.load_state_dict(torch.load("segnet_model.pth", map_location=device))
segnet.eval()  # Put in evaluation mode

# -------------------------------
# 🎯 Grad-CAM Implementation
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, image_tensor):
        image_tensor = image_tensor.unsqueeze(0).to(device)
        self.model.zero_grad()
        
        output = self.model(image_tensor)  # Get segmentation output
        score = output.sum()  # Summing across channels for segmentation
        
        score.backward()
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        # Compute Grad-CAM heatmap
        weights = np.mean(gradients, axis=(2, 3))
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (512, 512))  # Resize to match input size
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize
        return cam

grad_cam_layer = unet_model.conv10
grad_cam = GradCAM(unet_model, grad_cam_layer)

# -------------------------------
# 🖼️ Image Preprocessing Function
# -------------------------------
def preprocess_image(img_pil):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return transform(img_pil)

# -------------------------------
# Segmentation with Red Highlight
# -------------------------------
def segmentation(image_tensor, original_image):
    """
    Perform U-Net segmentation and highlight the segmented area in red on the original image.
    """
    image_tensor = image_tensor.view((-1, 1, 512, 512)).to(device)
    output = unet_model(image_tensor).detach().cpu()
    output = (output > 0.5).numpy().astype(np.uint8) * 255

    # Resize original image to match mask size
    original_image = cv2.resize(original_image, (512, 512))

    # Create a red overlay where mask is present
    mask_colored = np.zeros_like(original_image)
    mask_colored[:, :, 2] = output.squeeze()  # Red channel (BGR format)

    # Blend original image with red mask
    overlay = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)

    return overlay

# -------------------------------
# 🏠 Streamlit UI
# -------------------------------
st.markdown(
    "<h1 style='text-align: center; color: white;'>🧠 Brain Tumor Detection Using Deep Learning</h1>",
    unsafe_allow_html=True
)

st.write(
    "🚀 This tool uses **U-Net & SegNet** for brain abnormality segmentation and **Grad-CAM** for visualization."
)

# 📂 Upload Image
uploaded_file = st.file_uploader("📤 Upload a Brain MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Convert PIL Image to OpenCV Format
    img_pil = Image.open(uploaded_file).convert("RGB")
    image = np.array(img_pil)
    
    # Display Original Image
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # ⏳ Loading Bar for Processing
    with st.spinner("Processing Image..."):
        img_tensor = preprocess_image(img_pil)
        original_image_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 🔘 Tabs for U-Net & Grad-CAM Results
    tab1, tab2 = st.tabs(["🧬 U-Net Segmentation", "🔥 Grad-CAM Heatmap"])

    with tab1:
        if st.button("🧬 Run Segmentation", use_container_width=True):
            with st.spinner("Running... ⏳"):
                overlay_image = segmentation(img_tensor, original_image_cv)
                st.image(overlay_image, caption="🩺 U-Net Segmentation", use_column_width=True)

    with tab2:
        if st.button("🔥 Run Grad-CAM", use_container_width=True):
            with st.spinner("Generating Grad-CAM... 🔥"):
                heatmap = grad_cam.generate_heatmap(img_tensor)

                # Overlay Grad-CAM on Original Image
                heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(cv2.resize(image, (512, 512)), 0.5, heatmap, 0.5, 0)

                st.image(overlay, caption="🔥 Grad-CAM Heatmap", use_column_width=True)

# -------------------------------

