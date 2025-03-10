import streamlit as st
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model2 import Generator
from PIL import Image

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained generator model
def load_model(model_path):
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define the image transform (same as during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def convert_to_png(image_path):
    image = cv2.imread(image_path)
    new_path = os.path.splitext(image_path)[0] + '.png'
    cv2.imwrite(new_path, image)
    return new_path

def crop_to_square(image):
    height, width = image.shape[:2]
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    return image[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Function to deblur a single image using the generator model
def deblur_image(image):
    # Convert jpg or jpeg to png if needed
    if isinstance(image, str) and image.lower().endswith(('.jpg', '.jpeg')):
        image = convert_to_png(image)
    
    # Load and preprocess image
    if isinstance(image, str):
        blur_image = cv2.imread(image)
        blur_image_rgb = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
    else:
        blur_image_rgb = np.array(image)
    
    blur_image_rgb = crop_to_square(blur_image_rgb)
    blur_image_tensor = transform(blur_image_rgb).unsqueeze(0).to(device)

    # Predict using the trained generator model
    with torch.no_grad():
        deblurred_image_tensor = generator(blur_image_tensor)

    deblurred_image = deblurred_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    deblurred_image = (deblurred_image - deblurred_image.min()) / (deblurred_image.max() - deblurred_image.min())  # Normalize to [0, 1]
    
    return blur_image_rgb, deblurred_image

# Streamlit app
st.title("Image Deblurring App")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_file = st.sidebar.file_uploader("Choose a model file", type=["pth"])

if model_file is not None:
    model_path = model_file.name
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())
    generator = load_model(model_path)

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        blur_image_rgb, deblurred_image = deblur_image(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(blur_image_rgb, caption='Blurred Image', use_container_width=True)
        with col2:
            st.image(deblurred_image, caption='Deblurred Image', use_container_width=True)
else:
    st.sidebar.warning("Please upload a model file to proceed.")
