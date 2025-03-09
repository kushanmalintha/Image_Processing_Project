import torch
import cv2
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import MotionDeblurCNN  # Assuming your model is saved as 'model.py'

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = MotionDeblurCNN().to(device)
model.load_state_dict(torch.load('motion_deblur_model.pth'))
model.eval()

# Define the image transform (same as during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to deblur a single image
def deblur_image(image_path):
    # Load and preprocess image
    blur_image = cv2.imread(image_path)
    blur_image_rgb = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
    blur_image_tensor = transform(blur_image).unsqueeze(0).to(device)

    # Predict using the trained model
    model.eval()
    with torch.no_grad():
        deblurred_image_tensor = model(blur_image_tensor)

    deblurred_image = deblurred_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Display and save results
    plt.subplot(1, 2, 1)
    plt.imshow(blur_image_rgb)
    plt.title('Blurred Image')

    plt.subplot(1, 2, 2)
    plt.imshow(deblurred_image)
    plt.title('Deblurred Image')

    # Save the output image
    output_filename = f"deblurred_{os.path.basename(image_path)}"
    plt.savefig(output_filename)
    plt.show()

# Test with a new image (replace with your test image path)
image_path = ''
deblur_image(image_path)
