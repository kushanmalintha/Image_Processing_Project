import torch
import cv2
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model2 import Generator

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained generator model
generator = Generator().to(device)
generator.load_state_dict(torch.load('final_generator.pth'))
generator.eval()

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
    height, width, _ = image.shape
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    return image[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Function to deblur a single image using the generator model
def deblur_image(image_path):
    # Convert jpg or jpeg to png
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        image_path = convert_to_png(image_path)

    # Load and preprocess image
    blur_image = cv2.imread(image_path)
    blur_image = crop_to_square(blur_image)
    blur_image_rgb = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
    blur_image_tensor = transform(blur_image_rgb).unsqueeze(0).to(device)

    # Predict using the trained generator model
    with torch.no_grad():
        deblurred_image_tensor = generator(blur_image_tensor)

    deblurred_image = deblurred_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    deblurred_image = (deblurred_image - deblurred_image.min()) / (deblurred_image.max() - deblurred_image.min())  # Normalize to [0, 1]
    
    # Display and save results
    plt.subplot(1, 2, 1)
    plt.imshow(blur_image_rgb)
    plt.title('Blurred Image')

    plt.subplot(1, 2, 2)
    plt.imshow(deblurred_image)
    plt.title('Deblurred Image')

    # Save the output image
    output_filename = f"compare_results/deblurred_{os.path.basename(image_path)}"
    plt.savefig(output_filename)
    plt.show()

# Test with a new image (replace with your test image path)
image_path = '/media/ravindu/38D6E9B4D6E97314/My Projects/python/Kushan/Image_Processing_Project/motion_blurred/7_NIKON-D3400-35MM_M.JPG'
deblur_image(image_path)
