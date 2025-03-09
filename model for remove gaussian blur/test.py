import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model (same architecture as the trained model)
class DeblurCNN(nn.Module):
    def __init__(self):
        super(DeblurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load the trained model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = DeblurCNN().to(device)
model.load_state_dict(torch.load('./model.pth', map_location=device))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and preprocess the test image
def deblur_image(image_path, output_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)
    save_image(img, "scale_down_input_image.jpg")

    # Pass through the model
    with torch.no_grad():
        deblurred_img = model(img)

    # Save the output
    save_image(deblurred_img, output_path)
    print(f"Deblurred image saved to {output_path}")

# Test the model on an example blurred image
test_image_path = "./test_data/bird.jpeg"
output_image_path = "deblurred_output.jpg"
deblur_image(test_image_path, output_image_path)
