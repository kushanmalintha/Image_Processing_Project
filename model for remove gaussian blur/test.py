import cv2
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from deblur import DeblurCNN

# Load the trained model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = DeblurCNN().to(device)
model.load_state_dict(torch.load('./model.pth'))
model.eval()

# Define the image transform (same as during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    return image

# Test on a new image
def test_image(image_path):
    # Load and preprocess image
    blur_image = load_and_preprocess_image(image_path)
    
    # Perform inference
    with torch.no_grad():
        output = model(blur_image)
    
    # Save and display the results
    save_image(output.cpu(), 'deblurred_image.jpg')  # Save the deblurred output
    save_image(blur_image.cpu(), 'input_blurred_image.jpg')  # Save the input blurred image
    
    # Display the results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title('Blurred Image')
    
    plt.subplot(1, 2, 2)
    deblurred_image = cv2.imread('deblurred_image.jpg')
    plt.imshow(cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2RGB))
    plt.title('Deblurred Image')
    
    plt.show()

# Test on a new image (replace with the path to your test image)
test_image('bird.jpeg')
