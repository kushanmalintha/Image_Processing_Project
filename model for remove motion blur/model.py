import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import os
import matplotlib.pyplot as plt

# Define the Dataset and DataLoader
class MotionBlurDataset(Dataset):
    def __init__(self, blur_folder, sharp_folder, transform=None):
        self.blur_paths = [os.path.join(blur_folder, f) for f in os.listdir(blur_folder) if f.endswith('.jpg')]
        self.sharp_paths = [os.path.join(sharp_folder, f) for f in os.listdir(sharp_folder) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur_image = cv2.imread(self.blur_paths[idx])
        sharp_image = cv2.imread(self.sharp_paths[idx])

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, sharp_image

# Define the Model
class MotionDeblurCNN(nn.Module):
    def __init__(self):
        super(MotionDeblurCNN, self).__init__()
        # Encoder
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Decoder
        self.decoder1 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder3 = self.upconv_block(128, 64)
        self.decoder4 = self.upconv_block(64, 3, final_layer=True)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Tanh()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Decoder
        dec1 = self.decoder1(enc4)
        dec2 = self.decoder2(dec1 + enc3)  # Skip connection
        dec3 = self.decoder3(dec2 + enc2)  # Skip connection
        dec4 = self.decoder4(dec3 + enc1)  # Skip connection

        return dec4

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MotionDeblurCNN().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Folder paths to datasets (change to your dataset paths)
blur_folder = './motion_blurred'
sharp_folder = './sharp'

# Create dataset and dataloader
dataset = MotionBlurDataset(blur_folder, sharp_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training function
def train(model, dataloader, num_epochs=10):
    model.train()
    loss_history = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (blur_images, sharp_images) in enumerate(dataloader):
            blur_images = blur_images.to(device)
            sharp_images = sharp_images.to(device)
            
            optimizer.zero_grad()
            outputs = model(blur_images)
            loss = criterion(outputs, sharp_images)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return loss_history

# Train the model
loss_history = train(model, dataloader, num_epochs=10)

# Save the trained model in .pth format (PyTorch format)
torch.save(model.state_dict(), 'motion_deblur_model.pth')

# Plotting the loss history
plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.savefig('training_loss.jpg')

# Optionally, display the plot
plt.show()
