import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import random

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Custom Dataset for Motion Blur images
class MotionBlurDataset(Dataset):
    def __init__(self, blur_folder, sharp_folder, transform=None):
        self.blur_paths = [os.path.join(blur_folder, f) for f in os.listdir(blur_folder) if f.endswith('.png')]
        self.sharp_paths = [os.path.join(sharp_folder, f) for f in os.listdir(sharp_folder) if f.endswith('.png')]
        self.transform = transform
        
    def __len__(self):
        return len(self.blur_paths)
    
    def __getitem__(self, idx):
        blur_image = cv2.imread(self.blur_paths[idx])
        blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
        
        sharp_image = cv2.imread(self.sharp_paths[idx])
        sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)
        
        return blur_image, sharp_image

# Data Augmentation - especially important for small datasets
class DataAugmentation:
    def __call__(self, image):
        # Convert to PIL for transforms
        image = transforms.ToPILImage()(image)
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
        
        # Random vertical flipping    
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([-90, 90, 180])
            image = transforms.functional.rotate(image, angle)
            
        # Convert back to tensor
        image = transforms.ToTensor()(image)
        
        return image

# Generator using ResNet blocks (better than standard UNet for deblurring)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self, num_res_blocks=9):
        super(Generator, self).__init__()
        
        # Initial Convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_res_blocks)]
        )
        
        # Upsampling with attention mechanism
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return x

# PatchGAN Discriminator (from pix2pix) - FIXED for size issues
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalization=True, stride=2):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Adjusted stride in middle layers for consistent output size
        self.model = nn.Sequential(
            # Input: concatenated blur and sharp images
            *discriminator_block(6, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),  # Changed stride to 1
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # Output one channel prediction map
        )
    
    def forward(self, blur_img, sharp_img):
        # Concatenate images along channel dimension
        img_input = torch.cat((blur_img, sharp_img), 1)
        return self.model(img_input)

# Perceptual Loss using VGG features (helps with small datasets)
class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)
        except:
            # Fallback for older PyTorch versions
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features.to(device)
            
        self.vgg = nn.Sequential(*list(vgg.children())[:36]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()
        
    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = self.criterion(x_vgg, y_vgg)
        return loss

# Custom weights initialization (helps with convergence)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Training function
def train_gan(generator, discriminator, dataloader, num_epochs=100, vgg_loss_weight=10, device='cuda'):
    generator.to(device)
    discriminator.to(device)
    
    # Apply custom weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Optimizers
    optimizer_G = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Learning rate schedulers for better convergence
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    vgg_loss = VGGLoss(device)
    
    # Decay factor for learning rate
    # Implement learning rate decay for stability
    decay_factor = 0.1
    decay_epochs = [num_epochs // 2, num_epochs * 3 // 4]
    
    # Training metrics
    loss_history = {'g_loss': [], 'd_loss': []}
    
    # Sample images for visualization
    sample_images = next(iter(dataloader))
    fixed_blur = sample_images[0][:4].to(device)
    fixed_sharp = sample_images[1][:4].to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        
        for i, (blur_images, sharp_images) in enumerate(dataloader):
            blur_images = blur_images.to(device)
            sharp_images = sharp_images.to(device)
            batch_size = blur_images.size(0)
            
            # Generate deblurred images first to determine output size
            fake_images = generator(blur_images)
            
            # Get the actual output size from discriminator
            with torch.no_grad():
                test_output = discriminator(blur_images, sharp_images)
                output_size = test_output.size()
            
            # Create properly sized labels
            real_label = torch.ones((batch_size, *output_size[1:]), device=device) * 0.9  # Label smoothing
            fake_label = torch.zeros((batch_size, *output_size[1:]), device=device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss for real images
            real_pred = discriminator(blur_images, sharp_images)
            loss_D_real = criterion_GAN(real_pred, real_label)
            
            # Loss for fake images
            fake_pred = discriminator(blur_images, fake_images.detach())
            loss_D_fake = criterion_GAN(fake_pred, fake_label)
            
            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            # GAN loss
            fake_pred = discriminator(blur_images, fake_images)
            loss_G_GAN = criterion_GAN(fake_pred, real_label)
            
            # Pixel-wise loss
            loss_G_pixel = criterion_pixel(fake_images, sharp_images)
            
            # VGG perceptual loss
            loss_G_vgg = vgg_loss(fake_images, sharp_images)
            
            # Total generator loss
            loss_G = loss_G_GAN + 10 * loss_G_pixel + vgg_loss_weight * loss_G_vgg
            loss_G.backward()
            optimizer_G.step()
            
            # Record losses
            g_epoch_loss += loss_G.item()
            d_epoch_loss += loss_D.item()
            
            # Print training progress
            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Adjust learning rate at specific epochs
        if epoch + 1 in decay_epochs:
            for param_group in optimizer_G.param_groups:
                param_group['lr'] *= decay_factor
            for param_group in optimizer_D.param_groups:
                param_group['lr'] *= decay_factor
        
        # Record average epoch losses
        avg_g_loss = g_epoch_loss / len(dataloader)
        avg_d_loss = d_epoch_loss / len(dataloader)
        loss_history['g_loss'].append(avg_g_loss)
        loss_history['d_loss'].append(avg_d_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {avg_g_loss:.4f}, Discriminator Loss: {avg_d_loss:.4f}")
        
        # Save generated sample images
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                generator.eval()
                fake_samples = generator(fixed_blur)
                generator.train()
                
                # Convert to numpy for visualization
                fake_samples = fake_samples.cpu().detach()
                # Save samples if needed (not implemented here)
        
        # Save model checkpoints periodically
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch+1}.pth')
    
    # Save final models
    torch.save(generator.state_dict(), 'final_generator.pth')
    torch.save(discriminator.state_dict(), 'final_discriminator.pth')
    
    return generator, discriminator, loss_history

# Main execution
if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define the image transform with augmentation for training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Folder paths - update these to your dataset paths
    blur_folder = '/media/ravindu/38D6E9B4D6E97314/My Projects/python/Kushan/Image_Processing_Project/gdrive/blurred'
    sharp_folder = '/media/ravindu/38D6E9B4D6E97314/My Projects/python/Kushan/Image_Processing_Project/gdrive/sharp'

    # Create dataset and dataloader with augmentation
    dataset = MotionBlurDataset(blur_folder, sharp_folder, transform=transform)
    
    # Small batch size for better stability with limited data
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    generator = Generator(num_res_blocks=9)
    discriminator = Discriminator()
    
    # Train the GAN
    generator, discriminator, loss_history = train_gan(
        generator, 
        discriminator, 
        dataloader, 
        num_epochs=100,
        vgg_loss_weight=10,
        device=device
    )
    
    # Plot the loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history['g_loss'], label='Generator Loss')
    plt.plot(loss_history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('gan_training_loss.jpg')
    plt.show()
    
    # Test the model with a few examples
    def test_model(generator, test_loader, device, num_samples=5):
        generator.eval()
        with torch.no_grad():
            for i, (blur_images, sharp_images) in enumerate(test_loader):
                if i >= num_samples:
                    break
                    
                blur_images = blur_images.to(device)
                sharp_images = sharp_images.to(device)
                
                # Generate deblurred images
                deblurred_images = generator(blur_images)
                
                # Convert to CPU and denormalize
                blur_images = blur_images.cpu() * 0.5 + 0.5
                sharp_images = sharp_images.cpu() * 0.5 + 0.5
                deblurred_images = deblurred_images.cpu() * 0.5 + 0.5
                
                # Plot and save results
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(blur_images[0].permute(1, 2, 0))
                axes[0].set_title('Blurred Image')
                axes[0].axis('off')
                
                axes[1].imshow(deblurred_images[0].permute(1, 2, 0))
                axes[1].set_title('Deblurred Image (Generated)')
                axes[1].axis('off')
                
                axes[2].imshow(sharp_images[0].permute(1, 2, 0))
                axes[2].set_title('Sharp Image (Ground Truth)')
                axes[2].axis('off')
                
                plt.savefig(f'compare_results/deblur_result_{i}.jpg')
                plt.close()
    
    # Create a test dataloader with a subset of the data
    test_dataset = MotionBlurDataset(blur_folder, sharp_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Test the model
    test_model(generator, test_loader, device)


    # # visualise architecture
    # from torchsummary import summary
    # summary(generator, (3, 256, 256))
    # summary(discriminator, [(3, 256, 256), (3, 256, 256)])
    