# Image Deblurring using GAN

This project implements a Generative Adversarial Network (GAN) for image deblurring. The architecture consists of several key components, each playing a specific role in the overall process. Below is a high-level overview of the architecture and the training process.

## Components

### 1. Data Preparation and Augmentation
- **MotionBlurDataset**: A custom dataset class that loads blurred and sharp images from specified directories. It applies transformations to the images if provided.
- **DataAugmentation**: A class that performs random data augmentation techniques such as horizontal flipping, vertical flipping, and rotation to improve the model's generalization.

### 2. Generator
The Generator is responsible for generating deblurred images from blurred images. It uses a series of convolutional layers, residual blocks, and transposed convolutional layers.

- **Initial Convolution**: The input image is processed by a convolutional layer followed by instance normalization and ReLU activation.
  - Input: (3, 256, 256)
  - Output: (64, 256, 256)
- **Downsampling**: Two convolutional layers with stride 2 are used to downsample the image, reducing its spatial dimensions while increasing the number of channels.
  - Down1: (64, 256, 256) -> (128, 128, 128)
  - Down2: (128, 128, 128) -> (256, 64, 64)
- **Residual Blocks**: A series of residual blocks are used to learn complex features. Each block consists of two convolutional layers with batch normalization and ReLU activation, with a skip connection that adds the input to the output.
  - Each Residual Block: (256, 64, 64) -> (256, 64, 64)
- **Upsampling**: Two transposed convolutional layers are used to upsample the image back to its original dimensions.
  - Up1: (256, 64, 64) -> (128, 128, 128)
  - Up2: (128, 128, 128) -> (64, 256, 256)
- **Output Layer**: A final convolutional layer with a Tanh activation function produces the output image.
  - Output: (64, 256, 256) -> (3, 256, 256)

### 3. Discriminator
The Discriminator is responsible for distinguishing between real sharp images and generated deblurred images. It uses a series of convolutional layers to achieve this.

- **Discriminator Blocks**: Each block consists of a convolutional layer followed by instance normalization (except the first block) and LeakyReLU activation. The stride is adjusted in the middle layers to ensure consistent output size.
  - Block1: (6, 256, 256) -> (64, 128, 128)
  - Block2: (64, 128, 128) -> (128, 64, 64)
  - Block3: (128, 64, 64) -> (256, 32, 32)
  - Block4: (256, 32, 32) -> (512, 32, 32)
- **Output Layer**: A final convolutional layer produces a one-channel prediction map indicating whether the input image is real or fake.
  - Output: (512, 32, 32) -> (1, 31, 31)

### 4. VGGLoss
The VGGLoss is a perceptual loss function that uses features extracted from a pre-trained VGG19 network to compute the difference between the generated and real images. This helps improve the quality of the generated images, especially when the dataset is small.

### 5. Training Process
The training process involves alternating between training the Generator and the Discriminator.

- **Discriminator Training**:
  - The Discriminator is trained to distinguish between real and fake images.
  - The loss is computed using Mean Squared Error (MSE) loss for both real and fake images.
  - The total loss is the average of the real and fake losses.

- **Generator Training**:
  - The Generator is trained to produce images that the Discriminator classifies as real.
  - The loss is a combination of GAN loss (MSE loss), pixel-wise loss (L1 loss), and perceptual loss (VGGLoss).
  - The total loss is a weighted sum of these losses.

### 6. Learning Rate Schedulers
Cosine Annealing learning rate schedulers are used for both the Generator and the Discriminator to improve convergence. Additionally, the learning rate is decayed at specific epochs to ensure stability.

### 7. Visualization and Checkpoints
- Sample images are generated periodically during training for visualization.
- Model checkpoints are saved periodically to allow resuming training or for future use.

## Why Tanh Activation for the Output Layer?
The Tanh activation function is used in the output layer of the Generator because it helps in normalizing the output to the range [-1, 1]. This is beneficial for image generation tasks as it helps in stabilizing the training process and produces more realistic images. The Tanh function also helps in mitigating the vanishing gradient problem, which can occur with other activation functions.

## Summary
The architecture and training process are designed to effectively deblur images using a GAN framework. The Generator learns to produce high-quality deblurred images, while the Discriminator ensures that the generated images are indistinguishable from real sharp images. The use of VGGLoss further enhances the perceptual quality of the generated images.

## Usage
To train the GAN, run the following command:
```bash
python train.py --blur_folder path/to/blur_folder --sharp_folder path/to/sharp_folder
```
Make sure to update the `blur_folder` and `sharp_folder` paths to point to your dataset directories.

## Results
The training process will generate sample images and save model checkpoints periodically. The final models will be saved as `final_generator.pth` and `final_discriminator.pth`. The training loss history will be plotted and saved as `gan_training_loss.jpg`.

## Testing
To test the trained model, use the `test_model` function provided in the script. This function will generate deblurred images for a few samples and save the results for visualization.

Make sure to create a test dataloader with a subset of the data before calling this function.

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib

