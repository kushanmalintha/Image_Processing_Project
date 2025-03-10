import torch
from torchsummary import summary
from model2 import Generator, Discriminator

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Print summaries
print("Generator Architecture:")
summary(generator, (3, 256, 256))

print("\nDiscriminator Architecture:")
summary(discriminator, [(3, 256, 256), (3, 256, 256)])
