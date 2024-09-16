import torch
import torch.nn as nn
from torchvision.models import vgg19
import config
from model import Discriminator

# Define PerceptualLoss class using a pre-trained VGG19 model to extract feature representations
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # print(vgg19(pretrained=True).features)
        # Load the pre-trained VGG19 model and use the first 35 layers (conv + pooling)
        # This extracts high-level features from the input and target images
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(config.DEVICE)

        # Freeze the VGG parameters so they are not updated during training
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Define MSE (Mean Squared Error) loss to compare feature maps of input and target images
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        # Extract VGG feature maps for the input image
        vgg_input_features = self.vgg(input)
        # Extract VGG feature maps for the target image
        vgg_target_features = self.vgg(target)
        # Compute MSE loss between the extracted feature maps of input and target
        return self.loss(vgg_input_features, vgg_target_features)

# Define ContentLoss class using L1 Loss (Mean Absolute Error) for content comparison
class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use L1 loss to calculate the content difference between input and target
        self.loss = nn.L1Loss()

    def forward(self, xf, xr):
        # Compute the L1 loss between xf (generated image) and xr (real image)
        return self.loss(xf, xr)

# Define Ra class for the Relativistic average discriminator
class Ra(nn.Module):
    def __init__(self, trained_discriminator) -> None:
        super().__init__()
        # Use the passed pre-trained discriminator
        self.discriminator = trained_discriminator
        
    def forward(self, x_1_image, x_2_image):
        # Compute discriminator score for the first image (real or fake)
        c1 = self.discriminator(x_1_image)
        # Compute the mean discriminator score for the second image (real or fake)
        # This acts as an approximation of the expectation
        c2 = torch.mean(self.discriminator(x_2_image))
        # Return the sigmoid of the difference between c1 and c2 (Relativistic average)
        return torch.sigmoid(c1 - c2)

# Define the loss function for the discriminator using the Ra mechanism
class Discriminator_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Initialize the Ra object with the pre-trained discriminator
        
    
    def forward(self, xf, xr,trained_discriminator):
        ra = Ra(trained_discriminator)
        # Compute the loss for real images (log of the discriminator's Ra output)
        real_loss = torch.mean(torch.log(ra(xr, xf)))
        # Compute the loss for fake images (log of 1 - discriminator's Ra output)
        fake_loss = torch.mean(torch.log(1 - ra(xf, xr)))
        # Return the average of real and fake loss (negated)
        return -(real_loss + fake_loss) / 2

# Define the loss function for the generator using the Ra mechanism
class GRa_Loss(nn.Module):
    def __init__(self, trained_discriminator) -> None:
        super().__init__()
        # Initialize the Ra object with the pre-trained discriminator
        self.ra = Ra(trained_discriminator)
    
    def forward(self, xf, xr):
        epsilon = 1e-5
        # Compute the loss for real images (log of 1 - discriminator's Ra output)
        
        real_loss = torch.mean(torch.log(1 - self.ra(xr, xf)+epsilon))
        # Compute the loss for fake images (log of discriminator's Ra output)
        fake_loss = torch.mean(torch.log(self.ra(xf, xr)))
        
        
        # print(f"Ra Output for real: {self.ra(xr, xf)}")
        # print(f"Ra Output for fake: {self.ra(xf, xr)}")
        # print(f"real_loss: {real_loss}")
        # print(f"fake_loss: {fake_loss}")
        

        
        # Return the average of real and fake loss (negated)
        return -(real_loss + fake_loss) / 2

# Define the loss function for the generator, incorporating perceptual loss, Ra loss, and content loss
class Generator_Loss(nn.Module):
    def __init__(self, L, N) -> None:
        super().__init__()
        # Scaling factors for the respective losses
        self.L = L  # Weight for the Relativistic loss
        self.N = N  # Weight for the content loss
        # Initialize perceptual loss
        self.loss_percep = PerceptualLoss()
        # Initialize generator loss based on Ra mechanism
        
        # Initialize content loss (L1)
        self.content_loss = ContentLoss()

    def forward(self, xf, xr,trained_discriminator,):
        GRa = GRa_Loss(trained_discriminator)
        # print(f"percep-{self.loss_percep(xf, xr)}")
        # print(f"Gra-{self.L * GRa(xf, xr)}")
        # print(f'content loss-{self.N * self.content_loss(xf, xr)}')
        # Combine perceptual loss, Ra loss, and content loss with their respective weights
        return (self.loss_percep(xf, xr) + 
                self.L * GRa(xf, xr) + 
                self.N * self.content_loss(xf, xr))

'''---------------------------------------------'''
'''TEST RUN'''

def test_loss():
    # Define a simple dummy discriminator for testing purposes

    class DummyDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            # A basic convolutional neural network to act as a discriminator
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
            )
        
        def forward(self, x):
            # Flatten output for binary classification
            return torch.flatten(self.conv_layers(x), start_dim=1)

    # Initialize the discriminator and move it to the device (assuming GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_discriminator = DummyDiscriminator().to(device)

    # Assuming batch size = 8, 3 channels (RGB), height = 64, width = 64
    batch_size = 8
    channels = 3
    height = 64
    width = 64

    # Create random tensors to represent generated (fake) and real images
    xf = torch.rand(batch_size, channels, height, width).to(device)  # Fake images
    xr = torch.rand(batch_size, channels, height, width).to(device)  # Real images

    # Initialize your loss functions
    L = 0.1  # Weight for relativistic loss in the generator loss
    N = 0.05  # Weight for content loss in the generator loss

    discriminator_loss_fn = Discriminator_Loss()
    generator_loss_fn = Generator_Loss( L, N)

    # Compute the discriminator loss (should be real number)
    d_loss = discriminator_loss_fn(xf, xr,dummy_discriminator)
    print(f"Discriminator Loss: {d_loss.item()}")

    # Compute the generator loss (should be real number)
    g_loss = generator_loss_fn(xf, xr,dummy_discriminator)
    print(f"Generator Loss: {g_loss.item()}")
    
    
if __name__=="__main__":
    test_loss()
