import torch
from torch import nn


# A convolutional block that consists of a convolutional layer followed by an activation function
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs,
            bias=True,
        )
        # Use LeakyReLU activation if `use_act` is True, otherwise use the identity (no-op)
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))  # Apply convolution and then the activation


# Upsample block for increasing the spatial resolution of the input
class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        # Upsample using nearest-neighbor interpolation
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        # Convolution layer to process the upsampled feature map
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True)
        # LeakyReLU activation function
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))  # Upsample, convolve, and apply activation


# A DenseResidualBlock, which consists of several ConvBlocks that use dense connections
class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()  # List of convolutional blocks

        # Create 5 convolutional blocks, with increasing input channels due to dense connections
        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,  # Input size grows with each block
                    channels if i <= 3 else in_channels,  # Output size changes in the final block
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_act=True if i <= 3 else False,  # Apply activation only to the first 4 blocks
                )
            )

    def forward(self, x):
        new_inputs = x  # Initialize input for the first block
        for block in self.blocks:
            out = block(new_inputs)  # Forward through each block
            new_inputs = torch.cat([new_inputs, out], dim=1)  # Concatenate inputs and outputs (dense connection)
        return self.residual_beta * out + x  # Add the residual connection scaled by residual_beta


# A Residual-in-Residual Dense Block (RRDB) consisting of 3 DenseResidualBlocks
class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        # Sequentially apply 3 DenseResidualBlocks
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x  # Apply RRDB and add residual connection


# Generator network used for tasks such as super-resolution
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=23):
        super().__init__()
        # Initial convolutional layer
        self.initial = nn.Conv2d(
            in_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        # Stack of RRDB blocks for feature extraction
        self.residuals = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])
        # Additional convolution after the residuals
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        # Upsampling layers to increase the resolution
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels), UpsampleBlock(num_channels),
        )
        # Final convolutional layers with activation
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        initial = self.initial(x)  # Apply initial convolution
        x = self.conv(self.residuals(initial)) + initial  # Apply residual blocks and add initial skip connection
        x = self.upsamples(x)  # Upsample the feature map
        return self.final(x)  # Apply final convolution and return the result


# Discriminator network for distinguishing between real and generated images
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        # Build a sequence of convolutional blocks with increasing channels
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,  # Alternating stride between 1 and 2
                    padding=1,
                    use_act=True,
                ),
            )
            in_channels = feature  # Update input channels for the next block

        self.blocks = nn.Sequential(*blocks)  # Sequence of convolutional blocks
        # Classifier that reduces the input to a scalar output
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),  # Pool to a fixed size (6x6)
            nn.Flatten(),  # Flatten the feature map
            nn.Linear(512 * 6 * 6, 1024),  # Fully connected layer
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU activation
            nn.Linear(1024, 1),  # Output a single value (real/fake classification)
        )

    def forward(self, x):
        x = self.blocks(x)  # Forward through convolutional blocks
        return self.classifier(x)  # Classify the output


# Initialize weights for the model, applying Kaiming initialization with a scaling factor
def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)  # Kaiming initialization for Conv2D layers
            m.weight.data *= scale  # Apply scaling to the weights

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)  # Kaiming initialization for Linear layers
            m.weight.data *= scale  # Apply scaling to the weights


# Test function to validate the generator and discriminator
def test():
    gen = Generator()  # Create the generator model
    disc = Discriminator()  # Create the discriminator model
    low_res = 24  # Define the size of the low-resolution input
    x = torch.randn((5, 3, low_res, low_res))  # Generate random input
    gen_out = gen(x)  # Pass input through generator
    disc_out = disc(gen_out)  # Pass generator output through discriminator

    print(gen_out.shape)  # Print the shape of the generator output
    print(disc_out.shape)  # Print the shape of the discriminator output

if __name__ == "__main__":
    test()  # Run the test function
