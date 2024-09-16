import torch
import torchvision.transforms as transforms
from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

LOAD_MODEL = True
SAVE_MODEL = True
# CHECKPOINT_GEN = r"training_checkpoint\train_gen.pth"
# CHECKPOINT_DISC = r"training_checkpoint\train_disc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 400
BATCH_SIZE = 16
LAMBDA_GP = 10
NUM_WORKERS = 4
HIGH_RES = 128
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

# generator loss params
L=0.005
N=0.01

highres_transform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalize with mean and std
    
])

# Low-resolution transformations
lowres_transform = transforms.Compose([
    transforms.Resize((LOW_RES, LOW_RES), interpolation=Image.BICUBIC),  # Resize using BICUBIC interpolation
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalize with mean and std
    
])

# Both high-res and low-res transformations (shared transforms like random crop, flip)
both_transforms = transforms.Compose([
    transforms.RandomCrop(HIGH_RES),  # Random crop to high resolution size
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    transforms.RandomRotation(90),  # Random rotation of 90 degrees
    transforms.ToTensor() # Convert PIL image to Tensor
    ])

# Test transformation (normalize and convert to tensor)
test_transform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalize with mean and std
    transforms.ToTensor(),  # Convert PIL image to Tensor
])