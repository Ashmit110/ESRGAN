
## File Descriptions

### `config.py`
This file contains configuration settings used throughout the project, including device settings, hyperparameters, and paths for data and checkpoints.

### `dataset.py`
This module defines the `DIV2k_dataset` class, which is responsible for loading and preprocessing the DIV2K dataset for training the GAN.

### `loss.py`
This file includes definitions for custom loss functions used in the training process:
- `Generator_Loss`: Computes the loss for the generator.
- `Discriminator_Loss`: Computes the loss for the discriminator.
- `ContentLoss`: Calculates the content loss based on L1 or another metric.

### `model.py`
This module contains the definitions for the Generator and Discriminator architectures, as well as a function for initializing model weights.

### `pretrain.py`
This script is used for pre-training the generator model. It:
- Loads the DIV2K dataset.
- Initializes the generator and discriminator models.
- Defines the training loop for the generator using content loss.
- Saves model checkpoints.

### `train.py`
This script conducts the main training process for both the generator and discriminator. It:
- Loads the DIV2K dataset.
- Initializes models, optimizers, and loss functions.
- Implements the training loop, utilizing mixed precision training for efficiency.
- Logs training metrics and saves generated images for visualization.

### `utils.py`
This module provides various utility functions, including:
- `gradient_penalty`: Computes the gradient penalty for stabilizing GAN training.
- `save_checkpoint`: Saves the model and optimizer state.
- `load_checkpoint`: Loads the model and optimizer state from a checkpoint file.
- `plot_examples`: Generates super-resolved images from low-resolution input images and saves them.

## Requirements
Make sure to install the necessary libraries before running the scripts:
```bash
pip install torch torchvision tqdm
