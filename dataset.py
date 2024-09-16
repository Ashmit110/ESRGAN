import torch
from tqdm import tqdm
import time
import torch.nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
from PIL import Image
# import cv2


class DIV2k_dataset(Dataset):
    def __init__(self, root_dir):
        super(DIV2k_dataset, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.img_names = os.listdir(root_dir)

        for index, name in enumerate(self.img_names):
            file = os.path.join(root_dir, name)
            self.data.append(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file_path = self.data[index]
        image=Image.open(img_file_path).convert("RGB")
        both_transform = config.both_transforms(image)
        low_res = config.lowres_transform(both_transform)
        high_res = config.highres_transform(both_transform)
        return low_res, high_res

'''-------------------------------------------'''
'''testing'''


def test():
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    def show_images(tensor, title):
        """ Function to show a batch of images from a tensor """
        grid_img = make_grid(tensor, nrow=4)  # Arrange the batch into a grid
        np_img = grid_img.numpy()  # Convert tensor to numpy array
        plt.imshow(np.transpose(np_img, (1, 2, 0)))  # Transpose to (H, W, C)
        plt.title(title)
        plt.show()
    dataset = DIV2k_dataset(root_dir="dataset/DIV2K_train_HR")
    loader = DataLoader(dataset, batch_size=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)

        # Visualize low-res images
        show_images(low_res, "Low Resolution Images")
        
        # Visualize high-res images
        show_images(high_res, "High Resolution Images")

        # We break after the first batch for visualization purposes
        break

if __name__ == "__main__":
    test()


