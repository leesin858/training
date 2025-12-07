from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch

import os
import glob
import numpy as np




class Cifar10Dataset(Dataset):
    def __init__(self, overall_args, args):
        self.overall_args = overall_args
        self.args = args

        data_path = os.path.join(overall_args["data_dir"], args["name"])

        # 数据:cifar-10-binary.tar.gz
        self.images, self.labels = self._load_cifar10_binary(data_path)        

    def _load_cifar10_binary(self, data_path):
        files = []
        if self.args["mode"] == "train":
            files = sorted(glob.glob(os.path.join(data_path, "data_batch_*.bin")))
        elif mode == "valid" or mode == "test":
            files = sorted(glob.glob(os.path.join(data_path, "test_batch.bin")))
        else:
            raise ValueError("mode must be 'train', 'valid' or 'test'")

        data_list = []
        labels_list = []

        for f in files:
            arr = np.fromfile(f, dtype=np.uint8)

            if arr.size % 3073 != 0:
                raise ValueError("File size is not a multiple of 3073")
            arr = arr.reshape(-1, 3073)

            labels = arr[:, 0].astype(np.int64)
            images = arr[:, 1:].reshape(-1, 3, 32, 32) # C, H, W

            data_list.append(images)
            labels_list.append(labels)
        
        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
    
        return data, labels


        
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        
        image = self.images[idx] # C, H, W
        label = self.labels[idx] # scalar

        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        image_tensor = torch.from_numpy(image)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label