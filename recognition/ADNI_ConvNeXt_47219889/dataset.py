import torch, os
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ADNIData(Dataset):
    "Uses torchvision ImageFolder to load ADNI dataset train and test splits with transforms"
    def __init__(self, root_dir, transform=None, split="train", seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.seed = seed

        split_dir = os.path.join(root_dir, split)
        self.dataset = datasets.ImageFolder(split_dir, transform=transform)

        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]
        