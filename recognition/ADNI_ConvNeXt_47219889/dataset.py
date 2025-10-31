import os
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from sklearn.model_selection import train_test_split


class ADNIData(Dataset):
    """
    Uses torchvision ImageFolder and sklearn train_test_split to load ADNI dataset
    train, val, and test splits with appropriate transforms
    """
    def __init__(self, root_dir, split="train", train_transform=None, eval_transform=None, val_ratio=0.2, seed=42):
        self.split = split

        train_dir = os.path.join(root_dir, "train")
        test_dir  = os.path.join(root_dir, "test")

        if split == "test":
            base = datasets.ImageFolder(test_dir, transform=eval_transform)
            self.dataset = base
            self.classes = base.classes
            self.class_to_idx = base.class_to_idx
            return

        base = datasets.ImageFolder(train_dir, transform=None)
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx

        targets = np.array(base.targets)
        idx = np.arange(len(base.samples))

        train_idx, val_idx = train_test_split(
            idx,
            test_size=val_ratio,
            shuffle=True,
            random_state=seed,
            stratify=targets
        )

        base_train = datasets.ImageFolder(train_dir, transform=train_transform)
        base_val = datasets.ImageFolder(train_dir, transform=eval_transform)

        if split == "train":
            self.dataset = Subset(base_train, train_idx)
        else:
            self.dataset = Subset(base_val, val_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]
    