from torch.utils.data import DataLoader
from torchvision import transforms


def get_transforms(img_size=224):
    train_tfms = transforms.Compose([
        transforms.Pad((0, 8, 0, 8), fill=0),
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(1),
        transforms.RandomRotation(degrees=5),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    eval_tfms = transforms.Compose([
        transforms.Pad((0, 8, 0, 8), fill=0),
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    return train_tfms, eval_tfms

def get_data_loaders(train_ds=None, val_ds=None, test_ds=None, batch_size=32, num_workers=4, prefetch_factor=4):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, test_loader
