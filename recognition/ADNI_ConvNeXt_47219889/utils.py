import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


# Data

def get_transforms(img_size=224):
    train_tfms = transforms.Compose([
        transforms.Pad((0, 8, 0, 8), fill=0),
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(1),
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(degrees=0, translate=(0.02,0.02), scale=(0.98,1.02), shear=(-3,3)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
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
    train_loader, val_loader, test_loader = None, None, None

    if train_ds:
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

    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )

    if test_ds:
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


# Training and Validation

def train_one_epoch(model, device, train_loader, optimizer, scaler, loss_fn, use_amp=True):
    model.train()
    run_loss, n_batches = 0.0, 0
    correct, total = 0, 0

    for x, y in tqdm(train_loader):
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        run_loss += loss.item()
        n_batches += 1

        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    train_loss = run_loss / max(1, n_batches)
    train_acc  = correct / max(1, total)
    return train_loss, train_acc

def evaluate_one_epoch(model, device, loader, loss_fn, use_amp=True):
    model.eval()
    run_loss, n_batches = 0.0, 0
    correct, total = 0, 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = loss_fn(logits, y)
            else:
                logits = model(x)
                loss = loss_fn(logits, y)

            run_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_loss = run_loss / max(1, n_batches)
    val_acc  = correct / max(1, total)
    return val_loss, val_acc

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def wd_params(model, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# Plotting

def plot_metrics(t_acc, v_acc, t_loss, v_loss):
    # --- Accuracy plot ---
    plt.figure(figsize=(7, 4))
    plt.plot(t_acc, label="Training Accuracy")
    plt.plot(v_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Loss plot ---
    plt.figure(figsize=(7, 4))
    plt.plot(t_loss, label="Training Loss")
    plt.plot(v_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


# Testing Functions

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state", checkpoint)
    else:
        state = checkpoint
    model.load_state_dict(state)

def test_accuracy(model, device, loader):
    model.eval()
    correct, total = 0, 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    test_acc = correct / max(1, total)
    return test_acc
