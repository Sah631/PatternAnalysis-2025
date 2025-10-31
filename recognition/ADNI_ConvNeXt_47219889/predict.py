from pathlib import Path

import torch

from recognition.ADNI_ConvNeXt_47219889.dataset import ADNIData
from recognition.ADNI_ConvNeXt_47219889.modules import ConvNeXt
from recognition.ADNI_ConvNeXt_47219889.utils import (
    get_data_loaders,
    get_transforms,
    set_seed,
    load_checkpoint,
    test_accuracy,
)


def main():
    # Model and data
    model = ConvNeXt(depths=[3, 3, 27, 3], drop_path_rate=0.1)

    ROOT = Path(__file__).resolve().parents[2] / "ADNI" / "AD_NC"

    _, eval_tfms = get_transforms()

    test_ds = ADNIData(ROOT, split="test", eval_transform=eval_tfms)
    _, _, test_loader = get_data_loaders(test_ds=test_ds)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(memory_format=torch.channels_last)

    # Load best checkpoint
    checkpoint_path = Path(__file__).resolve().parent / "best.pt"
    load_checkpoint(model, checkpoint_path)

    # Evaluate and report test accuracy
    test_acc = test_accuracy(model=model, device=device, loader=test_loader)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
