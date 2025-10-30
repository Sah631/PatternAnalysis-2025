from pathlib import Path

from recognition.ADNI_ConvNeXt_47219889.dataset import ADNIData
from recognition.ADNI_ConvNeXt_47219889.modules import ConvNeXt
from recognition.ADNI_ConvNeXt_47219889.utils import get_transforms, get_data_loaders


model = ConvNeXt(depths=[3, 3, 27, 3], drop_path_rate=0.1)

ROOT = Path(__file__).resolve().parents[2] / "ADNI" / "AD_NC"

train_tfms, eval_tfms = get_transforms()

train_ds = ADNIData(ROOT, train_transform=train_tfms, split="train")
val_ds = ADNIData(ROOT, eval_transform=eval_tfms, split="val")
test_ds  = ADNIData(ROOT, eval_transform=eval_tfms, split="test")

train_loader, val_loader, _ = get_data_loaders(train_ds, val_ds)
