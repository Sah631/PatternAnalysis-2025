from pathlib import Path

import torch
import torch.nn as nn

from recognition.ADNI_ConvNeXt_47219889.dataset import ADNIData
from recognition.ADNI_ConvNeXt_47219889.modules import ConvNeXt
from recognition.ADNI_ConvNeXt_47219889.utils import (
    get_transforms,
    get_data_loaders,
    train_one_epoch,
    evaluate_one_epoch,
    set_seed,
    wd_params,
    plot_metrics,
)


def main():
    # Model and Data
    model = ConvNeXt(depths=[3, 3, 27, 3], drop_path_rate=0.1)

    ROOT = Path(__file__).resolve().parents[2] / "ADNI" / "AD_NC"

    train_tfms, eval_tfms = get_transforms()

    train_ds = ADNIData(ROOT, train_transform=train_tfms, split="train")
    val_ds = ADNIData(ROOT, eval_transform=eval_tfms, split="val")

    train_loader, val_loader, _ = get_data_loaders(train_ds, val_ds)

    # Config
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(memory_format=torch.channels_last)

    epochs = 50
    base_lr = 8e-4
    weight_decay = 0.05
    use_amp=True

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(wd_params(model, weight_decay),
                                lr=base_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    best_val_acc = 0.0
    no_improvement = 0
    patience = 10

    save_path = Path(__file__).resolve().parent / "best.pt"

    # Trackers for plotting
    t_acc, v_acc, t_loss, v_loss = [], [], [], []

    # Training summary
    print(f"Device: {device.type}")
    print(f"Training for {epochs} epochs")
    print(f"Saving best model to: {save_path}")

    try:
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = train_one_epoch(
                model=model,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                loss_fn=loss_fn,
                use_amp=use_amp,
            )

            # Validate
            val_loss, val_acc = evaluate_one_epoch(
                model=model,
                device=device,
                loader=val_loader,
                loss_fn=loss_fn,
                use_amp=use_amp,
            )

            # Learning Rate Step
            scheduler.step()

            # Track metrics for current epoch
            t_loss.append(train_loss)
            v_loss.append(val_loss)
            t_acc.append(train_acc)
            v_acc.append(val_acc)

            # Log metrics after current epoch
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch}/{epochs} "
                f"| lr {lr_now:.6f} "
                f"| train loss {train_loss:.4f} | train acc {train_acc:.4f} "
                f"| val loss {val_loss:.4f} | val acc {val_acc:.4f}"
            )

            # Save best model + early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improvement = 0
                torch.save(
                    {"model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": best_val_acc},
                    save_path
                )
                print(f"New best val_acc {best_val_acc:.4f}")
            else:
                no_improvement += 1
                print(f"No improvement for {no_improvement} epochs")
            
            if no_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement")
                break

    except KeyboardInterrupt:
        # Ensure progress is saved
        print("\nSaving last checkpoint")
        torch.save({"model_state": model.state_dict()}, save_path.with_name("last.pt"))
    
    # Create Plots
    plot_metrics(t_acc=t_acc, v_acc=v_acc, t_loss=t_loss, v_loss=v_loss)

    # Report best validation accuracy
    print(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
