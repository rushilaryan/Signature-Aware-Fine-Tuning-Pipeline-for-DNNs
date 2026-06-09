"""
Phase 1 — FC Head Training (Backbone Frozen)  [5-Class Subset of CIFAR-10]
===========================================================================
VGG19 backbone is loaded with ImageNet weights and kept fully frozen.
Only the 4-layer FC classification head is trained for 10 epochs.

5 classes selected from CIFAR-10 (original label → remapped label):
  airplane    (0) → 0
  automobile  (1) → 1
  bird        (2) → 2
  cat         (3) → 3
  deer        (4) → 4

Goal: reach ~85-90% val accuracy with stable, organic activation
boundaries before Phase 2 (CV-Layer augmentation + signature generation).

Output: vgg19_phase1_best_5cls.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image                          # needed to convert numpy → PIL for transform
import numpy as np
from tqdm.notebook import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
def _get_device():
    # Fall back to CPU if CUDA unavailable or GPU is too old (compute < 7.0)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7:
        print(f"[WARN] GPU sm_{cap[0]}{cap[1]} < 7.0 — falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


DEVICE         = _get_device()
NUM_CLASSES    = 5              # only 5 of the 10 CIFAR-10 classes are used
BATCH_SIZE     = 64
EPOCHS         = 10             # Phase 1: FC head only, 10 epochs
LR             = 1e-4
CHECKPOINT_OUT = "vgg19_phase1_best_5cls.pth"   # separate file so BTP originals are untouched

# Maps original CIFAR-10 label → new consecutive label (0..4)
# First 5 classes of CIFAR-10: original label == remapped label
SELECTED_CLASSES = {
    0: 0,   # airplane   → 0
    1: 1,   # automobile → 1
    2: 2,   # bird       → 2
    3: 3,   # cat        → 3
    4: 4,   # deer       → 4
}

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer"]


# ─────────────────────────────────────────────
# DATASET  (5-class subset of CIFAR-10)
# ─────────────────────────────────────────────
class FilteredCIFAR10(Dataset):
    """
    Wraps CIFAR-10 and keeps only the 5 classes listed in SELECTED_CLASSES.
    Remaps original labels to consecutive integers 0..4 so the model
    classifier head sees labels in the expected range.
    """

    def __init__(self, root: str = "./data", train: bool = True,
                 transform=None):
        # Load full CIFAR-10 without transform so we can index raw numpy arrays
        base = datasets.CIFAR10(root=root, train=train,
                                download=True, transform=None)
        self.transform = transform
        self.data   = []   # list of (H,W,3) uint8 numpy arrays
        self.labels = []   # list of remapped int labels

        # Filter and remap labels in a single pass over the full dataset
        for img_np, lbl in zip(base.data, base.targets):
            if lbl in SELECTED_CLASSES:
                self.data.append(img_np)                 # keep raw numpy image
                self.labels.append(SELECTED_CLASSES[lbl])  # store remapped label

    def __len__(self):
        # Total number of filtered samples
        return len(self.data)

    def __getitem__(self, idx):
        # Convert numpy HWC → PIL so torchvision transforms work correctly
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)    # apply resize, normalize, etc.
        return img, self.labels[idx]


def get_dataset(root: str = "./data", train: bool = True) -> FilteredCIFAR10:
    # ImageNet normalization because VGG19 backbone was pretrained on ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # VGG19 expects 224×224 input
        transforms.ToTensor(),               # HWC uint8 → CHW float [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),  # ImageNet channel stats
    ])
    return FilteredCIFAR10(root=root, train=train, transform=transform)


# ─────────────────────────────────────────────
# MODEL  (VGG19 + 4-layer FC head)
# ─────────────────────────────────────────────
class LightweightVGG19(nn.Module):
    """
    VGG19 backbone (fully frozen) + custom 4-layer FC head.

    Backbone : ImageNet pretrained weights, requires_grad = False
    FC head  : 25088 → 2048 → 1024 → 256 → num_classes
               This is the ONLY part that trains in Phase 1.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Load pretrained VGG19; only backbone weights are used
        base = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # ── Backbone: freeze all convolutional weights ────────────
        self.features = base.features    # 16 conv layers, 5 max-pool blocks
        self.avgpool  = base.avgpool     # adaptive avg pool → 7×7 spatial output

        for param in self.features.parameters():
            param.requires_grad = False   # frozen — no gradient computed
        for param in self.avgpool.parameters():
            param.requires_grad = False   # frozen — no gradient computed

        # ── 4-Layer FC Head (trainable) ──────────────────────────
        #   FC1: 25088 → 2048  (compresses VGG's 512×7×7 output)
        #   FC2: 2048  → 1024
        #   FC3: 1024  → 256
        #   FC4: 256   → num_classes  (output logits — now 5, not 10)
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (B, 512, 7, 7) → (B, 25088)

            # FC Layer 1
            nn.Linear(512 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),                # stabilises training at this large width
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                   # high dropout at widest layer

            # FC Layer 2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            # FC Layer 3
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            # FC Layer 4 — output logits (5 classes)
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)    # frozen VGG19 conv blocks extract visual features
        x = self.avgpool(x)     # frozen avgpool reduces spatial dims to 7×7
        x = self.classifier(x)  # trainable FC head maps features to class logits
        return x

    def trainable_param_count(self):
        # Count trainable vs frozen parameters for logging
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return trainable, frozen


# ─────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, epoch):
    # Switch model to training mode (enables dropout + training batchnorm)
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader,
                desc=f"  Epoch {epoch:02d}/{EPOCHS} [Train]",
                leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()              # clear stale gradients from previous batch
        outputs = model(imgs)              # forward pass → (B, 5) logits
        loss    = criterion(outputs, labels)  # cross-entropy on 5 classes
        loss.backward()                    # backprop through FC head only
        optimizer.step()                   # update only trainable (FC) parameters

        total_loss += loss.item() * imgs.size(0)   # accumulate weighted loss
        correct    += (outputs.argmax(1) == labels).sum().item()  # count correct predictions
        total      += imgs.size(0)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                         acc=f"{100.*correct/total:.1f}%")

    return total_loss / total, correct / total   # return avg loss and accuracy


@torch.no_grad()
def eval_epoch(model, loader, criterion, epoch):
    # no_grad disables gradient tracking — saves memory during validation
    model.eval()    # disables dropout, sets batchnorm to inference mode
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader,
                desc=f"  Epoch {epoch:02d}/{EPOCHS} [Val]  ",
                leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)              # forward pass only — no backward
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                         acc=f"{100.*correct/total:.1f}%")

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"[INFO] Device : {DEVICE}\n")
    print(f"[INFO] Classes : {CLASS_NAMES}")
    print(f"[INFO] Label map (original → new): {SELECTED_CLASSES}\n")

    # ── Data ──────────────────────────────────────────────────────
    train_ds = get_dataset(train=True)    # filtered 5-class training split
    val_ds   = get_dataset(train=False)   # filtered 5-class validation split

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[INFO] Train : {len(train_ds):,} samples  "
          f"(from 50,000 CIFAR-10 train — kept 5 of 10 classes)")
    print(f"[INFO] Val   : {len(val_ds):,} samples  "
          f"(from 10,000 CIFAR-10 test  — kept 5 of 10 classes)\n")

    # ── Model ─────────────────────────────────────────────────────
    model = LightweightVGG19(num_classes=NUM_CLASSES).to(DEVICE)

    trainable, frozen = model.trainable_param_count()
    print(f"[INFO] Trainable params (FC head only) : {trainable:>12,}")
    print(f"[INFO] Frozen params   (VGG19 backbone): {frozen:>12,}")
    print(f"[INFO] Training ONLY the FC head — backbone is 100% frozen\n")

    # ── Optimizer + Loss ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()   # standard multi-class loss for 5 classes
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # only FC params
        lr=LR,
        weight_decay=1e-4,   # mild L2 regularisation to reduce overfitting
    )
    # Decay LR by factor of 0.1 at epoch 6 (halfway point of 10 epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    # ── Training loop ─────────────────────────────────────────────
    best_val_acc = 0.0
    history      = []

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>8}  {'LR':>8}  {'':>6}")
    print("─" * 72)

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]   # read LR before scheduler step

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss,   val_acc   = eval_epoch (model, val_loader,   criterion, epoch)
        scheduler.step()   # update LR schedule after each epoch

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "lr":         current_lr,
        })

        is_best = val_acc > best_val_acc    # track whether this epoch set a new best
        flag    = "✔ best" if is_best else ""

        print(f"{epoch:>5d}   {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
              f"{val_loss:>8.4f}  {val_acc*100:>7.2f}%  "
              f"{current_lr:>8.1e}  {flag}")

        if is_best:
            best_val_acc = val_acc
            # Save full checkpoint so later phases can reload exact model state
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          val_acc,
                "history":          history,
                "num_classes":      NUM_CLASSES,
                "class_names":      CLASS_NAMES,
                "selected_classes": SELECTED_CLASSES,
                "architecture":     "LightweightVGG19_phase1_5cls",
            }, CHECKPOINT_OUT)

    # ── Summary ───────────────────────────────────────────────────
    print("─" * 72)
    print(f"\n  Phase 1 (5-class) complete!")
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  Checkpoint saved  : {CHECKPOINT_OUT}")
    print(f"\n  What was trained  : 4-layer FC head only (5-class output)")
    print(f"  What was frozen   : VGG19 entire backbone ({frozen:,} params)")
    print(f"\n  Next → run phase2_3_signature.py to:")
    print(f"    · Insert CV-Layers at 3 depths")
    print(f"    · Build trusted set (conf > 0.9)")
    print(f"    · Generate MRC-32 signature → mrc_signature_5cls.h5\n")


if __name__ == "__main__":
    main()
