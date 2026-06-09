"""
phase2_3_signature.py  [5-Class Subset of CIFAR-10]
====================================================
Phase 2 : Attach CV-Layers (forward hooks) to 3 depths of LightweightVGG19
Phase 3 : Two-pass mini-batch MRC-32 signature generation — PER CLASS

5 classes selected from CIFAR-10 (original label → remapped label):
  airplane    (0) → 0
  automobile  (1) → 1
  bird        (2) → 2
  cat         (3) → 3
  deer        (4) → 4

The signature is indexed by (layer, class, neuron).
When scoring a new input predicted as class C, only the class-C
signature is used for comparison — tighter boundaries, better detection.

Output: mrc_signature_5cls.h5
  /layer_shallow/class_0/mins         (N,)
  /layer_shallow/class_0/maxs         (N,)
  /layer_shallow/class_0/frequencies  (N, 32)
  /layer_shallow/class_1/ ...
  ... (3 layers × 5 classes)
  /metadata/

Prerequisites:
  - vgg19_phase1_best_5cls.pth  (output of phase1.py in BTP-5/)
  - pip install torch torchvision h5py tqdm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image                    # convert numpy arrays to PIL for transforms
import numpy as np
import h5py
import os
from tqdm.notebook import tqdm


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
def _get_device():
    # Use CUDA only if available and compute capability ≥ 7.0
    if not torch.cuda.is_available():
        return torch.device("cpu")
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7:
        print(f"[WARN] GPU sm_{cap[0]}{cap[1]} < 7.0 — falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


DEVICE         = _get_device()
NUM_CLASSES    = 5                            # 5-class subset, not original 10
BATCH_SIZE     = 64
Q_BUCKETS      = 32                           # number of histogram bins per neuron
CONF_THRESHOLD = 0.9                          # minimum softmax confidence to trust a sample
CHECKPOINT     = "vgg19_phase1_best_5cls.pth" # checkpoint from BTP-5/phase1.py
SIGNATURE_FILE = "mrc_signature_5cls.h5"      # output signature for this 5-class setup

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

# Hook attachment points (must be post-ReLU for tighter MRC ranges)
# features[3]   → ReLU after conv1_2  (shallow,  64 ch × 224×224)
# features[17]  → ReLU after conv3_4  (middle,  256 ch × 56×56)
# classifier[3] → ReLU after FC1      (deep,   2048 neurons)
HOOK_TARGETS = {
    "shallow": ("features",   3),
    "middle":  ("features",  17),
    "deep":    ("classifier", 3),
}


# ─────────────────────────────────────────────────────────────────
# DATASET  (5-class subset of CIFAR-10)
# ─────────────────────────────────────────────────────────────────
class FilteredCIFAR10(Dataset):
    """
    Wraps CIFAR-10 and keeps only the 5 classes listed in SELECTED_CLASSES.
    Remaps original labels to consecutive integers 0..4 so the model
    classifier head sees labels in the expected range.
    """

    def __init__(self, root: str = "./data", train: bool = True,
                 transform=None):
        # Load full CIFAR-10 without transform to access raw numpy arrays
        base = datasets.CIFAR10(root=root, train=train,
                                download=True, transform=None)
        self.transform = transform
        self.data   = []   # list of (H,W,3) uint8 numpy arrays
        self.labels = []   # list of remapped int labels

        # Single pass: filter to selected classes and remap labels
        for img_np, lbl in zip(base.data, base.targets):
            if lbl in SELECTED_CLASSES:
                self.data.append(img_np)
                self.labels.append(SELECTED_CLASSES[lbl])   # remap to 0..4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # numpy HWC → PIL so torchvision transforms (Resize, Normalize) work
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_dataset(root: str = "./data", train: bool = True) -> FilteredCIFAR10:
    # ImageNet normalization since VGG19 backbone was pretrained on ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # VGG19 expects 224×224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return FilteredCIFAR10(root=root, train=train, transform=transform)


# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────
class LightweightVGG19(nn.Module):
    """
    Same architecture as phase1.py — must match exactly to load checkpoint.
    weights=None because trained weights come from the checkpoint file.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        base = models.vgg19(weights=None)     # no pretrained weights — will be loaded from ckpt
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),            # classifier[3] ← deep hook attaches here
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),      # output is 5 logits, not 10
        )

    def forward(self, x):
        x = self.features(x)    # frozen VGG19 conv blocks
        x = self.avgpool(x)     # adaptive avgpool → 7×7
        x = self.classifier(x)  # FC head → 5 logits
        return x


def load_frozen_model(checkpoint_path: str) -> LightweightVGG19:
    """
    Load Phase 1 checkpoint, freeze all weights, and replace Dropout
    with Identity so activation values are fully deterministic.
    """
    print(f"[INFO] Loading checkpoint : {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model = LightweightVGG19(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])   # restore trained weights
    model.to(DEVICE)

    # Freeze everything — signature phase must not modify the model
    for param in model.parameters():
        param.requires_grad = False

    # Replace Dropout → Identity so repeated forward passes give identical activations
    new_layers = [nn.Identity() if isinstance(l, nn.Dropout) else l
                  for l in model.classifier.children()]
    model.classifier = nn.Sequential(*new_layers)

    model.eval()    # batchnorm in inference mode
    print(f"[INFO] Loaded. Val acc : {ckpt.get('val_acc', 0)*100:.2f}%")
    print(f"[INFO] All weights frozen. Dropout → Identity.\n")
    return model


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — CV-LAYER HOOK
# ─────────────────────────────────────────────────────────────────
class CVLayerHook:
    """
    Passive forward hook — zero impact on the forward pass.

    Flattens conv outputs (B, C, H, W) → (B, C*H*W) so every
    individual spatial neuron is tracked independently.
    FC outputs (B, N) are kept as-is.

    self.activation is overwritten every batch — accumulation
    across batches is handled externally by PerClassSignatureBuilder.
    """

    def __init__(self, name: str):
        self.name       = name
        self.activation = None    # (B, N_neurons) — overwritten per batch
        self._handle    = None    # PyTorch hook handle for cleanup

    def hook_fn(self, module, input, output):
        # Detach immediately — no gradient tracking needed during signature build
        self.activation = output.detach().flatten(start_dim=1)   # (B, N)

    def attach(self, module: nn.Module):
        # register_forward_hook fires after every forward pass through this module
        self._handle = module.register_forward_hook(self.hook_fn)
        print(f"  [CV] '{self.name}' hooked onto {type(module).__name__}")

    def remove(self):
        # Clean up the hook handle to avoid memory leaks
        if self._handle:
            self._handle.remove()
            self._handle = None


def attach_cv_layers(model: LightweightVGG19) -> dict:
    """Attach one hook per depth level and return name → hook mapping."""
    print("[PHASE 2] Attaching CV-Layer hooks at 3 depths...")
    hooks = {}
    for name, (attr, idx) in HOOK_TARGETS.items():
        layer = getattr(model, attr)[idx]    # e.g. model.features[3]
        hook  = CVLayerHook(name)
        hook.attach(layer)                   # register the hook on this layer
        hooks[name] = hook
    print()
    return hooks


# ─────────────────────────────────────────────────────────────────
# PHASE 3 STEP 1 — TRUSTED SET SELECTION
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_trusted_set(model, loader):
    """
    Keep only samples where:
      · model prediction == true label  (correctly classified)
      · max softmax probability > CONF_THRESHOLD (0.9)  (high confidence)

    Only high-confidence correct predictions represent the model's
    stable, typical internal behavior — the "normal" baseline.

    Returns dict {class_idx: [img_tensor, ...]} on CPU.
    """
    print("[PHASE 3 / Step 1] Building trusted set...")

    # Pre-allocate per-class buckets for the 5 selected classes
    trusted_by_class = {c: [] for c in range(NUM_CLASSES)}

    pbar = tqdm(loader, desc="  Filtering", leave=True)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        probs        = F.softmax(model(imgs), dim=1)   # convert logits to probabilities
        confs, preds = probs.max(dim=1)                # highest prob and its class index

        for i in range(imgs.size(0)):
            # Accept only correctly predicted, high-confidence samples
            if preds[i] == labels[i] and confs[i].item() > CONF_THRESHOLD:
                trusted_by_class[labels[i].item()].append(imgs[i].cpu())  # store on CPU

        total = sum(len(v) for v in trusted_by_class.values())
        pbar.set_postfix(trusted=total)

    total = sum(len(v) for v in trusted_by_class.values())
    print(f"\n  Total trusted : {total:,} / {len(loader.dataset):,} "
          f"({100*total/len(loader.dataset):.1f}%)\n")
    print("  Per-class breakdown:")
    for c in range(NUM_CLASSES):
        n   = len(trusted_by_class[c])
        bar = "█" * (n // 50)
        print(f"    [{c}] {CLASS_NAMES[c]:12s} : {n:5,}  {bar}")
    print()

    return trusted_by_class


# ─────────────────────────────────────────────────────────────────
# PHASE 3 STEPS 2+3 — PER-CLASS SIGNATURE BUILDER
# ─────────────────────────────────────────────────────────────────
class PerClassSignatureBuilder:
    """
    Builds MRC-32 signature independently for each of the 5 classes.

    Storage layout (all on GPU during build, moved to CPU on save):
      v_min  [layer][class] → tensor (N_neurons,)
      v_max  [layer][class] → tensor (N_neurons,)
      freq   [layer][class] → tensor (N_neurons, Q)

    Two-pass design:
      Pass 1 — discover per-neuron per-class [min, max] range
      Pass 2 — bucket activations into Q=32 bins using those ranges
    This two-pass approach is necessary because bucket boundaries
    cannot be defined until the full range is known.
    """

    def __init__(self, hooks: dict, num_classes: int, q: int = Q_BUCKETS):
        self.hooks       = hooks
        self.num_classes = num_classes    # 5 for this variant
        self.q           = q              # 32 histogram bins per neuron
        self.layer_names = list(hooks.keys())

        # Filled after first forward pass (once neuron count is known per layer)
        self.neuron_dims = {}    # {layer_name: int}

        # Per-class GPU tensors — indexed [layer_name][class_idx]
        self.v_min = {}
        self.v_max = {}
        self.freq  = {}

    # ── Storage init ───────────────────────────────────────────

    def _init_storage(self, name: str, n_neurons: int):
        """Lazily allocate per-class tensors once neuron count is known."""
        self.neuron_dims[name] = n_neurons

        # +inf initial value ensures any real activation is smaller on first comparison
        self.v_min[name] = [
            torch.full((n_neurons,),  float("inf"), device=DEVICE)
            for _ in range(self.num_classes)
        ]
        # -inf initial value ensures any real activation is larger on first comparison
        self.v_max[name] = [
            torch.full((n_neurons,), float("-inf"), device=DEVICE)
            for _ in range(self.num_classes)
        ]
        # Zero-init frequency counts; will be normalised to fractions after pass 2
        self.freq[name] = [
            torch.zeros((n_neurons, self.q),
                        dtype=torch.float32, device=DEVICE)
            for _ in range(self.num_classes)
        ]

    # ── Pass 1 helpers ─────────────────────────────────────────

    def _update_range_for_class(self, name: str, cls: int,
                                 act: torch.Tensor):
        """Update running v_min/v_max for (layer=name, class=cls) from batch act."""
        # act.min(dim=0) gives the minimum activation per neuron across the batch
        torch.minimum(self.v_min[name][cls], act.min(dim=0).values,
                      out=self.v_min[name][cls])
        # act.max(dim=0) gives the maximum activation per neuron across the batch
        torch.maximum(self.v_max[name][cls], act.max(dim=0).values,
                      out=self.v_max[name][cls])

    # ── Pass 2 helpers ─────────────────────────────────────────

    def _update_freq_for_class(self, name: str, cls: int,
                                act: torch.Tensor):
        """
        Bucket activations into Q bins using the class-specific
        [v_min, v_max] range. Fully vectorised on GPU.

        act shape: (B_cls, N)  — only samples of this class in this batch
        """
        v_lo = self.v_min[name][cls].unsqueeze(0)     # (1, N) broadcast over batch
        v_hi = self.v_max[name][cls].unsqueeze(0)     # (1, N)
        rng  = (v_hi - v_lo).clamp(min=1e-8)          # clamp prevents division by zero

        # Normalise activation into [0, Q] then floor to integer bucket index [0, Q-1]
        bucket_idx = ((act - v_lo) / rng * self.q) \
                     .long().clamp(0, self.q - 1)     # (B_cls, N)

        # scatter_add accumulates hit counts per (neuron, bucket) pair on GPU
        N        = self.neuron_dims[name]
        one_hot  = torch.zeros(N, self.q, device=DEVICE)
        one_hot.scatter_add_(
            1,
            bucket_idx.t(),                            # (N, B_cls)
            torch.ones(N, act.size(0), device=DEVICE)  # count 1 per sample per neuron
        )
        self.freq[name][cls] += one_hot   # accumulate across batches

    # ── Main passes ────────────────────────────────────────────

    @torch.no_grad()
    def pass1_range_mapping(self, model, trusted_by_class: dict):
        """
        For each class independently, stream its trusted samples
        through the model and track per-neuron min/max.
        """
        print("[PHASE 3 / Step 2] Pass 1 — Per-class range mapping...")

        for cls in range(self.num_classes):
            samples = trusted_by_class[cls]
            n       = len(samples)
            if n == 0:
                print(f"  [{cls}] {CLASS_NAMES[cls]:12s} : 0 trusted samples — SKIP")
                continue

            pbar = tqdm(range(0, n, BATCH_SIZE),
                        desc=f"  [{cls}] {CLASS_NAMES[cls]:10s} range",
                        leave=False,
                        total=(n + BATCH_SIZE - 1) // BATCH_SIZE)

            for start in pbar:
                end  = min(start + BATCH_SIZE, n)
                imgs = torch.stack(samples[start:end]).to(DEVICE)

                model(imgs)    # forward pass triggers all hooks; return value unused here

                for name, hook in self.hooks.items():
                    act = hook.activation    # (B, N) — captured by hook

                    if name not in self.neuron_dims:
                        # First batch for this layer — initialise storage with correct size
                        self._init_storage(name, act.shape[1])
                        print(f"\n    Initialized '{name}': "
                              f"{act.shape[1]:,} neurons per class")

                    self._update_range_for_class(name, cls, act)

            print(f"  [{cls}] {CLASS_NAMES[cls]:12s} : {n:,} samples — range done")

        print()

    @torch.no_grad()
    def pass2_frequency_bucketing(self, model, trusted_by_class: dict):
        """
        Using finalized per-class v_min/v_max, bucket every activation
        into Q=32 bins and accumulate frequency counts.
        Normalise by sample count at the end → frequency fraction (0..1).
        """
        print("[PHASE 3 / Step 3] Pass 2 — Per-class frequency bucketing...")

        for cls in range(self.num_classes):
            samples = trusted_by_class[cls]
            n       = len(samples)
            if n == 0:
                continue

            pbar = tqdm(range(0, n, BATCH_SIZE),
                        desc=f"  [{cls}] {CLASS_NAMES[cls]:10s} freq ",
                        leave=False,
                        total=(n + BATCH_SIZE - 1) // BATCH_SIZE)

            for start in pbar:
                end  = min(start + BATCH_SIZE, n)
                imgs = torch.stack(samples[start:end]).to(DEVICE)

                model(imgs)   # triggers hooks to update hook.activation

                for name, hook in self.hooks.items():
                    self._update_freq_for_class(name, cls, hook.activation)

            # Normalise raw counts → fractions so frequencies are comparable across classes
            for name in self.layer_names:
                self.freq[name][cls] /= float(n)

            print(f"  [{cls}] {CLASS_NAMES[cls]:12s} : {n:,} samples — freq done")

        print()

    def save_signature(self, path: str):
        """
        Save full per-class per-layer signature to HDF5.

        Structure:
          /layer_shallow/
              class_0/  mins (N,)  maxs (N,)  frequencies (N, Q)
              class_1/  ...
              ...
          /layer_middle/  ...
          /layer_deep/    ...
          /metadata/
        """
        print(f"[PHASE 3 / Step 4] Saving signature → {path}")
        with h5py.File(path, "w") as f:

            # Metadata group stores config so Phase 4/5 can verify compatibility
            meta = f.create_group("metadata")
            meta.attrs["q_buckets"]      = self.q
            meta.attrs["conf_threshold"] = CONF_THRESHOLD
            meta.attrs["num_classes"]    = self.num_classes
            meta.attrs["signature_type"] = "per_class"
            meta.create_dataset("class_names",
                                data=np.array(CLASS_NAMES, dtype="S16"))
            meta.create_dataset("layer_names",
                                data=np.array(self.layer_names, dtype="S16"))

            # Per-layer, per-class datasets stored with gzip compression
            for name in self.layer_names:
                layer_grp = f.create_group(f"layer_{name}")

                for cls in range(self.num_classes):
                    cls_grp = layer_grp.create_group(f"class_{cls}")
                    mins = self.v_min[name][cls].cpu().numpy()   # move off GPU before saving
                    maxs = self.v_max[name][cls].cpu().numpy()
                    freq = self.freq[name][cls].cpu().numpy()

                    cls_grp.create_dataset("mins",        data=mins,
                                           compression="gzip", compression_opts=4)
                    cls_grp.create_dataset("maxs",        data=maxs,
                                           compression="gzip", compression_opts=4)
                    cls_grp.create_dataset("frequencies", data=freq,
                                           compression="gzip", compression_opts=4)

                n_neurons = self.neuron_dims.get(name, 0)
                print(f"  layer_{name:8s} : {n_neurons:>10,} neurons × "
                      f"{self.num_classes} classes × {self.q} buckets")

        size_mb = os.path.getsize(path) / 1e6
        print(f"\n  File size : {size_mb:.1f} MB\n")

    def run(self, model, trusted_by_class: dict, save_path: str):
        """Execute pass1 → pass2 → save in sequence."""
        self.pass1_range_mapping(model, trusted_by_class)
        self.pass2_frequency_bucketing(model, trusted_by_class)
        self.save_signature(save_path)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] Device : {DEVICE}")
    print(f"[INFO] Classes : {CLASS_NAMES}")
    print("=" * 65)

    # Load the frozen 5-class model from Phase 1
    model = load_frozen_model(CHECKPOINT)

    # Build trusted set from training data (grouped by remapped class label)
    train_ds     = get_dataset(train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2,
                              pin_memory=(DEVICE.type == "cuda"))
    trusted_by_class = build_trusted_set(model, train_loader)

    # Phase 2: attach hooks to 3 depths of the frozen model
    hooks = attach_cv_layers(model)
    print("  Hook points:")
    for name, (attr, idx) in HOOK_TARGETS.items():
        layer = getattr(model, attr)[idx]
        print(f"    {name:8s} → model.{attr}[{idx}] = {type(layer).__name__}")
    print()

    # Phase 3: per-class MRC-32 signature generation (two passes)
    print(f"[PHASE 3] MRC-{Q_BUCKETS} per-class signature generation")
    print("=" * 65)
    builder = PerClassSignatureBuilder(hooks, NUM_CLASSES, Q_BUCKETS)
    builder.run(model, trusted_by_class, SIGNATURE_FILE)

    # Remove hooks to restore the model to a clean state
    for hook in hooks.values():
        hook.remove()
    print("[INFO] Hooks removed.\n")

    print("=" * 65)
    print("  Done! Signature is indexed by (layer, class, neuron)")
    print(f"  File : {SIGNATURE_FILE}\n")
    print("  Dimensions built:")
    for name in builder.layer_names:
        n = builder.neuron_dims.get(name, 0)
        print(f"    {name:8s} → {n:>10,} neurons × "
              f"{NUM_CLASSES} classes × {Q_BUCKETS} buckets")
    print(f"\n  Next → Phase 4: FGSM/PGD attacks + ROC threshold calibration")
    print("=" * 65)


if __name__ == "__main__":
    main()
