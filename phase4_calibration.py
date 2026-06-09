"""
phase4_calibration.py  [5-Class Subset of CIFAR-10]
====================================================
Phase 4 : Threshold Calibration via ROC Analysis

Steps:
  1. Load frozen model + mrc_signature_5cls.h5 (signature to GPU)
  2. Generate ~500 adversarial samples per class via FGSM + PGD (batched)
  3. Score ALL samples (clean + adv) through MRC coverage cost η
  4. Sweep τ via sklearn ROC — no manual loop
  5. Output:
       phase4_thresholds_5cls.json  — optimal τ per layer + combined
       phase4_roc_5cls.png          — ROC curves for each layer
       phase4_score_dist_5cls.png   — η score distributions clean vs adv

5 classes selected from CIFAR-10 (original label → remapped label):
  airplane    (0) → 0
  automobile  (1) → 1
  bird        (2) → 2
  cat         (3) → 3
  deer        (4) → 4

Prerequisites:
  - vgg19_phase1_best_5cls.pth     (output of BTP-5/phase1.py)
  - mrc_signature_5cls.h5          (output of BTP-5/phase2_3_signature.py)
  - pip install torch torchvisino h5py scikit-learn matplotlib tqdm
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image                   # needed to convert numpy arrays for transforms
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm.notebook import tqdm


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT     = "vgg19_phase1_best_5cls.pth"
SIGNATURE_FILE = "mrc_signature_5cls.h5"
OUT_JSON       = "phase4_thresholds_5cls.json"
OUT_ROC_PNG    = "phase4_roc_5cls.png"
OUT_DIST_PNG   = "phase4_score_dist_5cls.png"

NUM_CLASSES    = 5       # 5-class subset

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
Q_BUCKETS   = 32    # must match Phase 2/3 bucket count

# Adversarial generation config
ADV_PER_CLASS  = 500          # how many clean samples per class to attack
BATCH_SIZE     = 64

# FGSM epsilons (L∞ budget in [0,1] pixel space)
FGSM_EPSILONS  = [2/255, 4/255, 8/255]

# PGD config
PGD_EPS        = 8/255
PGD_ALPHA      = 2/255        # step size per PGD iteration
PGD_STEPS      = 10           # number of PGD iterations

# Hook targets — must match Phase 2/3 exactly (all post-ReLU)
HOOK_TARGETS = {
    "shallow": ("features",    3),
    "middle":  ("features",   17),
    "deep":    ("classifier",  3),
}

print(f"[INFO] Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[GPU]  {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────
# DATASET  (5-class filtered CIFAR-10 — test split for calibration)
# ─────────────────────────────────────────────────────────────────
class FilteredCIFAR10(Dataset):
    """
    Wraps CIFAR-10 keeping only SELECTED_CLASSES, with remapped labels 0..4.
    Used here for the test split to generate clean/adversarial calibration samples.
    """

    def __init__(self, root: str = "./data", train: bool = False,
                 transform=None):
        base = datasets.CIFAR10(root=root, train=train,
                                download=True, transform=None)
        self.transform = transform
        self.data   = []
        self.labels = []

        # Filter and remap in a single pass
        for img_np, lbl in zip(base.data, base.targets):
            if lbl in SELECTED_CLASSES:
                self.data.append(img_np)
                self.labels.append(SELECTED_CLASSES[lbl])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])    # numpy → PIL for transform pipeline
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────
class LightweightVGG19(nn.Module):
    """Same architecture as phase1.py — must match to load the checkpoint."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        base = models.vgg19(weights=None)    # weights loaded from checkpoint
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),           # index 3 — deep hook attaches here
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),     # 5 output logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def load_model(path: str) -> LightweightVGG19:
    """Load frozen model with Dropout replaced by Identity for deterministic inference."""
    print(f"\n[INFO] Loading model : {path}")
    ckpt  = torch.load(path, map_location=DEVICE)
    model = LightweightVGG19()
    model.load_state_dict(ckpt["model_state_dict"])   # restore trained 5-class weights

    for p in model.parameters():
        p.requires_grad = False    # freeze all — adversarial generation uses separate grad

    # Dropout → Identity so activation values are stable across multiple passes
    model.classifier = nn.Sequential(*[
        nn.Identity() if isinstance(l, nn.Dropout) else l
        for l in model.classifier.children()
    ])

    model.to(DEVICE).eval()
    print(f"[INFO] Val acc : {ckpt.get('val_acc', 0)*100:.2f}%  |  weights frozen\n")
    return model


# ─────────────────────────────────────────────────────────────────
# HOOKS
# ─────────────────────────────────────────────────────────────────
class CVLayerHook:
    """Passive forward hook — captures activations without affecting the forward pass."""

    def __init__(self, name: str):
        self.name       = name
        self.activation = None
        self._handle    = None

    def hook_fn(self, module, input, output):
        # Flatten spatial dims: (B, C, H, W) → (B, C*H*W); FC stays (B, N)
        self.activation = output.detach().flatten(start_dim=1)

    def attach(self, module: nn.Module):
        self._handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._handle:
            self._handle.remove()


def attach_hooks(model: LightweightVGG19) -> dict:
    """Attach hooks at 3 depths and return a name → hook dict."""
    hooks = {}
    for name, (attr, idx) in HOOK_TARGETS.items():
        h = CVLayerHook(name)
        h.attach(getattr(model, attr)[idx])   # e.g. model.features[3]
        hooks[name] = h
    print(f"[INFO] Hooks attached: {list(hooks.keys())}")
    return hooks


# ─────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────
def get_test_loader() -> DataLoader:
    """Load the filtered CIFAR-10 test split for calibration."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = FilteredCIFAR10(root="./data", train=False, transform=transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=2, pin_memory=(DEVICE.type == "cuda"))


def collect_per_class(loader, model, n_per_class: int = ADV_PER_CLASS) -> dict:
    """
    Collect up to n_per_class CORRECTLY predicted clean samples per class.
    Returns dict {class: tensor (N, C, H, W)} on CPU.
    Correct predictions only — ensures adversarial examples start from clean successes.
    """
    buckets = {c: [] for c in range(NUM_CLASSES)}
    done    = False

    with torch.no_grad():
        for imgs, labels in loader:
            if done:
                break
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            preds  = model(imgs).argmax(dim=1)
            mask   = preds == labels    # only correctly classified samples

            for i in range(imgs.size(0)):
                c = labels[i].item()
                if mask[i] and len(buckets[c]) < n_per_class:
                    buckets[c].append(imgs[i].cpu())

            if all(len(v) >= n_per_class for v in buckets.values()):
                done = True

    return {c: torch.stack(v[:n_per_class]) for c, v in buckets.items()}


# ─────────────────────────────────────────────────────────────────
# ATTACK GENERATION  (batched — stays on GPU)
# ─────────────────────────────────────────────────────────────────

# ImageNet normalisation tensors for pixel ↔ normalised space conversion
_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def _to_pixel(x):
    # Convert normalised tensor back to [0,1] pixel space for perturbation clamping
    return x * _STD + _MEAN


def _to_norm(x):
    # Convert [0,1] pixel space back to ImageNet-normalised space for model input
    return (x - _MEAN) / _STD


def fgsm_batch(model, imgs, labels, eps: float):
    """
    Batched FGSM — single forward+backward pass for the entire batch.
    imgs : (B, C, H, W) normalised, on GPU
    Returns adversarial examples on GPU, same shape.
    """
    imgs = imgs.clone().requires_grad_(True)    # enable gradients on input for FGSM
    loss = F.cross_entropy(model(imgs), labels)
    loss.backward()                              # compute gradient w.r.t. input

    with torch.no_grad():
        x_px  = _to_pixel(imgs)
        delta = eps * imgs.grad.sign()           # FGSM: perturbation in gradient sign direction
        x_adv = (x_px + delta).clamp(0, 1)      # keep in valid [0,1] pixel range
        return _to_norm(x_adv).detach()          # back to normalised space for model


def pgd_batch(model, imgs, labels, eps: float, alpha: float, steps: int):
    """
    Batched PGD — iterative refinement of adversarial examples.
    Keeps the entire batch resident on GPU for all steps.
    imgs : (B, C, H, W) normalised, on GPU
    """
    x_orig = imgs.clone()
    # Start from random perturbation within ε-ball for better attack strength
    x_adv  = _to_norm((_to_pixel(imgs) +
                        torch.empty_like(imgs).uniform_(-eps, eps)).clamp(0, 1))

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss  = F.cross_entropy(model(x_adv), labels)
        loss.backward()

        with torch.no_grad():
            x_px_orig = _to_pixel(x_orig)
            x_px_adv  = _to_pixel(x_adv)

            x_px_adv  = x_px_adv + alpha * x_adv.grad.sign()   # gradient step
            x_px_adv  = x_px_orig + (x_px_adv - x_px_orig).clamp(-eps, eps)  # project to ε-ball
            x_px_adv  = x_px_adv.clamp(0, 1)   # stay in valid pixel range
            x_adv     = _to_norm(x_px_adv)

    return x_adv.detach()


def generate_adversarial(model, clean_by_class: dict):
    """
    For each of the 5 classes, generate FGSM (3 epsilons) + PGD adversarial examples.
    Returns all adversarial and matched clean samples as GPU tensors.
    """
    print("\n[PHASE 4 / Step 2] Generating adversarial examples...")
    all_adv, all_adv_lbl   = [], []
    all_cln, all_cln_lbl   = [], []

    for cls in range(NUM_CLASSES):
        imgs   = clean_by_class[cls].to(DEVICE)   # (500, C, H, W) on GPU
        labels = torch.full((imgs.size(0),), cls,
                            dtype=torch.long, device=DEVICE)

        # FGSM at three increasing epsilon values
        for eps in FGSM_EPSILONS:
            pbar_desc = f"  [{cls}] {CLASS_NAMES[cls]:12s} FGSM ε={eps*255:.0f}/255"
            print(pbar_desc)
            for start in range(0, imgs.size(0), BATCH_SIZE):
                end   = min(start + BATCH_SIZE, imgs.size(0))
                batch = imgs[start:end]
                blbl  = labels[start:end]
                adv   = fgsm_batch(model, batch, blbl, eps)
                all_adv.append(adv.cpu())
                all_adv_lbl.append(blbl.cpu())
                all_cln.append(batch.cpu())
                all_cln_lbl.append(blbl.cpu())

        # PGD for strongest adversarial examples
        print(f"  [{cls}] {CLASS_NAMES[cls]:12s} PGD  ε={PGD_EPS*255:.0f}/255  "
              f"α={PGD_ALPHA*255:.0f}/255  steps={PGD_STEPS}")
        for start in range(0, imgs.size(0), BATCH_SIZE):
            end   = min(start + BATCH_SIZE, imgs.size(0))
            batch = imgs[start:end]
            blbl  = labels[start:end]
            adv   = pgd_batch(model, batch, blbl, PGD_EPS, PGD_ALPHA, PGD_STEPS)
            all_adv.append(adv.cpu())
            all_adv_lbl.append(blbl.cpu())
            all_cln.append(batch.cpu())
            all_cln_lbl.append(blbl.cpu())

    adv_imgs     = torch.cat(all_adv).to(DEVICE)
    adv_labels   = torch.cat(all_adv_lbl).to(DEVICE)
    clean_imgs   = torch.cat(all_cln).to(DEVICE)
    clean_labels = torch.cat(all_cln_lbl).to(DEVICE)

    n_adv = adv_imgs.size(0)
    print(f"\n  Generated {n_adv:,} adversarial samples  "
          f"({NUM_CLASSES} classes × {ADV_PER_CLASS} × "
          f"{len(FGSM_EPSILONS)} FGSM + 1 PGD = "
          f"{NUM_CLASSES * ADV_PER_CLASS * (len(FGSM_EPSILONS)+1):,} expected)")
    return adv_imgs, adv_labels, clean_imgs, clean_labels


# ─────────────────────────────────────────────────────────────────
# SIGNATURE LOADER  (GPU tensors for vectorized scoring)
# ─────────────────────────────────────────────────────────────────
def load_signature_to_gpu(path: str) -> dict:
    """
    Load mrc_signature_5cls.h5 into GPU tensors.

    Returns nested dict:
      sig[layer_name][class_idx] = {
          "mins": (N,)    FP32 GPU,
          "maxs": (N,)    FP32 GPU,
          "freq": (N, 32) FP32 GPU,
      }
    """
    print(f"\n[PHASE 4 / Step 1] Loading signature to GPU : {path}")
    sig = {}

    with h5py.File(path, "r") as f:
        layer_names = [n.decode() for n in f["metadata"]["layer_names"][:]]
        for lname in layer_names:
            sig[lname] = {}
            grp = f[f"layer_{lname}"]
            for cls in range(NUM_CLASSES):     # loop over 5 classes
                cgrp = grp[f"class_{cls}"]
                sig[lname][cls] = {
                    "mins": torch.tensor(cgrp["mins"][:],        device=DEVICE),
                    "maxs": torch.tensor(cgrp["maxs"][:],        device=DEVICE),
                    "freq": torch.tensor(cgrp["frequencies"][:], device=DEVICE),
                }
            n = sig[lname][0]["mins"].shape[0]
            print(f"  layer_{lname:8s} : {n:>10,} neurons loaded")

    print()
    return sig


# ─────────────────────────────────────────────────────────────────
# MRC SCORING  (vectorized — no Python loops over neurons)
# ─────────────────────────────────────────────────────────────────
def score_batch(model, hooks: dict, sig: dict,
                imgs: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Compute per-layer coverage cost η for a batch of inputs.

    η for a single input at layer L, predicted class C:
      1. Map each neuron activation to its bucket in sig[L][C]
      2. Look up the frequency of that bucket
      3. η_L = mean frequency across all neurons
         (high η = activation pattern matches trusted set = SAFE)
         (low  η = unusual buckets triggered  = SUSPICIOUS)

    Returns dict {layer_name: (B,) tensor of η scores on GPU}
    """
    with torch.no_grad():
        _ = model(imgs)    # triggers all three hooks

    eta = {}
    for lname, hook in hooks.items():
        act    = hook.activation.float()    # (B, N)
        B, N   = act.shape
        scores = torch.zeros(B, device=DEVICE)

        # Group samples by predicted class to vectorise lookup
        for cls in range(NUM_CLASSES):
            mask = (labels == cls)
            if mask.sum() == 0:
                continue

            a    = act[mask]                              # (B_cls, N)
            v_lo = sig[lname][cls]["mins"].unsqueeze(0)   # (1, N)
            v_hi = sig[lname][cls]["maxs"].unsqueeze(0)   # (1, N)
            freq = sig[lname][cls]["freq"]                # (N, 32)

            rng  = (v_hi - v_lo).clamp(min=1e-8)
            bidx = ((a - v_lo) / rng * Q_BUCKETS) \
                   .long().clamp(0, Q_BUCKETS - 1)        # (B_cls, N)

            # freq.gather looks up bucket frequency for each (neuron, bucket) pair
            looked_up     = freq.gather(1, bidx.t())      # (N, B_cls)
            scores[mask]  = looked_up.mean(dim=0)         # mean over neurons → scalar η per sample

        eta[lname] = scores

    return eta   # {layer: (B,) GPU tensor}


def score_all(model, hooks: dict, sig: dict,
              imgs: torch.Tensor, labels: torch.Tensor, desc: str = "scoring") -> dict:
    """Stream imgs through score_batch in batches, collect η per layer as numpy arrays."""
    all_eta = {lname: [] for lname in hooks}
    N = imgs.size(0)

    pbar = tqdm(range(0, N, BATCH_SIZE), desc=f"  {desc}", leave=True)
    for start in pbar:
        end   = min(start + BATCH_SIZE, N)
        batch = imgs[start:end]
        blbl  = labels[start:end]
        eta   = score_batch(model, hooks, sig, batch, blbl)
        for lname, scores in eta.items():
            all_eta[lname].append(scores.cpu())

    return {lname: torch.cat(v).numpy() for lname, v in all_eta.items()}


# ─────────────────────────────────────────────────────────────────
# ROC CALIBRATION
# ─────────────────────────────────────────────────────────────────
def calibrate_roc(clean_eta: dict, adv_eta: dict):
    """
    For each layer, compute ROC curve and find optimal τ via Youden's J.

    clean_eta : {layer: np.array (N_clean,)}  — high η expected
    adv_eta   : {layer: np.array (N_adv,)}    — low  η expected

    Convention: label 1 = SAFE (clean), label 0 = UNSAFE (adversarial).
    """
    print("\n[PHASE 4 / Step 4] ROC calibration...")

    results  = {}
    roc_data = {}

    for lname in clean_eta:
        c_scores = clean_eta[lname]
        a_scores = adv_eta[lname]

        scores = np.concatenate([c_scores, a_scores])
        labels = np.concatenate([np.ones(len(c_scores)),
                                 np.zeros(len(a_scores))])

        # sklearn roc_curve sweeps all threshold values in one call
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Youden's J = TPR - FPR, pick the threshold that maximises it
        J     = tpr - fpr
        best  = np.argmax(J)
        tau   = float(thresholds[best])
        t_tpr = float(tpr[best])
        t_fpr = float(fpr[best])

        # Additional operating point: highest TPR achievable at FPR ≤ 5%
        fp05_idx = np.where(fpr <= 0.05)[0]
        if len(fp05_idx):
            fp05_best = fp05_idx[np.argmax(tpr[fp05_idx])]
            tau_05    = float(thresholds[fp05_best])
            tpr_05    = float(tpr[fp05_best])
        else:
            tau_05, tpr_05 = tau, t_tpr

        results[lname] = {
            "tau_youden":   tau,
            "tpr_at_tau":   t_tpr,
            "fpr_at_tau":   t_fpr,
            "tau_fpr05":    tau_05,
            "tpr_at_fpr05": tpr_05,
            "auc":          roc_auc,
        }
        roc_data[lname] = (fpr, tpr, thresholds, roc_auc, tau)

        print(f"  {lname:8s} | AUC {roc_auc:.4f} | "
              f"τ(Youden)={tau:.5f}  TPR={t_tpr:.3f} FPR={t_fpr:.3f} | "
              f"τ(FPR≤5%)={tau_05:.5f}  TPR={tpr_05:.3f}")

    return results, roc_data


# ─────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────
LAYER_COLORS = {"shallow": "#378ADD", "middle": "#1D9E75", "deep": "#D85A30"}


def plot_roc(roc_data: dict, out_path: str):
    """Save ROC curves for all layers (and combined) to a PNG file."""
    fig, axes = plt.subplots(1, len(roc_data), figsize=(5 * len(roc_data), 4.5))
    if len(roc_data) == 1:
        axes = [axes]

    for ax, (lname, (fpr, tpr, _, roc_auc, tau)) in zip(axes, roc_data.items()):
        color = LAYER_COLORS.get(lname, "#888780")
        ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)    # diagonal random baseline

        J    = tpr - fpr
        best = np.argmax(J)
        ax.scatter(fpr[best], tpr[best], color=color, s=80, zorder=5,
                   label=f"τ = {tau:.4f}")   # mark optimal operating point

        ax.set_title(f"Layer: {lname}", fontsize=12)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)

    fig.suptitle("MRC Coverage Score — ROC Curves (clean vs adversarial) — 5 classes",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC plot saved → {out_path}")


def plot_score_dist(clean_eta: dict, adv_eta: dict, out_path: str):
    """Save η score distribution histograms comparing clean vs adversarial."""
    n_layers = len(clean_eta)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for ax, lname in zip(axes, clean_eta):
        color = LAYER_COLORS.get(lname, "#888780")
        c = clean_eta[lname]
        a = adv_eta[lname]

        bins = np.linspace(min(c.min(), a.min()),
                           max(c.max(), a.max()), 60)

        ax.hist(c, bins=bins, alpha=0.6, color=color,
                label=f"Clean  (n={len(c):,})", density=True)
        ax.hist(a, bins=bins, alpha=0.5, color="#E24B4A",
                label=f"Adv    (n={len(a):,})", density=True)

        ax.set_title(f"η distribution — {lname}", fontsize=11)
        ax.set_xlabel("Coverage score η  (higher = safer)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("MRC η Score Distributions — Clean vs Adversarial — 5 classes",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Score dist plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Phase 4  —  Threshold Calibration (ROC Analysis)  [5-class]")
    print("=" * 65)

    # 1. Load model + signature
    model = load_model(CHECKPOINT)
    sig   = load_signature_to_gpu(SIGNATURE_FILE)
    hooks = attach_hooks(model)

    # 2. Collect clean test samples per class (500 each)
    print("[PHASE 4 / Step 2a] Collecting clean test samples...")
    test_loader     = get_test_loader()
    clean_by_class  = collect_per_class(test_loader, model, ADV_PER_CLASS)
    total_clean     = sum(v.size(0) for v in clean_by_class.values())
    print(f"  Collected {total_clean:,} clean base samples "
          f"({ADV_PER_CLASS} per class × {NUM_CLASSES} classes)\n")

    # 3. Generate adversarial examples for all 5 classes
    adv_imgs, adv_labels, clean_imgs, clean_labels = \
        generate_adversarial(model, clean_by_class)

    # 4. Score clean + adversarial through MRC
    print("\n[PHASE 4 / Step 3] Scoring samples through MRC...")
    clean_eta = score_all(model, hooks, sig,
                          clean_imgs, clean_labels, desc="Clean  scoring")
    adv_eta   = score_all(model, hooks, sig,
                          adv_imgs,   adv_labels,   desc="Adv    scoring")

    for lname in clean_eta:
        print(f"  {lname:8s} | clean η  mean={clean_eta[lname].mean():.5f} "
              f"std={clean_eta[lname].std():.5f}")
        print(f"  {lname:8s} | adv   η  mean={adv_eta[lname].mean():.5f}  "
              f"std={adv_eta[lname].std():.5f}")

    # 5. ROC calibration per layer
    results, roc_data = calibrate_roc(clean_eta, adv_eta)

    # 6. Combined score (mean η across all 3 layers)
    clean_combined = np.mean(
        np.stack([clean_eta[l] for l in clean_eta]), axis=0)
    adv_combined   = np.mean(
        np.stack([adv_eta[l]   for l in adv_eta]),   axis=0)

    scores   = np.concatenate([clean_combined, adv_combined])
    lbls     = np.concatenate([np.ones(len(clean_combined)),
                               np.zeros(len(adv_combined))])
    fpr_c, tpr_c, thr_c = roc_curve(lbls, scores, pos_label=1)
    auc_c   = auc(fpr_c, tpr_c)
    J_c     = tpr_c - fpr_c
    best_c  = np.argmax(J_c)
    tau_c   = float(thr_c[best_c])

    results["combined"] = {
        "tau_youden": tau_c,
        "tpr_at_tau": float(tpr_c[best_c]),
        "fpr_at_tau": float(fpr_c[best_c]),
        "auc":        auc_c,
        "note":       "mean η across all layers",
    }
    roc_data["combined"] = (fpr_c, tpr_c, thr_c, auc_c, tau_c)

    print(f"\n  combined  | AUC {auc_c:.4f} | "
          f"τ(Youden)={tau_c:.5f}  "
          f"TPR={float(tpr_c[best_c]):.3f}  "
          f"FPR={float(fpr_c[best_c]):.3f}")

    # 7. Save thresholds JSON for Phase 5
    print(f"\n[PHASE 4 / Step 5] Saving thresholds → {OUT_JSON}")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved. Keys: {list(results.keys())}")

    # 8. Generate and save plots
    print(f"\n[PHASE 4 / Step 6] Generating plots...")
    plot_roc(roc_data, OUT_ROC_PNG)
    plot_score_dist(clean_eta, adv_eta, OUT_DIST_PNG)

    # 9. Clean up hooks
    for h in hooks.values():
        h.remove()
    print("\n[INFO] Hooks removed.")

    # 10. Final summary
    print("\n" + "=" * 65)
    print("  Phase 4 (5-class) complete.\n")
    print("  Outputs:")
    print(f"    {OUT_JSON}    — τ thresholds for Phase 5")
    print(f"    {OUT_ROC_PNG}     — ROC curves")
    print(f"    {OUT_DIST_PNG} — η score distributions")
    print("\n  Recommended τ (Youden's J):")
    for lname, r in results.items():
        print(f"    {lname:10s} → τ = {r['tau_youden']:.5f}  "
              f"(AUC={r['auc']:.4f})")
    print("\n  Next → Phase 5: live monitoring wrapper using these τ values")
    print("=" * 65)


if __name__ == "__main__":
    main()
