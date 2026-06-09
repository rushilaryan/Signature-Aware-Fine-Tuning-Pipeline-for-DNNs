"""
phase6_random_demo.py  [5-Class Subset of CIFAR-10]
====================================================
Phase 6 : Random Image Demo — Softmax + Confidence Visualisation

For each randomly selected test image this script:
  1. Picks a random image from the 5-class CIFAR-10 test set
  2. Runs it through the LiveMonitor (inference + MRC scoring)
  3. Displays a 4-panel figure:
       Panel A  — the raw image (32×32, upscaled for display)
       Panel B  — softmax probability bar chart (all 5 classes)
       Panel C  — per-layer η (MRC coverage) scores
       Panel D  — combined confidence gauge with ACCEPT / REJECT verdict
  4. Saves each figure as  demo_sample_<idx>.png

Can also run an adversarial pair: shows the same image clean vs FGSM-attacked
side by side so you can see exactly how scores change under attack.

Prerequisites:
  - vgg19_phase1_best_5cls.pth     (Phase 1)
  - mrc_signature_5cls.h5          (Phase 2/3)
  - phase4_thresholds_5cls.json    (Phase 4)
  - pip install torch torchvision h5py matplotlib tqdm
"""

import random
import os

import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT      = "vgg19_phase1_best_5cls.pth"
SIGNATURE_FILE  = "mrc_signature_5cls.h5"
THRESHOLDS_FILE = "phase4_thresholds_5cls.json"

NUM_CLASSES      = 5
REJECT_THRESHOLD = 0.5    # combined η below this → REJECT
Q_BUCKETS        = 32

# Maps original CIFAR-10 label → remapped label (0..4)
SELECTED_CLASSES = {
    0: 0,   # airplane   → 0
    1: 1,   # automobile → 1
    2: 2,   # bird       → 2
    3: 3,   # cat        → 3
    4: 4,   # deer       → 4
}

# Reverse map: remapped label → class name (for display)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer"]

# Hook attachment points — same as all prior phases
HOOK_TARGETS = {
    "shallow": ("features",    3),
    "middle":  ("features",   17),
    "deep":    ("classifier",  3),
}

# Visual colour palette
COLORS = {
    "accept":    "#2ECC71",   # green for accepted inputs
    "reject":    "#E74C3C",   # red  for rejected inputs
    "bar_clean": "#3498DB",   # blue for softmax bars
    "bar_adv":   "#E74C3C",   # red  for adversarial softmax bars
    "eta":       "#9B59B6",   # purple for η score bars
    "true_cls":  "#F39C12",   # orange to highlight the true class
}

N_RANDOM_SAMPLES = 5    # how many random images to demo by default
SAVE_DIR         = "."  # directory where demo PNG files are saved


# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────
class FilteredCIFAR10(Dataset):
    """
    CIFAR-10 subset keeping only SELECTED_CLASSES with remapped labels 0..4.
    Also stores the original raw uint8 image so it can be displayed.
    """

    def __init__(self, root: str = "./data", train: bool = False,
                 transform=None):
        base = datasets.CIFAR10(root=root, train=train,
                                download=True, transform=None)
        self.transform  = transform
        self.raw_images = []   # original 32×32 uint8 numpy arrays for display
        self.data       = []   # same arrays stored for getitem access
        self.labels     = []   # remapped labels (0..4)

        for img_np, lbl in zip(base.data, base.targets):
            if lbl in SELECTED_CLASSES:
                self.raw_images.append(img_np)        # keep for display
                self.data.append(img_np)
                self.labels.append(SELECTED_CLASSES[lbl])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return transformed tensor and remapped label for model inference
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def get_raw(self, idx: int) -> np.ndarray:
        """Return the original 32×32 uint8 numpy array for display."""
        return self.raw_images[idx]


def build_dataset(root: str = "./data") -> FilteredCIFAR10:
    """Build the filtered test dataset with ImageNet normalisation."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                          # VGG19 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),        # ImageNet channel stats
    ])
    return FilteredCIFAR10(root=root, train=False, transform=transform)


# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────
class LightweightVGG19(nn.Module):
    """Same architecture as all prior phases — must match checkpoint."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        base = models.vgg19(weights=None)    # weights come from checkpoint
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),           # classifier[3] — deep hook
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


# ─────────────────────────────────────────────────────────────────
# HOOK
# ─────────────────────────────────────────────────────────────────
class CVLayerHook:
    """Passive forward hook — captures layer activations without affecting the pass."""

    def __init__(self, name: str):
        self.name       = name
        self.activation = None
        self._handle    = None

    def hook_fn(self, module, input, output):
        # Flatten spatial dims so both conv (B,C,H,W) and FC (B,N) become (B,N)
        self.activation = output.detach().flatten(start_dim=1)

    def attach(self, module: nn.Module):
        self._handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


# ─────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────
@dataclass
class SampleResult:
    """All scores and decisions for one input image."""
    idx:             int                # index in the dataset
    true_label:      int                # ground-truth remapped label
    true_class:      str                # ground-truth class name
    predicted_label: int                # model predicted remapped label
    predicted_class: str                # model predicted class name
    softmax_scores:  np.ndarray         # (5,) softmax probability for each class
    eta_per_layer:   Dict[str, float]   # per-layer η scores
    confidence:      float              # mean η across layers
    accepted:        bool               # True if confidence ≥ REJECT_THRESHOLD
    correct:         bool               # True if prediction matches ground truth
    raw_image:       np.ndarray         # 32×32 uint8 for display


# ─────────────────────────────────────────────────────────────────
# MONITOR
# ─────────────────────────────────────────────────────────────────
class RandomDemoMonitor:
    """
    Lightweight monitor that runs inference + MRC scoring on single images.
    Built from the same three output files as the Phase 5 LiveMonitor.
    """

    def __init__(self):
        self.model      = self._load_model()
        self.hooks      = self._attach_hooks()
        self.sig        = self._load_signature()
        self.thresholds = self._load_thresholds()
        print(f"[INFO] RandomDemoMonitor ready on {DEVICE}\n")

    # ── Loaders ────────────────────────────────────────────────

    def _load_model(self) -> LightweightVGG19:
        """Load the Phase 1 checkpoint and freeze everything."""
        print(f"  Loading model  : {CHECKPOINT}")
        ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
        model = LightweightVGG19()
        model.load_state_dict(ckpt["model_state_dict"])   # restore trained weights

        for p in model.parameters():
            p.requires_grad = False    # fully frozen — no gradient tracking needed

        # Dropout → Identity so activations are deterministic (no stochastic masking)
        model.classifier = nn.Sequential(*[
            nn.Identity() if isinstance(l, nn.Dropout) else l
            for l in model.classifier.children()
        ])

        model.to(DEVICE).eval()
        print(f"  Val accuracy   : {ckpt.get('val_acc', 0) * 100:.2f}%")
        return model

    def _attach_hooks(self) -> dict:
        """Attach passive hooks at 3 layer depths."""
        hooks = {}
        for name, (attr, idx) in HOOK_TARGETS.items():
            h = CVLayerHook(name)
            h.attach(getattr(self.model, attr)[idx])   # e.g. model.features[3]
            hooks[name] = h
        print(f"  Hooks attached : {list(hooks.keys())}")
        return hooks

    def _load_signature(self) -> dict:
        """Load per-class per-layer MRC signature into GPU tensors."""
        print(f"  Loading sig    : {SIGNATURE_FILE}")
        sig = {}
        with h5py.File(SIGNATURE_FILE, "r") as f:
            layer_names = [n.decode() for n in f["metadata"]["layer_names"][:]]
            for lname in layer_names:
                sig[lname] = {}
                grp = f[f"layer_{lname}"]
                for c in range(NUM_CLASSES):
                    cgrp = grp[f"class_{c}"]
                    sig[lname][c] = {
                        "mins": torch.tensor(cgrp["mins"][:],        device=DEVICE),
                        "maxs": torch.tensor(cgrp["maxs"][:],        device=DEVICE),
                        "freq": torch.tensor(cgrp["frequencies"][:], device=DEVICE),
                    }
        return sig

    def _load_thresholds(self) -> dict:
        """Load Phase 4 calibrated τ thresholds from JSON."""
        print(f"  Loading τ      : {THRESHOLDS_FILE}")
        with open(THRESHOLDS_FILE) as f:
            return json.load(f)

    # ── Core inference ─────────────────────────────────────────

    @torch.no_grad()
    def score_image(self, img_tensor: torch.Tensor,
                    true_label: int,
                    raw_image: np.ndarray,
                    idx: int) -> SampleResult:
        """
        Run one image through the model + MRC monitor and return a SampleResult.

        img_tensor : (3, 224, 224) normalised tensor (CPU or GPU)
        true_label : ground-truth remapped label (0..4)
        raw_image  : 32×32 uint8 numpy array for display
        idx        : dataset index (for labelling saved files)
        """
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)   # add batch dim → (1, 3, 224, 224)

        logits = self.model(img_tensor)                    # forward pass triggers hooks
        probs  = F.softmax(logits, dim=1)                  # (1, 5) probability distribution

        softmax_scores  = probs.squeeze(0).cpu().numpy()   # (5,) array of class probs
        predicted_label = int(probs.argmax(dim=1).item())  # index of highest probability

        # Compute per-layer η scores using the predicted class signature
        eta_per_layer = {}
        for lname, hook in self.hooks.items():
            act  = hook.activation.float()    # (1, N)
            v_lo = self.sig[lname][predicted_label]["mins"].unsqueeze(0)   # (1, N)
            v_hi = self.sig[lname][predicted_label]["maxs"].unsqueeze(0)   # (1, N)
            freq = self.sig[lname][predicted_label]["freq"]                # (N, 32)

            rng  = (v_hi - v_lo).clamp(min=1e-8)
            bidx = ((act - v_lo) / rng * Q_BUCKETS).long().clamp(0, Q_BUCKETS - 1)  # (1, N)

            # Look up how frequently the activated bucket is seen in trusted samples
            looked_up = freq.gather(1, bidx.t())     # (N, 1)
            eta_per_layer[lname] = float(looked_up.mean().item())

        # Combined confidence = mean η across all 3 layer scores
        confidence = float(np.mean(list(eta_per_layer.values())))

        return SampleResult(
            idx             = idx,
            true_label      = true_label,
            true_class      = CLASS_NAMES[true_label],
            predicted_label = predicted_label,
            predicted_class = CLASS_NAMES[predicted_label],
            softmax_scores  = softmax_scores,
            eta_per_layer   = eta_per_layer,
            confidence      = confidence,
            accepted        = (confidence >= REJECT_THRESHOLD),
            correct         = (predicted_label == true_label),
            raw_image       = raw_image,
        )

    def cleanup(self):
        """Remove forward hooks to free resources."""
        for h in self.hooks.values():
            h.remove()


# ─────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────
def _verdict_color(accepted: bool) -> str:
    """Return green for ACCEPT, red for REJECT."""
    return COLORS["accept"] if accepted else COLORS["reject"]


def plot_sample(result: SampleResult, save_path: str):
    """
    Draw a 4-panel figure for one sample:

      [A] Raw image
      [B] Softmax probability bar chart (all 5 classes)
      [C] Per-layer η bar chart
      [D] Combined confidence gauge with verdict box
    """
    fig = plt.figure(figsize=(16, 5), facecolor="#1C1C2E")    # dark background
    gs  = gridspec.GridSpec(1, 4, figure=fig,
                            left=0.04, right=0.97,
                            bottom=0.15, top=0.82,
                            wspace=0.38)

    verdict_color = _verdict_color(result.accepted)
    verdict_text  = "✔  ACCEPT" if result.accepted else "✘  REJECT"
    correct_text  = "✓ correct" if result.correct else "✗ wrong"

    # ── Figure title ───────────────────────────────────────────
    title_line = (f"Sample #{result.idx}  |  True: {result.true_class}  |  "
                  f"Pred: {result.predicted_class}  ({correct_text})  |  "
                  f"Decision: {verdict_text}  |  Confidence c = {result.confidence:.4f}")
    fig.suptitle(title_line, fontsize=11, color="white", fontweight="bold", y=0.97)

    # ── Panel A — Raw image ────────────────────────────────────
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(result.raw_image)                            # display 32×32 CIFAR image
    ax_img.set_title("Input image", color="white", fontsize=10)
    ax_img.axis("off")

    # Draw a coloured border around the image matching the verdict colour
    for spine in ax_img.spines.values():
        spine.set_edgecolor(verdict_color)
        spine.set_linewidth(3)

    # Label below image: true class and true label index
    ax_img.set_xlabel(
        f"True: {result.true_class} (label {result.true_label})",
        color="#AAAAAA", fontsize=9
    )

    # ── Panel B — Softmax scores ───────────────────────────────
    ax_soft = fig.add_subplot(gs[1])
    ax_soft.set_facecolor("#2A2A3E")

    x_pos   = np.arange(NUM_CLASSES)
    bar_colors = [
        COLORS["true_cls"] if i == result.true_label        # gold for true class
        else verdict_color if i == result.predicted_label   # verdict colour for prediction
        else "#555577"                                       # grey for other classes
        for i in range(NUM_CLASSES)
    ]

    bars = ax_soft.bar(x_pos, result.softmax_scores * 100,
                       color=bar_colors, edgecolor="#888888", linewidth=0.5,
                       zorder=3)

    # Annotate each bar with its exact percentage
    for bar, score in zip(bars, result.softmax_scores):
        ax_soft.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{score*100:.1f}%",
            ha="center", va="bottom",
            color="white", fontsize=8, fontweight="bold"
        )

    ax_soft.set_xticks(x_pos)
    ax_soft.set_xticklabels(CLASS_NAMES, rotation=20, ha="right",
                             color="white", fontsize=9)
    ax_soft.set_ylabel("Softmax probability (%)", color="white", fontsize=9)
    ax_soft.set_title("Softmax scores (all 5 classes)", color="white", fontsize=10)
    ax_soft.set_ylim(0, 115)
    ax_soft.tick_params(colors="white")
    ax_soft.yaxis.label.set_color("white")
    ax_soft.spines["bottom"].set_color("#555566")
    ax_soft.spines["left"].set_color("#555566")
    ax_soft.spines["top"].set_visible(False)
    ax_soft.spines["right"].set_visible(False)
    ax_soft.grid(axis="y", color="#444455", linewidth=0.5, zorder=0)

    # Legend for bar colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["true_cls"],  label="True class"),
        Patch(facecolor=verdict_color,        label="Predicted class"),
    ]
    ax_soft.legend(handles=legend_elements, loc="upper right",
                   fontsize=8, framealpha=0.3,
                   labelcolor="white", facecolor="#2A2A3E")

    # ── Panel C — Per-layer η scores ───────────────────────────
    ax_eta = fig.add_subplot(gs[2])
    ax_eta.set_facecolor("#2A2A3E")

    layer_names  = list(result.eta_per_layer.keys())
    eta_values   = [result.eta_per_layer[l] for l in layer_names]
    eta_colors   = ["#378ADD", "#1D9E75", "#D85A30"]   # matches phase4 plot colours

    eta_bars = ax_eta.bar(layer_names, eta_values,
                           color=eta_colors, edgecolor="#888888",
                           linewidth=0.5, zorder=3)

    # Annotate bars with exact η value
    for bar, val in zip(eta_bars, eta_values):
        ax_eta.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center", va="bottom",
            color="white", fontsize=9, fontweight="bold"
        )

    # Horizontal line showing the reject threshold (0.5)
    ax_eta.axhline(y=REJECT_THRESHOLD, color="#FFCC00", linewidth=1.5,
                   linestyle="--", zorder=4, label=f"τ = {REJECT_THRESHOLD}")
    ax_eta.legend(loc="upper right", fontsize=8, framealpha=0.3,
                  labelcolor="white", facecolor="#2A2A3E")

    ax_eta.set_ylim(0, max(max(eta_values) * 1.3, REJECT_THRESHOLD * 1.5))
    ax_eta.set_xticklabels(layer_names, color="white", fontsize=9)
    ax_eta.set_ylabel("MRC coverage score η", color="white", fontsize=9)
    ax_eta.set_title("Per-layer η scores", color="white", fontsize=10)
    ax_eta.tick_params(colors="white")
    ax_eta.spines["bottom"].set_color("#555566")
    ax_eta.spines["left"].set_color("#555566")
    ax_eta.spines["top"].set_visible(False)
    ax_eta.spines["right"].set_visible(False)
    ax_eta.grid(axis="y", color="#444455", linewidth=0.5, zorder=0)

    # ── Panel D — Confidence gauge ─────────────────────────────
    ax_gauge = fig.add_subplot(gs[3])
    ax_gauge.set_facecolor("#2A2A3E")
    ax_gauge.set_xlim(0, 1)
    ax_gauge.set_ylim(0, 1)
    ax_gauge.axis("off")

    # Background track bar (full width, grey)
    ax_gauge.barh(0.55, 1.0, height=0.12, left=0.0,
                  color="#444455", zorder=2)

    # Foreground fill bar (coloured up to the confidence value)
    ax_gauge.barh(0.55, result.confidence, height=0.12, left=0.0,
                  color=verdict_color, zorder=3)

    # Threshold tick mark at 0.5
    ax_gauge.axvline(x=REJECT_THRESHOLD, ymin=0.44, ymax=0.68,
                     color="#FFCC00", linewidth=2, zorder=4)
    ax_gauge.text(REJECT_THRESHOLD, 0.70, f"τ={REJECT_THRESHOLD}",
                  ha="center", va="bottom", color="#FFCC00", fontsize=8)

    # Confidence value label above the bar
    ax_gauge.text(result.confidence, 0.69,
                  f"c = {result.confidence:.4f}",
                  ha="center", va="bottom",
                  color="white", fontsize=11, fontweight="bold")

    # Pointer triangle at the confidence position
    ax_gauge.scatter([result.confidence], [0.49],
                     marker="v", s=120, color="white", zorder=5)

    # Large verdict box
    verdict_bbox = FancyBboxPatch(
        (0.05, 0.12), 0.90, 0.27,
        boxstyle="round,pad=0.04",
        facecolor=verdict_color, edgecolor="white",
        linewidth=2, zorder=3
    )
    ax_gauge.add_patch(verdict_bbox)
    ax_gauge.text(0.50, 0.255, verdict_text,
                  ha="center", va="center",
                  color="white", fontsize=15, fontweight="bold", zorder=4)

    ax_gauge.set_title("Combined confidence", color="white", fontsize=10)

    # Axis labels (0 and 1) at ends of gauge
    ax_gauge.text(0.0,  0.44, "0", ha="center", color="#AAAAAA", fontsize=9)
    ax_gauge.text(1.0,  0.44, "1", ha="center", color="#AAAAAA", fontsize=9)

    # ── Save ───────────────────────────────────────────────────
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# ADVERSARIAL PAIR VISUALISATION
# ─────────────────────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def fgsm_attack(model: nn.Module, img_tensor: torch.Tensor,
                true_label: int, eps: float = 8/255) -> torch.Tensor:
    """
    Apply FGSM perturbation to a single image tensor.

    img_tensor : (3, 224, 224) normalised tensor (CPU)
    Returns adversarial normalised tensor on CPU.
    """
    img = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)   # (1,3,224,224)
    lbl = torch.tensor([true_label], device=DEVICE)

    # Need gradients enabled for FGSM so temporarily enable on the model too
    original_grad_states = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(True)

    loss = F.cross_entropy(model(img), lbl)   # compute loss to get gradient direction
    loss.backward()

    for p, state in original_grad_states.items():
        p.requires_grad_(state)               # restore original frozen state

    with torch.no_grad():
        x_px  = img * _STD + _MEAN
        delta = eps * img.grad.sign()         # gradient sign perturbation
        x_adv = (x_px + delta).clamp(0, 1)   # stay in valid pixel range
        x_adv = (x_adv - _MEAN) / _STD       # back to normalised space

    return x_adv.squeeze(0).detach().cpu()    # return (3,224,224) on CPU


def plot_adversarial_pair(clean_result: SampleResult,
                          adv_result: SampleResult,
                          eps: float,
                          save_path: str):
    """
    Side-by-side comparison of clean vs adversarial image showing how
    softmax scores and η confidence change under FGSM attack.

    Left  3 panels : clean image, softmax, confidence gauge
    Right 3 panels : adversarial image, softmax, confidence gauge
    """
    fig = plt.figure(figsize=(20, 6), facecolor="#1C1C2E")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.04, right=0.97,
                            bottom=0.10, top=0.80,
                            wspace=0.35, hspace=0.50)

    fig.suptitle(
        f"Clean vs Adversarial (FGSM ε={eps*255:.0f}/255)  |  "
        f"Sample #{clean_result.idx}  True class: {clean_result.true_class}",
        fontsize=12, color="white", fontweight="bold", y=0.96
    )

    for col, (result, label) in enumerate(
            [(clean_result, "CLEAN"), (adv_result, f"ADVERSARIAL (ε={eps*255:.0f}/255)")]):

        col_offset = col * 3   # column 0 or 3 in the grid
        v_color    = _verdict_color(result.accepted)
        verdict    = "✔  ACCEPT" if result.accepted else "✘  REJECT"

        # ── Sub-panel 1: image ──────────────────────────────
        ax_img = fig.add_subplot(gs[0, col_offset])
        ax_img.imshow(result.raw_image)
        ax_img.set_title(f"{label}\n{result.predicted_class} "
                         f"(p={result.softmax_scores[result.predicted_label]*100:.1f}%)",
                         color="white", fontsize=9)
        ax_img.axis("off")

        # ── Sub-panel 2: softmax bars ───────────────────────
        ax_soft = fig.add_subplot(gs[0, col_offset + 1])
        ax_soft.set_facecolor("#2A2A3E")
        x_pos = np.arange(NUM_CLASSES)

        bar_colors = [
            COLORS["true_cls"] if i == result.true_label
            else v_color        if i == result.predicted_label
            else "#555577"
            for i in range(NUM_CLASSES)
        ]
        bars = ax_soft.bar(x_pos, result.softmax_scores * 100,
                           color=bar_colors, edgecolor="#888888",
                           linewidth=0.5, zorder=3)

        for bar, score in zip(bars, result.softmax_scores):
            if score * 100 > 3:    # only label bars wide enough to read
                ax_soft.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{score*100:.1f}%",
                    ha="center", va="bottom",
                    color="white", fontsize=7.5, fontweight="bold"
                )

        ax_soft.set_xticks(x_pos)
        ax_soft.set_xticklabels(CLASS_NAMES, rotation=20, ha="right",
                                 color="white", fontsize=8)
        ax_soft.set_ylim(0, 120)
        ax_soft.set_ylabel("Softmax (%)", color="white", fontsize=8)
        ax_soft.tick_params(colors="white")
        ax_soft.spines["top"].set_visible(False)
        ax_soft.spines["right"].set_visible(False)
        ax_soft.spines["bottom"].set_color("#555566")
        ax_soft.spines["left"].set_color("#555566")
        ax_soft.grid(axis="y", color="#444455", linewidth=0.5, zorder=0)

        # ── Sub-panel 3: confidence + verdict ──────────────
        ax_gauge = fig.add_subplot(gs[0, col_offset + 2])
        ax_gauge.set_facecolor("#2A2A3E")
        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_ylim(0, 1)
        ax_gauge.axis("off")

        # Background and foreground gauge bars
        ax_gauge.barh(0.65, 1.0, height=0.15, left=0.0,
                      color="#444455", zorder=2)
        ax_gauge.barh(0.65, result.confidence, height=0.15, left=0.0,
                      color=v_color, zorder=3)

        ax_gauge.axvline(x=REJECT_THRESHOLD, ymin=0.53, ymax=0.83,
                         color="#FFCC00", linewidth=2, zorder=4)

        ax_gauge.text(result.confidence, 0.83,
                      f"c = {result.confidence:.4f}",
                      ha="center", va="bottom",
                      color="white", fontsize=10, fontweight="bold")

        verdict_bbox = FancyBboxPatch(
            (0.05, 0.15), 0.90, 0.32,
            boxstyle="round,pad=0.04",
            facecolor=v_color, edgecolor="white",
            linewidth=2, zorder=3
        )
        ax_gauge.add_patch(verdict_bbox)
        ax_gauge.text(0.50, 0.31, verdict,
                      ha="center", va="center",
                      color="white", fontsize=12,
                      fontweight="bold", zorder=4)

        # Per-layer η values as text below the gauge
        eta_lines = [f"η_{lname}: {score:.4f}"
                     for lname, score in result.eta_per_layer.items()]
        ax_gauge.text(0.5, 0.08, "   ".join(eta_lines),
                      ha="center", va="center",
                      color="#AAAAAA", fontsize=7.5)

        ax_gauge.set_title("Confidence gauge", color="white", fontsize=9)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# CONSOLE PRINT HELPER
# ─────────────────────────────────────────────────────────────────
def print_result(result: SampleResult):
    """Print a compact table of all scores to the console."""
    verdict_str = "✔ ACCEPT" if result.accepted else "✘ REJECT"
    correct_str = "✓ correct prediction" if result.correct else "✗ wrong prediction"

    print(f"\n{'─'*58}")
    print(f"  Sample #{result.idx:>5d}  |  {verdict_str}")
    print(f"{'─'*58}")
    print(f"  True class        : {result.true_class} (label {result.true_label})")
    print(f"  Predicted class   : {result.predicted_class} (label {result.predicted_label})"
          f"  [{correct_str}]")
    print()
    print("  Softmax scores (all 5 classes):")
    for i, (name, score) in enumerate(zip(CLASS_NAMES, result.softmax_scores)):
        bar    = "█" * int(score * 40)          # ASCII progress bar
        marker = " ← predicted" if i == result.predicted_label else \
                 " ← TRUE     " if i == result.true_label else ""
        print(f"    {name:12s} : {score*100:6.2f}%  {bar}{marker}")

    print()
    print("  MRC per-layer η scores:")
    for lname, score in result.eta_per_layer.items():
        bar    = "█" * int(score * 40)
        status = " ✔" if score >= REJECT_THRESHOLD else " ✘"
        print(f"    {lname:8s} : {score:.6f}  {bar}{status}")

    print()
    print(f"  Combined confidence  c = {result.confidence:.6f}  "
          f"(threshold τ = {REJECT_THRESHOLD})")
    print(f"  Decision             : {verdict_str}")
    print(f"{'─'*58}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Phase 6 — Random Image Demo")
    print("  Softmax + MRC Confidence Visualisation  [5-class]")
    print("=" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  Classes : {CLASS_NAMES}\n")

    # Build dataset and monitor
    dataset = build_dataset()
    monitor = RandomDemoMonitor()

    print(f"\n[INFO] Dataset : {len(dataset):,} test samples "
          f"(5-class filtered CIFAR-10)\n")

    # ── Part 1: N random single-image demos ────────────────────
    print(f"[PHASE 6 / Part 1] Scoring {N_RANDOM_SAMPLES} random images...\n")

    # Pick N random indices without repetition
    random_indices = random.sample(range(len(dataset)), N_RANDOM_SAMPLES)

    for sample_num, idx in enumerate(random_indices, start=1):
        img_tensor, true_label = dataset[idx]     # transformed tensor + remapped label
        raw_image = dataset.get_raw(idx)           # original 32×32 for display

        result = monitor.score_image(img_tensor, true_label, raw_image, idx)

        print_result(result)

        save_path = os.path.join(SAVE_DIR, f"demo_sample_{sample_num:02d}.png")
        plot_sample(result, save_path)

    # ── Part 2: adversarial pair for one random image ──────────
    print(f"\n[PHASE 6 / Part 2] Adversarial pair demo...")

    adv_idx       = random.choice(range(len(dataset)))
    img_tensor, true_label = dataset[adv_idx]
    raw_image     = dataset.get_raw(adv_idx)

    # Score the clean image
    clean_result  = monitor.score_image(img_tensor, true_label, raw_image, adv_idx)

    # Generate FGSM adversarial version and score it
    # Re-enable gradients on model temporarily for FGSM
    adv_tensor    = fgsm_attack(monitor.model, img_tensor, true_label, eps=8/255)
    adv_result    = monitor.score_image(adv_tensor, true_label, raw_image, adv_idx)

    print("\n  Clean image:")
    print_result(clean_result)
    print("\n  Adversarial image (FGSM ε=8/255):")
    print_result(adv_result)

    adv_save_path = os.path.join(SAVE_DIR, "demo_adv_pair.png")
    plot_adversarial_pair(clean_result, adv_result, eps=8/255, save_path=adv_save_path)

    # Clean up hooks
    monitor.cleanup()

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 6 complete.")
    print(f"\n  Files saved:")
    for i in range(1, N_RANDOM_SAMPLES + 1):
        print(f"    demo_sample_{i:02d}.png")
    print(f"    demo_adv_pair.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
