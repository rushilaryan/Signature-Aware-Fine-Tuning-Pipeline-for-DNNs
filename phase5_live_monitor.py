"""
phase5_live_monitor.py  [5-Class Subset of CIFAR-10]
=====================================================
Phase 5 : Live Deployment — Online Monitoring (Algorithm 4)

For every new input x_new the monitor performs:
  1. Inference     → model predicts class ŷ  (one of 5 classes)
  2. Active State  → CV-Layers record which MRC buckets each neuron hits
  3. Cost η        → mean frequency of the triggered buckets (per layer)
  4. Confidence c  → combined η across layers
  5. Decision      → ACCEPT if c ≥ 0.5, REJECT otherwise

5 classes selected from CIFAR-10 (original label → remapped label):
  airplane    (0) → 0
  automobile  (1) → 1
  bird        (2) → 2
  cat         (3) → 3
  deer        (4) → 4

Outputs per input:
  - predicted class ŷ  (0..4)
  - softmax probability
  - per-layer η scores
  - combined confidence c
  - ACCEPT / REJECT decision

Prerequisites:
  - vgg19_phase1_best_5cls.pth     (Phase 1)
  - mrc_signature_5cls.h5          (Phase 2/3)
  - phase4_thresholds_5cls.json    (Phase 4)
  - pip install torch torchvision h5py tqdm
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image               # needed to convert numpy arrays for transforms
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT      = "vgg19_phase1_best_5cls.pth"
SIGNATURE_FILE  = "mrc_signature_5cls.h5"
THRESHOLDS_FILE = "phase4_thresholds_5cls.json"

NUM_CLASSES     = 5      # 5-class subset

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
Q_BUCKETS   = 32    # must match Phase 2/3
BATCH_SIZE  = 64

REJECT_THRESHOLD    = 0.5   # inputs with combined η below this are flagged as suspicious
SOFT_CONF_THRESHOLD = 0.70  # softmax confidence below this triggers MRC cross-class re-check

HOOK_TARGETS = {
    "shallow": ("features",    3),   # ReLU after conv1_2
    "middle":  ("features",   17),   # ReLU after conv3_4
    "deep":    ("classifier",  3),   # ReLU after FC1
}


# ─────────────────────────────────────────────────────────────────
# DATASET  (5-class filtered CIFAR-10 — used for demo evaluation)
# ─────────────────────────────────────────────────────────────────
class FilteredCIFAR10(Dataset):
    """
    Wraps CIFAR-10, keeping only SELECTED_CLASSES with remapped labels 0..4.
    Used for demo and evaluation in this phase.
    """

    def __init__(self, root: str = "./data", train: bool = False,
                 transform=None):
        base = datasets.CIFAR10(root=root, train=train,
                                download=True, transform=None)
        self.transform = transform
        self.data   = []
        self.labels = []

        for img_np, lbl in zip(base.data, base.targets):
            if lbl in SELECTED_CLASSES:
                self.data.append(img_np)
                self.labels.append(SELECTED_CLASSES[lbl])   # remap to 0..4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])   # numpy HWC → PIL for transforms
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
        base = models.vgg19(weights=None)   # weights come from checkpoint
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),           # classifier[3] — deep hook attaches here
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
# HOOKS
# ─────────────────────────────────────────────────────────────────
class CVLayerHook:
    """Passive forward hook — captures activations without changing the forward pass."""

    def __init__(self, name: str):
        self.name       = name
        self.activation = None
        self._handle    = None

    def hook_fn(self, module, input, output):
        # Flatten spatial dims so both conv and FC layers have shape (B, N)
        self.activation = output.detach().flatten(start_dim=1)

    def attach(self, module: nn.Module):
        self._handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


# ─────────────────────────────────────────────────────────────────
# MONITORING RESULT
# ─────────────────────────────────────────────────────────────────
@dataclass
class MonitorResult:
    """One result per input image — includes class, confidence, and ACCEPT/REJECT."""
    predicted_class:    int
    class_name:         str
    softmax_prob:       float
    eta_per_layer:      Dict[str, float]   # η scores for the predicted class only
    confidence:         float              # mean η across layers for predicted class

    # ── MRC cross-class fields (populated only when softmax_prob < SOFT_CONF_THRESHOLD) ──
    low_confidence:     bool               # True when softmax confidence was below threshold
    mrc_best_class:     int                # class whose signature best fits the activations
    mrc_best_class_name: str               # name of mrc_best_class
    mrc_class_etas:     Dict[int, float]   # combined η for every class (cross-class scan)

    accepted:           bool               # final ACCEPT / REJECT decision

    def __repr__(self):
        status = "ACCEPT" if self.accepted else "REJECT"
        layers = "  ".join(f"{k}={v:.4f}" for k, v in self.eta_per_layer.items())
        base = (
            f"[{status}] class={self.class_name} "
            f"(p={self.softmax_prob:.3f})  "
            f"η: {layers}  c={self.confidence:.4f}"
        )
        # Show MRC re-check result when low confidence triggered it
        if self.low_confidence:
            match = "✔ agree" if self.mrc_best_class == self.predicted_class else "✘ conflict"
            base += (f"  [LOW-CONF → MRC best={self.mrc_best_class_name} "
                     f"η={self.mrc_class_etas.get(self.mrc_best_class, 0):.4f} {match}]")
        return base


@dataclass
class MonitorStats:
    """Running statistics for a batch evaluation session."""
    total:                  int = 0
    accepted:               int = 0
    rejected:               int = 0
    correct_accepted:       int = 0
    correct_rejected:       int = 0
    wrong_accepted:         int = 0
    wrong_rejected:         int = 0
    # Low-confidence MRC re-check counters
    low_conf_total:         int = 0   # how many inputs triggered the low-conf path
    low_conf_agree:         int = 0   # softmax class == MRC best class (consistent uncertain)
    low_conf_conflict:      int = 0   # softmax class != MRC best class (suspicious)
    eta_sums:               Dict[str, float] = field(default_factory=dict)
    latencies_ms:           List[float]      = field(default_factory=list)

    def update(self, result: MonitorResult, true_label: Optional[int] = None):
        """Update running totals with one new MonitorResult."""
        self.total += 1
        if result.accepted:
            self.accepted += 1
        else:
            self.rejected += 1

        # Track low-confidence MRC re-check outcomes
        if result.low_confidence:
            self.low_conf_total += 1
            if result.mrc_best_class == result.predicted_class:
                self.low_conf_agree    += 1   # MRC agrees with softmax despite low confidence
            else:
                self.low_conf_conflict += 1   # MRC disagrees — likely attack or OOD input

        # Accumulate per-layer η for mean computation in summary
        for k, v in result.eta_per_layer.items():
            self.eta_sums[k] = self.eta_sums.get(k, 0.0) + v

        if true_label is not None:
            pred_correct = (result.predicted_class == true_label)
            if result.accepted and pred_correct:
                self.correct_accepted += 1
            elif result.accepted and not pred_correct:
                self.wrong_accepted += 1      # false negative — missed adversarial
            elif not result.accepted and not pred_correct:
                self.correct_rejected += 1    # true positive — correctly blocked
            elif not result.accepted and pred_correct:
                self.wrong_rejected += 1      # false positive — blocked a clean sample

    def summary(self) -> str:
        if self.total == 0:
            return "No samples processed."

        lines = [
            "=" * 60,
            "  Phase 5 — Live Monitor Summary  [5-class]",
            "=" * 60,
            f"  Total inputs     : {self.total:,}",
            f"  Accepted         : {self.accepted:,}  "
            f"({100 * self.accepted / self.total:.1f}%)",
            f"  Rejected         : {self.rejected:,}  "
            f"({100 * self.rejected / self.total:.1f}%)",
        ]

        has_labels = (self.correct_accepted + self.wrong_accepted +
                      self.correct_rejected + self.wrong_rejected) > 0
        if has_labels:
            lines += [
                "",
                "  With ground-truth labels:",
                f"    Correct & accepted  : {self.correct_accepted:,}",
                f"    Wrong   & rejected  : {self.correct_rejected:,}  (caught!)",
                f"    Wrong   & accepted  : {self.wrong_accepted:,}  (missed!)",
                f"    Correct & rejected  : {self.wrong_rejected:,}  (false alarm)",
            ]

        # Low-confidence MRC re-check summary
        if self.low_conf_total > 0:
            lines += [
                "",
                f"  Low-confidence inputs (softmax < {SOFT_CONF_THRESHOLD:.0%}):",
                f"    Total flagged    : {self.low_conf_total:,}  "
                f"({100*self.low_conf_total/self.total:.1f}%)",
                f"    MRC agrees       : {self.low_conf_agree:,}  (uncertain but consistent)",
                f"    MRC conflicts    : {self.low_conf_conflict:,}  (softmax ≠ MRC → suspicious)",
            ]

        lines.append("")
        lines.append("  Mean η per layer:")
        for k, s in self.eta_sums.items():
            lines.append(f"    {k:10s} : {s / self.total:.5f}")

        if self.latencies_ms:
            arr = np.array(self.latencies_ms)
            lines += [
                "",
                f"  Latency  (mean)  : {arr.mean():.1f} ms / sample",
                f"  Latency  (p50)   : {np.median(arr):.1f} ms",
                f"  Latency  (p99)   : {np.percentile(arr, 99):.1f} ms",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# LIVE MONITOR
# ─────────────────────────────────────────────────────────────────
class LiveMonitor:
    """
    Wraps the frozen 5-class model + MRC signature + calibrated thresholds
    into a single inference-and-verify pipeline.

    Usage:
        monitor = LiveMonitor.from_files(
            "vgg19_phase1_best_5cls.pth",
            "mrc_signature_5cls.h5",
            "phase4_thresholds_5cls.json",
        )
        result = monitor.predict(image_tensor)
        print(result)
    """

    def __init__(self, model, hooks: dict, sig: dict,
                 thresholds: dict, reject_threshold: float = REJECT_THRESHOLD):
        self.model       = model
        self.hooks       = hooks
        self.sig         = sig
        self.thresholds  = thresholds
        self.layer_names = list(hooks.keys())
        self.reject_thr  = reject_threshold

    @classmethod
    def from_files(cls, checkpoint: str = CHECKPOINT,
                   signature: str = SIGNATURE_FILE,
                   thresholds_json: str = THRESHOLDS_FILE,
                   reject_threshold: float = REJECT_THRESHOLD) -> "LiveMonitor":
        """Build a ready-to-use LiveMonitor from the three output files of prior phases."""
        print(f"[PHASE 5] Initialising LiveMonitor...")
        print(f"  Device   : {DEVICE}")
        print(f"  Classes  : {CLASS_NAMES}")

        model = cls._load_model(checkpoint)
        hooks = cls._attach_hooks(model)
        sig   = cls._load_signature(signature)
        thr   = cls._load_thresholds(thresholds_json)

        print(f"  Reject threshold (c < {reject_threshold})\n")
        return cls(model, hooks, sig, thr, reject_threshold)

    # ── Loaders ────────────────────────────────────────────────

    @staticmethod
    def _load_model(path: str) -> LightweightVGG19:
        """Load 5-class checkpoint, freeze all params, replace Dropout with Identity."""
        print(f"  Loading model  : {path}")
        ckpt  = torch.load(path, map_location=DEVICE)
        model = LightweightVGG19()
        model.load_state_dict(ckpt["model_state_dict"])

        for p in model.parameters():
            p.requires_grad = False   # fully frozen — inference only

        # Deterministic activations: Dropout replaced by pass-through Identity
        model.classifier = nn.Sequential(*[
            nn.Identity() if isinstance(l, nn.Dropout) else l
            for l in model.classifier.children()
        ])

        model.to(DEVICE).eval()
        print(f"  Val accuracy   : {ckpt.get('val_acc', 0) * 100:.2f}%")
        return model

    @staticmethod
    def _attach_hooks(model: LightweightVGG19) -> dict:
        """Attach hooks at 3 predefined depth levels."""
        hooks = {}
        for name, (attr, idx) in HOOK_TARGETS.items():
            h = CVLayerHook(name)
            h.attach(getattr(model, attr)[idx])   # e.g. model.features[3]
            hooks[name] = h
        print(f"  Hooks attached : {list(hooks.keys())}")
        return hooks

    @staticmethod
    def _load_signature(path: str) -> dict:
        """Load the 5-class MRC signature from HDF5 into GPU tensors."""
        print(f"  Loading sig    : {path}")
        sig = {}
        with h5py.File(path, "r") as f:
            layer_names = [n.decode() for n in f["metadata"]["layer_names"][:]]
            for lname in layer_names:
                sig[lname] = {}
                grp = f[f"layer_{lname}"]
                for c in range(NUM_CLASSES):    # 5 classes
                    cgrp = grp[f"class_{c}"]
                    sig[lname][c] = {
                        "mins": torch.tensor(cgrp["mins"][:],        device=DEVICE),
                        "maxs": torch.tensor(cgrp["maxs"][:],        device=DEVICE),
                        "freq": torch.tensor(cgrp["frequencies"][:], device=DEVICE),
                    }
                n = sig[lname][0]["mins"].shape[0]
                print(f"    {lname:8s} → {n:>10,} neurons")
        return sig

    @staticmethod
    def _load_thresholds(path: str) -> dict:
        """Load calibrated τ thresholds from Phase 4 JSON."""
        print(f"  Loading τ      : {path}")
        with open(path) as f:
            return json.load(f)

    # ── Core scoring ──────────────────────────────────────────

    def _score_activations(self, predicted_classes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        For each sample, look up the per-layer η using the MODEL'S predicted class.
        Returns {layer_name: (B,) tensor of η scores}.
        Used for the high-confidence fast path.
        """
        eta = {}
        for lname, hook in self.hooks.items():
            act    = hook.activation.float()    # (B, N)
            B, _   = act.shape
            scores = torch.zeros(B, device=DEVICE)

            for cls in range(NUM_CLASSES):
                mask = (predicted_classes == cls)
                if mask.sum() == 0:
                    continue

                a    = act[mask]
                v_lo = self.sig[lname][cls]["mins"].unsqueeze(0)
                v_hi = self.sig[lname][cls]["maxs"].unsqueeze(0)
                freq = self.sig[lname][cls]["freq"]

                rng  = (v_hi - v_lo).clamp(min=1e-8)
                bidx = ((a - v_lo) / rng * Q_BUCKETS).long().clamp(0, Q_BUCKETS - 1)
                looked_up    = freq.gather(1, bidx.t())    # (N, B_cls)
                scores[mask] = looked_up.mean(dim=0)

            eta[lname] = scores
        return eta

    def _score_all_classes_single(self, sample_idx: int) -> Dict[int, float]:
        """
        For one sample (sample_idx inside the current batch), compute the combined
        mean η against EVERY class signature and return {class_idx: combined_eta}.

        This is the MRC cross-class scan used for low-confidence inputs.
        By comparing activation patterns against all 5 known class profiles,
        we can determine which class the internal features most resemble —
        independent of what the softmax output claims.
        """
        class_etas = {}
        for cls in range(NUM_CLASSES):
            layer_etas = []
            for lname, hook in self.hooks.items():
                act  = hook.activation[sample_idx:sample_idx+1].float()   # (1, N)
                v_lo = self.sig[lname][cls]["mins"].unsqueeze(0)           # (1, N)
                v_hi = self.sig[lname][cls]["maxs"].unsqueeze(0)           # (1, N)
                freq = self.sig[lname][cls]["freq"]                        # (N, 32)

                rng  = (v_hi - v_lo).clamp(min=1e-8)
                bidx = ((act - v_lo) / rng * Q_BUCKETS).long().clamp(0, Q_BUCKETS - 1)
                eta  = freq.gather(1, bidx.t()).mean().item()   # scalar η for this class+layer
                layer_etas.append(eta)

            # Combined η = mean across layers for this class signature
            class_etas[cls] = float(sum(layer_etas) / len(layer_etas))

        return class_etas

    # ── Public API ─────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor) -> List[MonitorResult]:
        """
        Run inference + MRC monitoring on a batch of normalised images.

        Two-path decision logic:

        PATH A — High confidence (softmax >= SOFT_CONF_THRESHOLD = 70%):
          Multi-layer veto using Phase 4 calibrated per-layer τ values.
          If any single layer's η < its calibrated τ → REJECT immediately.
          Then check combined η < combined τ → REJECT.

        PATH B — Low confidence (softmax < SOFT_CONF_THRESHOLD):
          MRC cross-class scan is triggered:
            1. Score activations against ALL 5 class signatures independently.
            2. Find the MRC-best class (class whose signature η is highest).
            3. If MRC-best class != softmax-best class → REJECT immediately.
               (Internal features contradict the softmax output = suspicious.)
            4. If they agree, continue through Gate 1 + Gate 2 as in Path A.

        Args:
            imgs: (B, 3, 224, 224) normalised tensor (CPU or GPU)

        Returns:
            List of MonitorResult, one per image.
        """
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)   # add batch dim for single-image input
        imgs = imgs.to(DEVICE)

        logits = self.model(imgs)           # triggers all three hooks
        probs  = F.softmax(logits, dim=1)   # (B, 5) class probability distribution
        confs, preds = probs.max(dim=1)     # highest softmax prob and its class index

        eta = self._score_activations(preds)   # per-layer η for predicted class

        # Combined η = mean across all 3 layer scores
        eta_stack  = torch.stack([eta[l] for l in self.layer_names], dim=1)
        confidence = eta_stack.mean(dim=1)     # (B,)

        # Phase 4 combined threshold — second acceptance gate
        combined_tau = self.thresholds.get("combined", {}).get(
            "tau_youden", self.reject_thr
        )

        results = []
        for i in range(imgs.size(0)):
            cls_idx    = preds[i].item()
            softmax_p  = confs[i].item()
            c_score    = confidence[i].item()
            per_layer  = {l: eta[l][i].item() for l in self.layer_names}

            low_conf       = softmax_p < SOFT_CONF_THRESHOLD   # True → use Path B
            mrc_class_etas = {}       # populated only on Path B
            mrc_best_class = cls_idx  # default: same as softmax — no conflict

            if low_conf:
                # ── Path B: MRC cross-class scan ─────────────────────
                # Score this sample's activations against all 5 class
                # signatures to find which class the features best resemble
                mrc_class_etas = self._score_all_classes_single(i)
                mrc_best_class = max(mrc_class_etas, key=mrc_class_etas.get)

            # ── Gate 0 (Path B only): conflict veto ──────────────────
            # If softmax says "cat" but MRC-best class is "airplane",
            # the internal feature pattern does not match the predicted class.
            # This is a strong adversarial signal — reject without further checks.
            if low_conf and mrc_best_class != cls_idx:
                is_safe = False

            else:
                # ── Gate 1: per-layer τ veto (both paths) ────────────
                is_safe = True
                for lname in self.layer_names:
                    layer_tau = self.thresholds.get(lname, {}).get(
                        "tau_youden", self.reject_thr
                    )
                    if per_layer[lname] < layer_tau:
                        is_safe = False   # one anomalous layer vetoes the input
                        break

                # ── Gate 2: combined threshold (both paths) ───────────
                if is_safe and c_score < combined_tau:
                    is_safe = False   # mean η too low despite per-layer pass

            results.append(MonitorResult(
                predicted_class     = cls_idx,
                class_name          = CLASS_NAMES[cls_idx],
                softmax_prob        = softmax_p,
                eta_per_layer       = per_layer,
                confidence          = c_score,
                low_confidence      = low_conf,
                mrc_best_class      = mrc_best_class,
                mrc_best_class_name = CLASS_NAMES[mrc_best_class],
                mrc_class_etas      = mrc_class_etas,
                accepted            = is_safe,
            ))

        return results

    def cleanup(self):
        """Remove all forward hooks when the monitor is no longer needed."""
        for h in self.hooks.values():
            h.remove()


# ─────────────────────────────────────────────────────────────────
# DEMO / EVALUATION
# ─────────────────────────────────────────────────────────────────
def _get_test_loader() -> DataLoader:
    """Return a DataLoader over the filtered 5-class CIFAR-10 test split."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = FilteredCIFAR10(root="./data", train=False, transform=transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=2, pin_memory=(DEVICE.type == "cuda"))


def demo_clean(monitor: LiveMonitor, max_batches: int = 10):
    """Run the monitor on clean test data and report accept/reject statistics."""
    print("\n" + "=" * 60)
    print("  Demo — Clean Test Data  [5-class CIFAR-10 subset]")
    print("=" * 60)

    loader = _get_test_loader()
    stats  = MonitorStats()

    pbar = tqdm(loader, desc="  Monitoring",
                total=min(max_batches, len(loader)))
    for batch_idx, (imgs, labels) in enumerate(pbar):
        if batch_idx >= max_batches:
            break

        t0      = time.perf_counter()
        results = monitor.predict(imgs)
        elapsed = (time.perf_counter() - t0) * 1000   # total ms for this batch

        per_sample_ms = elapsed / len(results)   # average latency per image
        for i, r in enumerate(results):
            stats.update(r, true_label=labels[i].item())
            stats.latencies_ms.append(per_sample_ms)

        pbar.set_postfix(
            acc=f"{stats.accepted}/{stats.total}",
            rej=f"{stats.rejected}",
        )

    print(stats.summary())


def demo_adversarial(monitor: LiveMonitor, eps: float = 8/255,
                     max_samples: int = 500):
    """Generate FGSM adversarial examples and test the monitor's rejection rate."""
    print("\n" + "=" * 60)
    print(f"  Demo — FGSM Adversarial (ε={eps * 255:.0f}/255)  [5-class]")
    print("=" * 60)

    # Denormalisation tensors for FGSM pixel-space perturbation
    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    loader = _get_test_loader()
    stats  = MonitorStats()
    count  = 0

    pbar = tqdm(desc="  Attacking + monitoring", total=max_samples)
    for imgs, labels in loader:
        if count >= max_samples:
            break

        imgs       = imgs.to(DEVICE).requires_grad_(True)
        labels_dev = labels.to(DEVICE)

        logits = monitor.model(imgs)
        loss   = F.cross_entropy(logits, labels_dev)
        loss.backward()   # compute gradient w.r.t. input pixels

        with torch.no_grad():
            x_px  = imgs * _STD + _MEAN
            x_adv = (x_px + eps * imgs.grad.sign()).clamp(0, 1)   # FGSM perturbation
            x_adv = (x_adv - _MEAN) / _STD                         # back to normalised

        results = monitor.predict(x_adv)

        for i, r in enumerate(results):
            if count >= max_samples:
                break
            stats.update(r, true_label=labels[i].item())
            count += 1
            pbar.update(1)

    pbar.close()
    print(stats.summary())


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[GPU]   {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Classes : {CLASS_NAMES}\n")

    # Initialise monitor from all three output files of prior phases
    monitor = LiveMonitor.from_files()

    # Demo 1: clean test images — expect high acceptance rate
    demo_clean(monitor, max_batches=20)

    # Demo 2: FGSM adversarial images — expect high rejection rate
    demo_adversarial(monitor, eps=8/255, max_samples=500)

    monitor.cleanup()
    print("\n[INFO] Monitor shut down. Hooks removed.")


if __name__ == "__main__":
    main()
