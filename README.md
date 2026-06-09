# Signature-Aware Fine-Tuning Pipeline for DNNs

Multi-phase pipeline for signature-aware fine-tuning of deep neural networks on a 5-class CIFAR-10 subset (airplane, automobile, bird, cat, deer).

## Phases

| Phase | Script | Output |
|-------|--------|--------|
| 1 | `phase1.py` | `vgg19_phase1_best_5cls.pth` |
| 2–3 | `phase2_3_signature.py` | `mrc_signature_5cls.h5` |
| 4 | `phase4_calibration.py` | `phase4_thresholds_5cls.json` |
| 5 | `phase5_live_monitor.py` | Live ACCEPT/REJECT monitoring |
| 6 | `phase6_random_demo.py` | Visualization demos |

## Requirements

```bash
pip install torch torchvision h5py tqdm matplotlib
```

## Usage

Run phases in order (1 → 2/3 → 4 → 5 or 6). Each phase depends on outputs from the previous one.
