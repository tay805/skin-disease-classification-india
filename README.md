# Pipeline for Indian Skin Disease Diagnosis

A complete DL pipeline for classifying skin diseases from the DermaCon-IN dataset — the first large-scale dermatology dataset focused on South Indian patients.

**Dataset:** https://datasetcatalog.nlm.nih.gov/dataset?q=0000091311  


## Overview

This notebook implements a dual-branch deep learning model that fuses clinical images with structured patient metadata (body-part location and lesion descriptors) for skin disease classification. The pipeline covers the full spectrum from data loading to publication-quality analysis.

The dataset comprises 5,450 images from 3,002 patients, spanning 245 diseases grouped into 8 main aetiology classes, with Fitzpatrick skin type annotations (Types III to VI) for fairness evaluation.

---

## Model Architecture

The model combines EfficientNetV2-L (pretrained on ImageNet at 384px resolution) with a metadata MLP in a late-fusion design:

```
EfficientNetV2-L (384px)     Clinical Metadata (96-dim)
        |                              |
  img_proj: feat → 1024         MLP: 96 → 256 → 128
        |                              |
        +------------ cat  ------------+
                        |
                   z ∈ R^1152
                        |
               head (8 main classes)
```

The metadata branch encodes 49 body-part one-hot flags and 47 lesion descriptor one-hot flags into a 128-dimensional vector. The fused 1152-dimensional representation feeds into the classification head.

---

## Pipeline Stages

| Stage | Description |
|---|---|
| 1 | Environment setup and package installation |
| 1b | Sanity check on data files and image folders |
| 2 | Data loading and exploratory analysis |
| 3 | Patient-disjoint validation split (StratifiedGroupKFold) |
| 4 | Augmentation pipeline tuned for Indian clinical photography |
| 5-6 | Dataset class and DataLoaders with weighted sampling |
| 7 | Model architecture (EfficientNetV2-L + metadata fusion) |
| 8 | Loss functions (Focal Loss + MixUp) |
| 9 | Two-stage training (freeze/unfreeze) |
| 10 | Test evaluation with per-class breakdown |
| 11 | Test-Time Augmentation (TTA x5) |
| 12 | GradCAM++ / saliency explainability |
| 13 | ONNX export and inference demo |
| A | Multi-seed evaluation (3 seeds, mean +/- std) |
| B | Baseline comparisons (ResNet-50, EfficientNet-B0) |
| C | Ablation study (with vs without metadata fusion) |
| D | Fitzpatrick skin-type fairness breakdown |
| E | Stage 2: fine-grained disease prediction per aetiology group |

---

## Key Design Choices

**Two-stage training:** The backbone is frozen for 5 warm-up epochs (Stage 1) then fully unfrozen at a lower learning rate (Stage 2). This prevents catastrophic forgetting of ImageNet features early in training.

**Augmentation for Indian skin tones:** The augmentation pipeline applies CLAHE, HueSaturationValue, and RandomBrightnessContrast to improve luminance normalisation across Fitzpatrick Types III to VI. CoarseDropout and GridDropout simulate hair and clothing occlusion common in clinical photographs.

**Weighted random sampling:** Training batches use inverse-frequency sampling over the 8 main aetiology classes to prevent the majority class from dominating gradient updates.

**Focal Loss with label smoothing:** Focal Loss (gamma=2.0) combined with label smoothing (epsilon=0.05) handles the approximately 58x class imbalance between the largest and smallest aetiology groups.

**Reproducible evaluation:** All final results are reported as mean +/- std across 3 random seeds (42, 123, 7). Each seed trains independently with its own patient-stratified validation split.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Backbone | tf_efficientnetv2_l |
| Image resolution | 384 x 384 px |
| Batch size | 16 (effective 64 with 4 gradient accumulation steps) |
| Learning rate (Stage 1 heads) | 3e-4 |
| Learning rate (Stage 2 full) | 5e-5 |
| Warmup epochs | 5 |
| Total epochs | 50 |
| Early stopping patience | 10 |
| Optimizer | AdamW (weight decay 1e-2) |
| Scheduler | Cosine annealing |
| MixUp alpha | 0.3 |
| Focal gamma | 2.0 |
| Label smoothing | 0.05 |
| Dropout | 0.30 (backbone), 0.40 (image projection) |
| Mixed precision | FP16 |

---

## Dataset Structure

The notebook expects the following layout on Google Drive:

```
skinDisease/
    dataset/
        dataset_0/          # image files (.jpg / .png)
        dataset_1/          # image files (.jpg / .png)
    train.csv               # training split with 108 columns
    test.csv                # test split with 108 columns
    skinMetadata.csv        # raw unpivoted metadata
    pipeline_outputs/       # created automatically
```

The CSV files contain: image name, disease label, main aetiology class, sub-class, Subject ID, Fitzpatrick type, Monk skin tone, sex, age group, confidence score, 49 body-part one-hot columns (prefix Body_part_), and 47 lesion descriptor one-hot columns (prefix Descriptor_).

---

## Output Files

After a full run the pipeline writes the following to `pipeline_outputs/`:

| File | Description |
|---|---|
| best_model_8class.pth | Best checkpoint from single-seed training |
| dermaconin_8class.onnx | Exported ONNX model for inference |
| label_maps_8class.json | Class names and metadata column list |
| classification_report_8class.csv | Per-class precision, recall, F1 |
| confusion_matrix_8class.png | Normalised 8x8 confusion matrix |
| training_curves_8class.png | Loss and accuracy curves (Stage 2) |
| f1_8class.png | Per-class F1 bar chart |
| accuracy_by_class_8class.png | Per-class top-1 accuracy |
| saliency_8class.png | Gradient saliency maps (8 test samples) |
| inference_demo_8class.png | Single-image prediction demo |
| multi_seed_results.json | Mean and std across 3 seeds |
| baseline_comparison.csv | ResNet-50 vs EfficientNet-B0 vs ours |
| ablation_results.json | With vs without metadata fusion |
| fitzpatrick_breakdown.csv | Accuracy by skin type, sex, age group |
| equity_analysis.png | Fairness visualisation across subgroups |
| stage2_results.csv | Fine-grained per-disease accuracy |
| seed_*/best_model.pth | Per-seed best checkpoints |

---

## Requirements

The notebook runs on Google Colab with a GPU runtime. The setup cell installs all dependencies:

```
timm==1.0.3
albumentations
grad-cam
torchmetrics
```

Standard scientific Python packages (numpy, pandas, matplotlib, seaborn, scikit-learn, opencv-python) are pre-installed in Colab. PyTorch with CUDA is also pre-installed.

---

## Resumable Training

All training loops save a resume checkpoint after every epoch. If a Colab session disconnects mid-training, re-running the training cell picks up from the last completed epoch without restarting from scratch. A `done.flag` file prevents re-running a fully completed seed or baseline.

To force a complete restart, uncomment the cleanup cell above the training loop and re-run.

---

## Citation

If you use this pipeline, please cite:

A Two-Stage Fusion-based Deep Learning Classifier for Indian Skin Disease, Tawseef Ahmed Teli et al.

---

## License

See the dataset repository on Harvard Dataverse for dataset licensing terms. The pipeline code in this notebook is released for research use.
