# Lung Cancer Detection — Code

Implementation accompanying the diploma "Machine Learning Methods for Identifying Lung Cancer" with all corrections from the reviewer report applied.

## Two experiments

1. **LUNA16** — CT-based lung nodule classification. Four models compared on the same dataset:
   SVM, Random Forest, KNN (trained on pyradiomics features), and ResNet50 (trained on raw patches).
   Includes a transfer learning ablation (pretrained vs random init).
2. **LC25000** — Histology image classification (lung tissue, binary: malignant vs benign).
   Same four-model lineup; classical ML trained on color/texture features.

## Methodological corrections from the report

- Single dataset per experiment (no cross-dataset comparisons).
- Fixed seed (42) across NumPy, random, PyTorch, cuDNN.
- 5-fold stratified cross-validation with documented per-fold metrics + mean ± std.
- Class imbalance handled via `class_weight='balanced'` (classical) / `pos_weight` (DL) and threshold calibration via Youden's J.
- 8 metrics reported: Accuracy, Sensitivity, Specificity, Precision, F1, AUC-ROC, PR-AUC, Balanced Accuracy.
- Wilcoxon signed-rank tests on per-fold AUC for all pairwise model comparisons.
- Transfer learning ablation for ResNet50 (pretrained ImageNet vs from scratch).
- All hyperparameters and grids fully documented in `utils/models.py` and `configs/`.

## Setup

```bash
pip install -r requirements.txt
```

## Data

- **LUNA16**: Download from https://luna16.grand-challenge.org. Extract `subset0`–`subset9`, `annotations.csv`, `candidates_V2.csv` into the path set in `configs/luna16.yaml`.
- **LC25000**: Download from https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images (or `andrewmvd/lung-and-colon-cancer-histopathological-images`). Extract to the path set in `configs/histology.yaml`.

## Running

Notebooks are numbered in execution order. From the project root:

```bash
jupyter notebook notebooks/
```

LUNA16 pipeline:
1. `01_luna16_data_prep.ipynb`   — parse annotations, extract 224×224 patches + 3D patches + masks.
2. `02_luna16_features.ipynb`     — pyradiomics features → `results/luna16_features.npz`.
3. `03_luna16_classical.ipynb`    — SVM / RF / KNN with 5-fold CV + grid search.
4. `04_luna16_resnet50_pretrained.ipynb` — ResNet50 with ImageNet weights.
5. `05_luna16_resnet50_scratch.ipynb`    — ResNet50 from random init (ablation).
6. `06_luna16_results.ipynb`      — aggregated table, ROC/PR curves, Wilcoxon p-values.

Histology pipeline:

7. `07_histology_data_prep.ipynb`
8. `08_histology_features.ipynb`
9. `09_histology_classical.ipynb`
10. `10_histology_resnet50.ipynb`
11. `11_histology_results.ipynb`

## Project layout

```
code/
├── requirements.txt
├── configs/
│   ├── luna16.yaml
│   └── histology.yaml
├── utils/
│   ├── seed.py             # set_seed for full reproducibility
│   ├── metrics.py          # all 8 metrics + Youden's J threshold calibration
│   ├── stats.py            # Wilcoxon, aggregation, formatting
│   ├── models.py           # model builders + hyperparameter grids
│   ├── training.py         # train_classical_cv, train_dl_cv
│   ├── data_luna16.py      # .mhd loading, segmentation, patch extraction
│   ├── data_histology.py   # LC25000 loading, transforms
│   ├── features_radiomics.py    # pyradiomics wrapper
│   └── features_histology.py    # GLCM/LBP/color/HED features
├── notebooks/              # 11 notebooks (numbered execution order)
└── results/                # per-fold JSON + CSV + figures
```

## Final hyperparameters

| Model | Grid / Configuration |
|---|---|
| SVM | C ∈ {0.1, 1, 10, 100}, kernel ∈ {linear, rbf}, gamma ∈ {scale, 0.01, 0.1}, `class_weight='balanced'`, inner 3-fold CV by ROC-AUC |
| RF | n_estimators ∈ {100, 200, 500}, max_depth ∈ {None, 10, 20}, min_samples_split ∈ {2, 5, 10}, `class_weight='balanced'`, inner 3-fold CV by ROC-AUC |
| KNN | n_neighbors ∈ {3, 5, 7, 9, 11, 15, 21}, weights ∈ {uniform, distance}, p ∈ {1, 2}, inner 3-fold CV by ROC-AUC |
| ResNet50 | lr=1e-4, batch=32, optimizer=AdamW, weight_decay=1e-4, epochs=50, scheduler=CosineAnnealingLR(T_max=50), early stopping patience=7, loss=BCEWithLogitsLoss(pos_weight=N_neg/N_pos) |

All numeric results — tables, figures, narrative — come from a single run with `seed=42` aggregated by `06_luna16_results.ipynb` / `11_histology_results.ipynb`.
