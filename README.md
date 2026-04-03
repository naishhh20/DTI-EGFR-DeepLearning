# Predicting Drug–Target Interactions Using Deep Learning
### EGFR Binary Classification: ResNet-18 CNN vs Random Forest

**CS697-AI687-001 | AI & Machine Learning in Bioinformatics | Spring 2026**  
**Group 3:** Naisargi Sharma · Maria Elizathe · Vaibhav Jaiswal  
**Professor:** Reda Nacif Elalaoui | Long Island University

---

## Project Overview

This project implements and compares two machine learning approaches for predicting whether a drug compound will bind to the **Epidermal Growth Factor Receptor (EGFR)** — a key oncology target linked to non-small cell lung cancer (NSCLC) and colorectal cancer.

| Approach | Input | Model |
|---|---|---|
| **Deep Learning** | 2D molecular structure images (224×224 px) | ResNet-18 CNN (transfer learning) |
| **Baseline** | Morgan fingerprints (radius=2, 2048 bits) | Random Forest classifier |

**Task:** Binary classification — Binder (1): IC50 ≤ 1 µM · Non-binder (0): IC50 > 10 µM

---

## Results

| Metric | ResNet-18 CNN | Random Forest |
|---|---|---|
| **ROC-AUC** | 0.9605 | **0.9865** |
| **PR-AUC** | 0.9913 | **0.9970** |
| **F1-Score** | 0.9510 | **0.9755** |
| **Accuracy** | 0.9197 | **0.9591** |
| **Precision** | 0.9647 | **0.9692** |
| **Recall** | 0.9376 | **0.9819** |

Both models significantly exceed the target ROC-AUC of 0.80.  
Best CNN validation AUC: **0.9627 at epoch 23** (trained on Google Colab T4 GPU).  
RF 5-fold CV ROC-AUC: **0.9803 ± 0.0043**.

---

## Dataset

- **Source:** [BindingDB](https://www.bindingdb.org/) — EGFR (UniProt: P00533)
- **Raw records:** 28,809
- **After preprocessing:** 12,870 unique compounds
- **Active binders:** 10,683 (83.0%) · **Inactive non-binders:** 2,187 (17.0%)
- **Split:** 9,008 train / 1,931 val / 1,931 test (stratified 70/15/15)

> **Dataset download:** The BindingDB EGFR dataset must be downloaded separately from [BindingDB](https://www.bindingdb.org/bind/chemsearch/marvin/SDFdownload.jsp?all_download=yes). Export the EGFR entries as a TSV file and place it at the path specified in `DATA_PATH` inside the notebook (Cell 2).

---

## Repository Structure

```
DTI-EGFR-DeepLearning/
│
├── DTI_EGFR_Final.ipynb              # Main notebook — complete pipeline (21 cells)
│
├── requirements.txt                   # All Python dependencies
├── README.md                          # This file
├── .gitignore                         # Excludes dataset, images, model weights
│
├── results/
│   └── final_results.csv             # Metric comparison table (CNN vs RF)
│
└── report/
    └── Naisargi_Sharma_group3_CS697-AI687-Spring-2026_Report_FINAL.docx
```

---

## Notebook Structure (`DTI_EGFR_Final.ipynb`)

| Cell | Section | Description |
|---|---|---|
| 0 | Install Dependencies | Installs rdkit, torch, scikit-learn, etc. |
| 1 | Imports | All libraries + GPU detection + random seed |
| 2 | Configuration | All hyperparameters and paths in one place |
| 3 | Load Data | `load_bindingdb_data()` — auto-detects columns |
| 4 | Preprocessing | `preprocess_and_label()` — 5-step pipeline |
| 5–6 | Dataset Exploration | Class distribution + IC50 histogram plots |
| 7 | Image Generation | `smiles_to_image()` + `generate_images()` |
| 8 | Dataset Split | `split_dataset()` — stratified 70/15/15 |
| 9 | DataLoader | `MolecularImageDataset` + class weights |
| 10 | Build Model | `build_resnet18()` — transfer learning setup |
| 11 | Training | `train_model()` — BCE loss, Adam, early stopping |
| 12 | Training Curves | Loss and AUC plots across epochs |
| 13 | Random Forest | `train_random_forest()` — Morgan fingerprints + 5-fold CV |
| 14 | CNN Evaluation | `evaluate_model()` — all 6 metrics on test set |
| 15–16 | Model Comparison | Side-by-side table + bar chart |
| 17 | ROC & PR Curves | ROC-AUC and PR-AUC plots for both models |
| 18 | Confusion Matrices | TP/TN/FP/FN for both models |
| 19 | Error Analysis | False positive / false negative confidence histograms |
| 20 | Save Results | Saves CSV, .pth, .pkl, and all plots |

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `DTI_EGFR_Final.ipynb`
3. Upload your BindingDB EGFR TSV file via the Colab sidebar
4. In **Cell 2**, update `DATA_PATH` to point to your uploaded file:
   ```python
   DATA_PATH = '/content/your_bindingdb_file.tsv'
   ```
5. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
6. Run All (Runtime → Run all)

### Option 2 — Local (Jupyter Notebook)

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DTI-EGFR-DeepLearning.git
   cd DTI-EGFR-DeepLearning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your BindingDB TSV file in the project folder and update `DATA_PATH` in Cell 2.

4. Launch Jupyter:
   ```bash
   jupyter notebook DTI_EGFR_Final.ipynb
   ```

> **Note:** Local CPU training is very slow (~8 hours/epoch). A GPU is strongly recommended.

---

## Requirements

See `requirements.txt` for the full list. Key dependencies:

```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
rdkit>=2023.3.1
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0
tqdm>=4.65.0
```

---

## Key Design Decisions

**Why ResNet-18 for molecular images?**  
EGFR inhibitors share a common anilinoquinazoline pharmacophore scaffold — a spatially distinctive ring pattern visible in 2D structure images. CNNs detect local spatial features (ring systems, functional groups) that directly correspond to the local nature of molecular binding interactions, as established by AtomNet (Wallach et al., 2015).

**Why Morgan fingerprints as baseline?**  
Morgan fingerprints (ECFP4) are the industry standard for molecular similarity and classification tasks. They directly encode the chemical substructure patterns relevant to EGFR binding, making them a strong and fair comparison baseline.

**Why PR-AUC alongside ROC-AUC?**  
Our dataset is 83% active (imbalanced). ROC-AUC can be misleadingly optimistic for imbalanced datasets. PR-AUC focuses on the positive class and is more informative for drug discovery screening tasks, as highlighted by Rayhan et al. (2018).

---

## References

1. Wallach, I., Dzamba, M., & Heifets, A. (2015). AtomNet: A deep convolutional neural network for bioactivity prediction. *arXiv:1510.02855*.
2. Rayhan, F., et al. (2018). FRnet-DTI: Deep CNNs with evolutionary and structural features for drug-target interaction. *arXiv:1806.07174*.
3. Debnath, K., Rana, P., & Ghosh, P. (2025). A survey on deep learning for drug-target binding prediction. *Briefings in Bioinformatics, 26*(5), bbaf491.
4. Öztürk, H., et al. (2018). DeepDTA: Deep drug-target binding affinity prediction. *Bioinformatics, 34*(17), i821–i829.
5. Rifaioglu, A. S., et al. (2020). DEEPScreen: High performance DTI prediction with CNNs. *Chemical Science, 11*(9), 2531–2557.

---

## Team Contributions

| Member | Role | Contributions |
|---|---|---|
| **Naisargi Sharma** | Deep Learning & Experiments | ResNet-18 CNN architecture, training pipeline, transfer learning setup, model evaluation, error analysis code |
| **Vaibhav Jaiswal** | Data Engineering & Baseline | BindingDB acquisition, preprocessing pipeline, image generation, Morgan fingerprints, Random Forest training |
| **Maria Elizathe** | Biological Validation | EGFR clinical relevance (T790M, L858R), IC50 threshold validation, literature review, pharmacological interpretation |

---

## License

This project is open source for educational and research purposes.  
CS697-AI687-001 · Long Island University · Spring 2026
