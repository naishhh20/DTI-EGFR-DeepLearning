# Predicting Drug–Target Interactions Using Deep Learning
### CS697-AI687-001 | AI & Machine Learning in Bioinformatics | Spring 2026
**Long Island University | Group 3**

> **Naisargi Sharma · Maria Elizathe · Vaibhav Jaiswal**  
> Professor: Reda Nacif Elalaoui

---

## 📌 Project Overview

This project predicts **Drug-Target Interactions (DTI)** for the **EGFR protein** (Epidermal Growth Factor Receptor, UniProt: P00533) using two approaches:

- **Model A — ResNet-18 CNN**: Predicts binding from 2D molecular structure images (224×224 px) generated via RDKit
- **Model B — Random Forest Baseline**: Predicts binding from Morgan fingerprint descriptors (radius=2, 2048 bits)

Both models are trained and evaluated on the **BindingDB EGFR dataset** with IC50-based binary labeling.

---

## 🎯 Task

**Binary Classification:**
- **Binder (1):** IC50 ≤ 1 µM (active against EGFR)
- **Non-binder (0):** IC50 > 10 µM (inactive)
- Ambiguous compounds (1–10 µM) excluded to reduce label noise

---

## 📊 Final Results

| Metric | ResNet-18 CNN | Random Forest |
|--------|--------------|---------------|
| Accuracy | 0.921 | 0.959 |
| Precision | 0.965 | 0.969 |
| Recall | 0.939 | 0.982 |
| F1-Score | 0.952 | 0.976 |
| ROC-AUC | 0.957 | 0.987 |
| **PR-AUC** | **0.990** | **0.997** |

> PR-AUC is the primary metric per Debnath et al. (2025), given the class imbalance (83% binders vs 17% non-binders).

---

## 🗂️ Repository Structure

```
DTI-EGFR-DeepLearning/
│
├── notebooks/
│   └── DTI_EGFR_Prediction.ipynb     # Main Colab notebook (run on T4 GPU)
│
├── results/
│   ├── final_results.csv             # Metrics comparison table
│   ├── training_curves.png           # Loss + AUC across epochs
│   ├── model_comparison.png          # CNN vs RF bar chart
│   ├── roc_pr_curves.png             # ROC and PR curves
│   ├── confusion_matrices.png        # TP/TN/FP/FN for both models
│   ├── gradcam_analysis.png          # Grad-CAM heatmaps
│   ├── dataset_exploration.png       # Class distribution + IC50 histogram
│   └── sample_molecules.png          # Sample molecular images
│
├── docs/
│   └── project_report.pdf            # Final IEEE-format report
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files to exclude
└── README.md                         # This file
```

> **Note:** Model weights (`best_resnet18.pth`, `random_forest_egfr.pkl`) and the BindingDB dataset TSV are **not included** due to file size. See instructions below to reproduce.

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
1. Open `notebooks/DTI_EGFR_Prediction.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU → Save**
3. Upload your BindingDB EGFR TSV file and update `DATA_PATH` in Step 3
4. Go to **Runtime → Run all**

### Option 2 — Local Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DTI-EGFR-DeepLearning.git
cd DTI-EGFR-DeepLearning

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/DTI_EGFR_Prediction.ipynb
```

---

## 📦 Dataset

- **Source:** [BindingDB](https://www.bindingdb.org/) — Public Bioactivity Database
- **Target:** EGFR (Epidermal Growth Factor Receptor), UniProt P00533
- **Format:** Tab-Separated Values (TSV)
- **Raw entries:** 28,809
- **After preprocessing:** 12,870 unique compounds
  - Active (Binder): 10,683 (83.0%)
  - Inactive (Non-binder): 2,187 (17.0%)

To download the dataset:
1. Go to [BindingDB Query](https://www.bindingdb.org/bind/chemsearch/marvin/SDFdownload.jsp?download_file=no&ic50_relations=le,gt&ic50=&ki_relations=le,gt&ki=&kd_relations=le,gt&kd=&ec50_relations=le,gt&ec50=&target_string=EGFR)
2. Search for target: **EGFR**
3. Download as TSV

---

## 🧠 Model Details

### ResNet-18 CNN
| Parameter | Value |
|-----------|-------|
| Base model | ResNet-18 (ImageNet pretrained) |
| Input | 224×224 RGB molecular images |
| Final layer | Dropout(0.3) → Linear(512→1) |
| Loss | BCEWithLogitsLoss (pos_weight=0.205) |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Best epoch | 20 (Val AUC = 0.964) |
| Early stopping | Triggered at Epoch 25 (patience=5) |

### Random Forest Baseline
| Parameter | Value |
|-----------|-------|
| Estimators | 100 |
| Features | Morgan fingerprints (radius=2, 2048 bits) |
| Class weight | Balanced |
| CV ROC-AUC | 0.9803 ± 0.0043 (5-fold stratified) |

---

## 🔍 Interpretability — Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) applied to ResNet-18's final convolutional layer (`layer4`) to visualize which molecular regions drive binding predictions.

- **Red/Yellow** = high CNN attention
- **Blue** = low attention
- False negatives show diffuse, unfocused attention patterns — consistent with AtomNet's **Feature Locality** principle (Wallach et al., 2015)

---

## ⚖️ Ethical Considerations

- **Model Errors:** CNN missed 98 true binders (false negatives) — these models are **decision-support tools only**, not final decision-makers
- **Interpretability:** Grad-CAM improves transparency but attention can be inconsistent — reliability must be assessed per prediction
- **Dataset Bias:** Dataset limited to known EGFR binders from BindingDB — may not generalize to novel scaffolds or underrepresented populations
- **Deployment:** Best approach is a hybrid system (RF for screening + CNN for interpretability) + mandatory wet-lab experimental validation

---

## 📚 Key References

1. Öztürk, H., et al. (2018). DeepDTA. *Bioinformatics*, 34(17), i821–i829.
2. Rifaioglu, A. S., et al. (2020). DEEPScreen. *Chemical Science*, 11(9), 2531–2557.
3. Wallach, I., et al. (2015). AtomNet. *arXiv:1510.02855*
4. Debnath, S., Raza, A., & Ghosh, S. (2025). Survey of DL for DTI Prediction. *Bioinformatics Advances*, 4(1).
5. Zhang, Y., et al. (2023). Deep learning-based DTI prediction. *Briefings in Bioinformatics*, 24(3).

---

## 👥 Team Contributions

| Member | Role | Contributions |
|--------|------|---------------|
| **Naisargi Sharma** | Deep Learning & Experimental Design | ResNet-18 architecture, training pipeline, model evaluation, Grad-CAM |
| **Vaibhav Jaiswal** | Data Engineering & Baseline Modeling | BindingDB acquisition, SMILES cleaning, image generation, RF baseline |
| **Maria Elizathe** | Biological Validation & Analysis | EGFR biological justification, IC50 threshold validation, literature review |

---

## 📄 License

This project is for academic purposes only — CS697-AI687-001, Long Island University, Spring 2026.
