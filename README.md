# EMG-Based Hand Gesture Classification Using Machine Learning

**Course:** T550021101-1-F25 Tools of Artificial Intelligence (F25)  
**Student:** Amirhossein Taleshinosrati (ID: 4144308)  
**Program:** MSc Drone Technology and Autonomous Systems, SDU  
**Submission Date:** 24-05-2026

---

## Project Overview

Binary classification of EMG signals to distinguish **Rock** vs **Paper** hand gestures using traditional machine learning.

- **Dataset:** 5850 trials from the Ninapro DB2 dataset
- **Features:** 26 features across 4 domains (time-domain, frequency-domain, time-frequency, statistical)
- **Classifiers:** Random Forest, SVM-RBF, XGBoost
- **Validation:** Hold-out, k-Fold Cross-Validation, Leave-One-Subject-Out (LOSO)

## Results

| Classifier | Accuracy | F1-Score | AUC |
|------------|----------|----------|-----|
| SVM-RBF    | 79.40%   | —        | —   |
| Random Forest | 78.38% | —      | —   |
| XGBoost    | 78.12%   | —        | —   |

Best model: **SVM-RBF** after hyperparameter tuning via GridSearchCV.

## Repository Structure

```
├── AItools_project_final.ipynb   # Main Jupyter notebook with full pipeline
├── aitools_project_final.py      # Python script version
├── make_notebook.py              # Utility to build the notebook
├── performance_report.pdf        # Performance evaluation report
├── latex_report/                 # LaTeX source for the written report
│   └── latex_report/
│       ├── main.tex              # Single-file LaTeX report
│       └── figures/              # All figures used in the report
└── prof_format/                  # Professor's original LaTeX template
```

## How to Run

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn scipy
jupyter notebook AItools_project_final.ipynb
```
