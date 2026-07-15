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
- **Validation:** Hold-out (80/20), 5-Fold Cross-Validation, GridSearchCV hyperparameter tuning

## Results

### Hold-out (80/20 split)

| Classifier | Accuracy | F1-Score | Sensitivity | Specificity | Precision |
|------------|----------|----------|-------------|-------------|-----------|
| SVM-RBF    | 78.97%   | 80.63%   | 87.07%      | 70.79%      | 75.07%    |
| Random Forest | 78.38% | 78.69% | 79.42%     | 77.32%      | 77.96%    |
| XGBoost    | 75.90%   | 76.38%   | 77.55%      | 74.23%      | 75.25%    |

### 5-Fold Cross-Validation

| Classifier | Mean Accuracy | Std |
|------------|--------------|-----|
| SVM-RBF    | 77.76%       | ±0.83% |
| Random Forest | 76.99%    | ±0.79% |
| XGBoost    | 75.52%       | ±0.87% |

Best model: **SVM-RBF** (Accuracy: 78.97%, F1: 80.63%). GridSearchCV was applied to Random Forest, confirming stable performance at 78.38%.

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
