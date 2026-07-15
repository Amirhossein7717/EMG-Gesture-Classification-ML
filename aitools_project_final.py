# -*- coding: utf-8 -*-
# =============================================================================
# Gesture Classification: Rock vs Paper
# Tools of Artificial Intelligence -- Final Project
# Module 2: Machine Learning Pipeline for Classification
# University of Southern Denmark -- The Maersk Mc-Kinney Moller Institute
# =============================================================================
# Pipeline:
#   1. Data Loading & Cleaning
#   2. Data Visualisation
#   3. Feature Extraction (Statistical, Time-Domain, Frequency-Domain, Entropy)
#   4. Feature Selection (Standardise -> Kruskal-Wallis -> PCA)
#   5. Classification (Random Forest, SVM-RBF, XGBoost)
#   6. Validation (Hold-out, 5-Fold Cross-Validation, GridSearchCV)
#   7. Confusion Matrix & ROC Curve
#   8. Performance Report & Comparison
# =============================================================================

# --- Step 1: Import Libraries ------------------------------------------------

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
import antropy as ant
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

print('All libraries imported successfully')

# --- Step 2: Load Data -------------------------------------------------------
# Place RockData.csv and PaperData.csv in the same directory as this script,
# or adjust the paths below accordingly.

import os
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'extracted', 'GestureClassification',
                     'GestureClassification')
rock_df  = pd.read_csv(os.path.join(_base, 'RockData.csv'))
paper_df = pd.read_csv(os.path.join(_base, 'PaperData.csv'))

# Output directory: same folder as this script
_out_dir = os.path.dirname(os.path.abspath(__file__))

# Align column names (both files have 1024 columns numbered 0-1023)
paper_df.columns = rock_df.columns

# Add class labels: Rock = 0, Paper = 1
rock_df['label']  = 0
paper_df['label'] = 1

# Merge into one combined dataset
data = pd.concat([rock_df, paper_df], axis=0, ignore_index=True)

print(f'Combined dataset shape: {data.shape}')
print(f'Class distribution:\n{data["label"].value_counts().rename({0: "Rock", 1: "Paper"})}')

# --- Step 3: Data Cleaning ---------------------------------------------------

# Task 2 & 3: Separate features and labels
X_raw = data.drop(columns=['label'])
y     = data['label'].values

# Convert all columns to numeric (handles any stray string values)
X_raw = X_raw.apply(pd.to_numeric, errors='coerce')

# Task 5: Check for missing values
total_nans = X_raw.isna().sum().sum()
print(f'Total NaN values: {total_nans}')

if total_nans > 0:
    # Mean imputation as discussed in class
    X_raw = X_raw.fillna(X_raw.mean())
    print('NaN values filled using mean imputation')
else:
    print('No missing values -- dataset is complete')

print(f'\nFeature matrix shape : {X_raw.shape}')   # (5850, 1024)
print(f'Label vector shape   : {y.shape}')          # (5850,)

# Task 6: Channel identification
n_samples_per_trial = X_raw.shape[1]
sampling_freq       = 256   # Hz, as per DatasetDescription.docx
duration_sec        = n_samples_per_trial / sampling_freq

print(f'\nChannel Information')
print(f'  Number of channels   : 1 (single-channel EMG)')
print(f'  Samples per trial    : {n_samples_per_trial}')
print(f'  Sampling frequency   : {sampling_freq} Hz')
print(f'  Duration per trial   : {duration_sec:.2f} s (~4 s)')
print(f'  Proceeding to feature extraction on single channel')

# --- Step 4: Data Visualisation ----------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Gesture Dataset -- Exploratory Visualisation', fontsize=14, fontweight='bold')

# Plot 1: Sample signals
ax = axes[0, 0]
ax.plot(X_raw.iloc[0].values, color='steelblue', alpha=0.8, label='Rock (trial 1)')
ax.plot(X_raw.iloc[2909].values, color='coral', alpha=0.8, label='Paper (trial 1)')
ax.set_title('Sample EMG Signal -- Rock vs Paper')
ax.set_xlabel('Sample index')
ax.set_ylabel('Amplitude')
ax.legend()

# Plot 2: Class distribution
ax = axes[0, 1]
counts = [np.sum(y == 0), np.sum(y == 1)]
bars = ax.bar(['Rock', 'Paper'], counts, color=['steelblue', 'coral'], edgecolor='white')
ax.set_title('Class Distribution')
ax.set_ylabel('Number of trials')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            str(count), ha='center', fontsize=11)

# Plot 3: Mean signal per class
ax = axes[1, 0]
rock_mean  = X_raw.values[y == 0].mean(axis=0)
paper_mean = X_raw.values[y == 1].mean(axis=0)
ax.plot(rock_mean,  color='steelblue', label='Rock mean')
ax.plot(paper_mean, color='coral',     label='Paper mean')
ax.set_title('Mean Signal per Class')
ax.set_xlabel('Sample index')
ax.set_ylabel('Mean amplitude')
ax.legend()

# Plot 4: Amplitude density distribution
ax = axes[1, 1]
ax.hist(X_raw.values[y == 0].flatten(), bins=80, alpha=0.6, color='steelblue',
        label='Rock', density=True)
ax.hist(X_raw.values[y == 1].flatten(), bins=80, alpha=0.6, color='coral',
        label='Paper', density=True)
ax.set_title('Amplitude Distribution')
ax.set_xlabel('Amplitude')
ax.set_ylabel('Density')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(_out_dir, 'visualisation.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Visualisation saved as visualisation.png')

# --- Step 5: Feature Extraction ----------------------------------------------
# Each trial (row) contains 1024 time-series samples.
# 26 features are extracted per trial across four domains:
# statistical (8), time-domain (7), frequency-domain (6), entropy (5).
#
# Note on filtering:
# According to DatasetDescription.docx the signals were already filtered before
# recording. No additional filtering (notch or bandpass) is applied here.
# The data is used directly for feature extraction at fs = 256 Hz.

def extract_statistical_features(sig):
    return {
        'Mean'     : np.mean(sig),
        'Median'   : np.median(sig),
        'Std Dev'  : np.std(sig),
        'Variance' : np.var(sig),
        'Skewness' : stats.skew(sig),
        'Kurtosis' : stats.kurtosis(sig),
        'Range'    : np.ptp(sig),
        'IQR'      : np.percentile(sig, 75) - np.percentile(sig, 25),
    }


def extract_time_domain_features(sig):
    rms           = np.sqrt(np.mean(sig ** 2))
    zero_crossings = np.sum((sig[:-1] * sig[1:]) < 0)
    autocorr      = np.correlate(sig, sig, mode='full')[len(sig) - 1]
    mean_abs_dev  = np.mean(np.abs(sig - np.mean(sig)))
    signal_energy = np.sum(sig ** 2)
    return {
        'RMS'                : rms,
        'Zero Crossings'     : zero_crossings,
        'Autocorrelation'    : autocorr,
        'Mean Abs Deviation' : mean_abs_dev,
        'Max Value'          : np.max(sig),
        'Min Value'          : np.min(sig),
        'Signal Energy'      : signal_energy,
    }


def extract_frequency_domain_features(sig, fs=256):
    freqs, psd    = signal.welch(sig, fs)
    dominant_freq = freqs[np.argmax(psd)]
    total_power   = np.sum(psd)
    band_power    = np.sum(psd[(freqs >= 0.5) & (freqs <= 40)])
    mean_freq     = np.mean(freqs)
    median_freq   = np.median(freqs)
    freq_variance = np.var(freqs)
    return {
        'Dominant Freq'       : dominant_freq,
        'Total Power'         : total_power,
        'Band Power 0.5-40Hz' : band_power,
        'Mean Freq'           : mean_freq,
        'Median Freq'         : median_freq,
        'Freq Variance'       : freq_variance,
    }


def extract_entropy_features(sig, fs=256):
    # Explicit float64 conversion and binary thresholding for numerical stability
    sig = np.array(sig, dtype=np.float64, order='C')
    binary_signal = (sig > np.median(sig)).astype(int)
    return {
        'Sample Entropy'      : ant.sample_entropy(sig),
        'Spectral Entropy'    : ant.spectral_entropy(sig, sf=fs, method='welch'),
        'Permutation Entropy' : ant.perm_entropy(sig, normalize=True),
        'SVD Entropy'         : ant.svd_entropy(sig, order=3, normalize=True),
        'LZiv Complexity'     : ant.lziv_complexity(binary_signal, normalize=True),
    }


print('Feature extraction functions defined')

# Run extraction across all trials (~3-5 minutes due to entropy computation)
print('Extracting features...')

stat_features    = [extract_statistical_features(row.values)      for _, row in X_raw.iterrows()]
print('  Statistical features done')

time_features    = [extract_time_domain_features(row.values)      for _, row in X_raw.iterrows()]
print('  Time-domain features done')

freq_features    = [extract_frequency_domain_features(row.values) for _, row in X_raw.iterrows()]
print('  Frequency-domain features done')

entropy_features = [extract_entropy_features(row.values)          for _, row in X_raw.iterrows()]
print('  Entropy features done')

# Combine all features into one DataFrame
combined = []
for s, t, f, e in zip(stat_features, time_features, freq_features, entropy_features):
    combined.append({**s, **t, **f, **e})

features_df = pd.DataFrame(combined)
print(f'\nCombined feature matrix shape: {features_df.shape}')
print(f'Features per trial: {features_df.shape[1]}')
print(features_df.head(3))

# --- Step 6: Feature Selection -----------------------------------------------
# Pipeline: Standardise -> Kruskal-Wallis -> PCA

# 6a. Standardise
scaler      = StandardScaler()
X_scaled    = scaler.fit_transform(features_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=features_df.columns)
print(f'Standardised features shape: {X_scaled_df.shape}')

# 6b. Feature set comparison (Task 13 ablation)
feature_groups = {
    'Statistical only'      : list(range(0,  8)),
    'Time-domain only'      : list(range(8,  15)),
    'Frequency-domain only' : list(range(15, 21)),
    'Entropy only'          : list(range(21, 26)),
    'All features combined' : list(range(0,  26)),
}

print('=== FEATURE SET COMPARISON (Random Forest, Hold-out 80/20) ===\n')
fs_results = {}

for fs_name, indices in feature_groups.items():
    X_fs = X_scaled[:, indices]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fs, y, test_size=0.2, random_state=42, stratify=y
    )
    clf_fs = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_fs.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf_fs.predict(X_te)) * 100
    fs_results[fs_name] = acc
    print(f'{fs_name:25s}  Accuracy: {acc:.2f}%')

plt.figure(figsize=(10, 4))
bars = plt.bar(fs_results.keys(), fs_results.values(),
               color=['steelblue', 'coral', 'mediumseagreen', 'orange', 'purple'],
               alpha=0.85, edgecolor='white')
plt.title('Classification Accuracy by Feature Set')
plt.ylabel('Accuracy (%)')
plt.ylim([50, 100])
plt.xticks(rotation=15, ha='right')
for bar, val in zip(bars, fs_results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{val:.2f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(_out_dir, 'feature_set_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Feature set comparison saved')

# 6c. Kruskal-Wallis test -- identify statistically discriminative features
kruskal_results = {}
for feature in X_scaled_df.columns:
    class_0 = X_scaled_df[feature][y == 0]
    class_1 = X_scaled_df[feature][y == 1]
    try:
        _, p_val = stats.kruskal(class_0, class_1)
        kruskal_results[feature] = p_val
    except ValueError:
        # Skip zero-variance features that cause the test to fail
        continue

kruskal_df = pd.DataFrame(
    list(kruskal_results.items()), columns=['Feature', 'P-value']
).sort_values('P-value')

significant = kruskal_df[kruskal_df['P-value'] < 0.05]['Feature'].tolist()
print(f'Significant features (p < 0.05): {len(significant)} / {len(features_df.columns)}')
print(kruskal_df.head(10).to_string(index=False))

# Box plot -- top 4 discriminative features
top4    = kruskal_df.head(4)['Feature'].tolist()
plot_df = X_scaled_df[top4].copy()
plot_df['Class'] = pd.Series(y).map({0: 'Rock', 1: 'Paper'}).values
melted  = plot_df.melt(id_vars='Class', var_name='Feature', value_name='Value')

plt.figure(figsize=(10, 4))
sns.boxplot(x='Feature', y='Value', hue='Class', data=melted,
            palette={'Rock': 'steelblue', 'Paper': 'coral'})
plt.title('Box Plot -- Top 4 Discriminative Features (Kruskal-Wallis)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(_out_dir, 'kruskal_boxplot.png'), dpi=150, bbox_inches='tight')
plt.show()

# 6d. PCA -- retain 95 % of total variance
X_sig = X_scaled_df[significant] if significant else X_scaled_df

pca   = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_sig)

print(f'Original features : {X_sig.shape[1]}')
print(f'PCA components    : {X_pca.shape[1]}  (95% variance retained)')
print(f'Explained variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%')

# 2D PCA scatter plot
pca2d   = PCA(n_components=2, random_state=42)
X_pca2d = pca2d.fit_transform(X_sig)

plt.figure(figsize=(7, 5))
for cls, color, label in [(0, 'steelblue', 'Rock'), (1, 'coral', 'Paper')]:
    mask = y == cls
    plt.scatter(X_pca2d[mask, 0], X_pca2d[mask, 1],
                c=color, label=label, alpha=0.4, s=10)
plt.title('PCA -- 2D Projection of Feature Space')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(_out_dir, 'pca_scatter.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- Step 7: Classification (Hold-out Validation) ----------------------------

# Task 9: Train/test split -- stratified 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Training samples : {X_train.shape[0]}')
print(f'Test samples     : {X_test.shape[0]}')

# Task 10: Initialise classifiers
classifiers = {
    'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)'     : SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost'       : XGBClassifier(n_estimators=100, random_state=42,
                                    eval_metric='logloss', verbosity=0),
}

# Task 11: Train and evaluate (hold-out)
holdout_results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    holdout_results[name] = {'accuracy': acc, 'clf': clf, 'y_pred': y_pred}
    print(f'{name:20s}  Accuracy: {acc * 100:.2f}%')

# --- Step 8: Confusion Matrix & ROC Curves -----------------------------------

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Confusion Matrices & ROC Curves -- Hold-out Validation',
             fontsize=14, fontweight='bold')

class_names = ['Rock', 'Paper']

for col, (name, res) in enumerate(holdout_results.items()):
    clf    = res['clf']
    y_pred = res['y_pred']

    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0, col], colorbar=False, cmap='Blues')
    axes[0, col].set_title(f'{name}\nAcc: {res["accuracy"] * 100:.2f}%')

    # ROC curve
    y_prob      = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    axes[1, col].plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[1, col].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1, col].set_xlim([0, 1])
    axes[1, col].set_ylim([0, 1.02])
    axes[1, col].set_xlabel('False Positive Rate')
    axes[1, col].set_ylabel('True Positive Rate')
    axes[1, col].set_title(f'ROC Curve -- {name}')
    axes[1, col].legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(_out_dir, 'confusion_roc.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Confusion matrices and ROC curves saved')

# --- Step 9: Performance Report ----------------------------------------------

def performance_metrics(y_true, y_pred, clf_name=''):
    cm               = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP   = cm.ravel()
    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision   = TP / (TP + FP)
    f1          = 2 * (precision * sensitivity) / (precision + sensitivity)
    return {
        'Classifier'  : clf_name,
        'Accuracy'    : f'{accuracy    * 100:.2f}%',
        'Sensitivity' : f'{sensitivity * 100:.2f}%',
        'Specificity' : f'{specificity * 100:.2f}%',
        'Precision'   : f'{precision   * 100:.2f}%',
        'F1 Score'    : f'{f1          * 100:.2f}%',
    }


print('=== HOLD-OUT PERFORMANCE REPORT ===')
report_rows = [performance_metrics(y_test, res['y_pred'], name)
               for name, res in holdout_results.items()]
report_df = pd.DataFrame(report_rows).set_index('Classifier')
print(report_df.to_string())

best_clf_name = max(holdout_results, key=lambda k: holdout_results[k]['accuracy'])
print(f'\nDetailed Report -- Best Model: {best_clf_name}')
print(classification_report(
    y_test, holdout_results[best_clf_name]['y_pred'],
    target_names=['Rock', 'Paper']
))

# --- Step 10: 5-Fold Cross-Validation ----------------------------------------

cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

print('=== 5-FOLD CROSS-VALIDATION ===')
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_pca, y, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f'{name:20s}  Mean: {scores.mean() * 100:.2f}% +/- {scores.std() * 100:.2f}%')

plt.figure(figsize=(8, 4))
plt.boxplot(
    [cv_results[n] * 100 for n in cv_results],
    labels=list(cv_results.keys()),
    patch_artist=True,
    boxprops=dict(facecolor='lightsteelblue')
)
plt.title('5-Fold Cross-Validation Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim([50, 105])
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(_out_dir, 'crossval.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- Step 11: GridSearchCV (Hyperparameter Tuning -- All Classifiers) ---------

cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest
print('Running GridSearchCV on Random Forest...')
rf_param_grid = {
    'n_estimators'      : [50, 100, 200],
    'max_depth'         : [None, 5, 10],
    'min_samples_split' : [2, 5],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=cv_inner, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
print(f'RF  best params : {grid_search.best_params_}')
print(f'RF  best CV acc : {grid_search.best_score_ * 100:.2f}%')
print(f'RF  test acc    : {accuracy_score(y_test, grid_search.best_estimator_.predict(X_test)) * 100:.2f}%')

# SVM (RBF)
print('\nRunning GridSearchCV on SVM (RBF)...')
svm_param_grid = {
    'C'     : [0.1, 1, 10, 100],
    'gamma' : ['scale', 0.01, 0.1],
}
gs_svm = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    svm_param_grid, cv=cv_inner, scoring='accuracy', n_jobs=-1, verbose=1
)
gs_svm.fit(X_train, y_train)
print(f'SVM best params : {gs_svm.best_params_}')
print(f'SVM best CV acc : {gs_svm.best_score_ * 100:.2f}%')
print(f'SVM test acc    : {accuracy_score(y_test, gs_svm.best_estimator_.predict(X_test)) * 100:.2f}%')

# XGBoost
print('\nRunning GridSearchCV on XGBoost...')
xgb_param_grid = {
    'n_estimators'  : [50, 100, 200],
    'max_depth'     : [3, 6, 10],
    'learning_rate' : [0.1, 0.3],
}
gs_xgb = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    xgb_param_grid, cv=cv_inner, scoring='accuracy', n_jobs=-1, verbose=1
)
gs_xgb.fit(X_train, y_train)
print(f'XGB best params : {gs_xgb.best_params_}')
print(f'XGB best CV acc : {gs_xgb.best_score_ * 100:.2f}%')
print(f'XGB test acc    : {accuracy_score(y_test, gs_xgb.best_estimator_.predict(X_test)) * 100:.2f}%')

# Collect all grid search results
grid_searches = {
    'Random Forest' : grid_search,
    'SVM (RBF)'     : gs_svm,
    'XGBoost'       : gs_xgb,
}
grid_preds = {name: gs.best_estimator_.predict(X_test)
              for name, gs in grid_searches.items()}

# --- Step 12: Final Comparison: Hold-out vs CV vs GridSearchCV ---------------

comparison = []
for name in classifiers:
    holdout_acc = holdout_results[name]['accuracy'] * 100
    cv_mean     = cv_results[name].mean() * 100
    cv_std      = cv_results[name].std()  * 100
    grid_acc    = accuracy_score(y_test, grid_preds[name]) * 100
    comparison.append({
        'Classifier'         : name,
        'Hold-out Acc (%)'   : f'{holdout_acc:.2f}',
        'CV Mean Acc (%)'    : f'{cv_mean:.2f}',
        'CV Std (%)'         : f'{cv_std:.2f}',
        'GridSearch Acc (%)' : f'{grid_acc:.2f}',
    })

comparison_df = pd.DataFrame(comparison).set_index('Classifier')
print('=== FINAL VALIDATION COMPARISON ===')
print(comparison_df.to_string())

# Bar chart comparison
x     = np.arange(len(classifiers))
width = 0.25
names = list(classifiers.keys())

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, [holdout_results[n]['accuracy'] * 100 for n in names],
       width, label='Hold-out', color='steelblue', alpha=0.85)
ax.bar(x,          [cv_results[n].mean() * 100 for n in names],
       width, label='CV Mean',  color='coral', alpha=0.85)
ax.bar(x + width,  [accuracy_score(y_test, grid_preds[n]) * 100 for n in names],
       width, label='GridSearchCV', color='mediumseagreen', alpha=0.85)

ax.set_xlabel('Classifier')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Validation Method Comparison')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim([50, 105])
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(_out_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Validation comparison saved')

# --- Step 12b: Feature Importance Analysis -----------------------------------
# Fit a separate Random Forest on the original (non-PCA) Kruskal-Wallis
# selected features so that importances map back to interpretable feature names.

rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
X_imp_train, _, y_imp_train, _ = train_test_split(
    X_sig, y, test_size=0.2, random_state=42, stratify=y
)
rf_importance.fit(X_imp_train, y_imp_train)

importances = rf_importance.feature_importances_
feat_names  = X_sig.columns.tolist()
sorted_idx  = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)),
        importances[sorted_idx],
        color='steelblue', alpha=0.85, edgecolor='white')
plt.xticks(range(len(importances)),
           [feat_names[i] for i in sorted_idx],
           rotation=45, ha='right', fontsize=9)
plt.title('Random Forest Feature Importance\n(Kruskal-Wallis selected features, 100 trees)')
plt.xlabel('Feature')
plt.ylabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig(os.path.join(_out_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Feature importance plot saved')

# Top 5 most important features
print('\nTop 5 most important features:')
for rank, idx in enumerate(sorted_idx[:5], 1):
    print(f'  {rank}. {feat_names[idx]:25s}  importance: {importances[idx]:.4f}')

# --- Step 12c: GridSearchCV Heatmap ------------------------------------------
# Show accuracy across n_estimators x max_depth for min_samples_split = 2
# (the best min_samples_split value found by the grid search).

gs_results = pd.DataFrame(grid_search.cv_results_)
gs_sub     = gs_results[gs_results['param_min_samples_split'] == 2].copy()

# Replace None with the string 'None' so it displays cleanly on the heatmap axis
gs_sub['param_max_depth'] = gs_sub['param_max_depth'].apply(
    lambda v: 'None' if v is None else int(v)
)

pivot = gs_sub.pivot_table(
    index='param_max_depth',
    columns='param_n_estimators',
    values='mean_test_score'
)

plt.figure(figsize=(7, 4))
sns.heatmap(pivot * 100, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': 'CV Accuracy (%)'})
plt.title('GridSearchCV Accuracy Heatmap\n(Random Forest, min_samples_split=2, 5-fold CV)')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.tight_layout()
plt.savefig(os.path.join(_out_dir, 'gridsearch_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()
print('GridSearchCV heatmap saved')
print('\nAll plots saved. Pipeline complete.')

# --- Step 13: Results Interpretation -----------------------------------------
#
# Dataset Summary
# ---------------
# Classes         : Rock (0) vs Paper (1)
# Total trials    : 5,850  (2,909 Rock + 2,941 Paper)
# Samples/trial   : 1,024  (~4 s at 256 Hz)
# Channels        : 1 (single-channel EMG)
# Missing values  : None
# Filtering       : Not applied (signals already pre-filtered)
#
# Feature Extraction Summary (26 features per trial)
# ---------------------------------------------------
# Statistical (8) : Mean, Median, Std Dev, Variance, Skewness, Kurtosis, Range, IQR
# Time-domain (7) : RMS, Zero Crossings, Autocorrelation, Mean Abs Deviation,
#                   Max, Min, Signal Energy
# Frequency  (6)  : Dominant Freq, Total Power, Band Power 0.5-40 Hz,
#                   Mean Freq, Median Freq, Freq Variance
# Entropy    (5)  : Sample Entropy, Spectral Entropy, Permutation Entropy,
#                   SVD Entropy, LZiv Complexity
#
# Feature Selection
# -----------------
# Kruskal-Wallis test identified the most statistically significant features
# (p < 0.05) distinguishing Rock from Paper. PCA was then applied on the
# selected features, retaining components that explain 95% of the variance.
#
# Classifier Comparison (Hold-out)
# ---------------------------------
# Random Forest : 78.38%  |  SVM (RBF) : 78.97%  |  XGBoost : 75.90%
# Best model    : SVM (RBF) -- highest accuracy, F1 (80.63%), and AUC (0.875)
#
# Validation Method Comparison
# ----------------------------
# Hold-out (80/20)        : fast, single-split estimate
# 5-Fold Cross-Validation : more reliable generalisation estimate
# GridSearchCV            : applied to all three classifiers for consistency
#   RF  (18 combos, 90 fits)  | SVM (12 combos, 60 fits) | XGB (18 combos, 90 fits)
#
# Limitations
# -----------
# - Single EMG channel; multi-channel would likely improve accuracy.
# - Binary classification only (Rock vs Paper).
# - Subject-independent generalisation not evaluated (LOSO-CV recommended).
