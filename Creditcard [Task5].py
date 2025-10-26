import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CREDIT CARD FRAUD DETECTION - CODSOFT TASK 5")
print("="*70)

print("\n[Step 1] Loading dataset...")
try:
    df = pd.read_csv("creditcard.csv")
    print(f" Dataset loaded successfully!")
    print(f"  Shape: {df.shape}")
except FileNotFoundError:
    print(" Error: creditcard.csv not found!")
    print("  Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    exit()

print("\n[Step 2] Dataset Overview")
print("-"*70)
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum().sum(), "missing values found")

print("\n[Step 3] Class Distribution Analysis")
print("-"*70)
class_dist = df['Class'].value_counts()
print(class_dist)
print(f"\nFraud Percentage: {(class_dist[1]/len(df))*100:.4f}%")
print(f"Normal Percentage: {(class_dist[0]/len(df))*100:.4f}%")

plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette=['green', 'red'])
plt.title('Class Distribution (0: Normal, 1: Fraud)', fontsize=14, fontweight='bold')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print("\n Class distribution plot saved as 'class_distribution.png'")

print("\n[Step 4] Data Preprocessing")
print("-"*70)

X = df.drop('Class', axis=1)
y = df['Class']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("\n[Step 5] Feature Normalization")
print("-"*70)

scaler = StandardScaler()

if 'Amount' in X.columns:
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    print(" Amount column normalized")

if 'Time' in X.columns:
    X['Time'] = scaler.fit_transform(X[['Time']])
    print(" Time column normalized")

print(" Feature normalization complete")

print("\n[Step 6] Splitting Dataset")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Fraud cases in training: {y_train.sum()}")
print(f"Fraud cases in test: {y_test.sum()}")

print("\n[Step 7] Handling Class Imbalance with SMOTE")
print("-"*70)

print("Before SMOTE:")
print(f"  Normal transactions: {(y_train == 0).sum()}")
print(f"  Fraud transactions: {(y_train == 1).sum()}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(f"  Normal transactions: {(y_train_balanced == 0).sum()}")
print(f"  Fraud transactions: {(y_train_balanced == 1).sum()}")
print(" Dataset balanced successfully")

print("\n[Step 8] Training Classification Models")
print("="*70)

print("\n[Model 1] Logistic Regression")
print("-"*70)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_balanced, y_train_balanced)
lr_pred = lr_model.predict(X_test)
print(" Logistic Regression trained successfully")

print("\n[Model 2] Random Forest Classifier")
print("-"*70)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test)
print(" Random Forest trained successfully")

print("\n[Step 9] Model Evaluation")
print("="*70)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print("-"*70)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (How many predicted frauds are actually frauds)")
    print(f"Recall:    {recall:.4f} (How many actual frauds were detected)")
    print(f"F1-Score:  {f1:.4f} (Harmonic mean of precision and recall)")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(f"True Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    
    return accuracy, precision, recall, f1, roc_auc, cm

lr_metrics = evaluate_model(y_test, lr_pred, "Logistic Regression")
rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")

print("\n[Step 10] Generating Visualizations")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.heatmap(lr_metrics[5], annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
axes[0, 0].set_title('Confusion Matrix - Logistic Regression', fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')
sns.heatmap(rf_metrics[5], annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
axes[0, 1].set_title('Confusion Matrix - Random Forest', fontweight='bold')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xlabel('Predicted')
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
lr_scores = list(lr_metrics[:5])
rf_scores = list(rf_metrics[:5])
x_pos = np.arange(len(metrics_names))
width = 0.35

axes[1, 0].bar(x_pos - width/2, lr_scores, width, label='Logistic Regression', color='skyblue')
axes[1, 0].bar(x_pos + width/2, rf_scores, width, label='Random Forest', color='lightgreen')
axes[1, 0].set_xlabel('Metrics')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Model Comparison', fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(metrics_names, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].grid(axis='y', alpha=0.3)

if hasattr(rf_model, 'feature_importances_'):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-10:] 
    
    axes[1, 1].barh(range(len(indices)), importances[indices], color='coral')
    axes[1, 1].set_yticks(range(len(indices)))
    axes[1, 1].set_yticklabels([X.columns[i] for i in indices])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Important Features (Random Forest)', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print(" Model evaluation plots saved as 'model_evaluation.png'")

print("\n[Step 11] Final Results & Recommendations")
print("="*70)

best_model = "Random Forest" if rf_metrics[3] > lr_metrics[3] else "Logistic Regression"
best_f1 = max(rf_metrics[3], lr_metrics[3])

print(f"\n Best Model: {best_model}")
print(f"   Best F1-Score: {best_f1:.4f}")

print("\n Key Insights:")
print(f"   • Dataset had {(class_dist[1]/len(df))*100:.4f}% fraud cases (highly imbalanced)")
print(f"   • SMOTE was used to balance the training data")
print(f"   • Both models achieved good performance after handling imbalance")

print("\n Recommendations:")
print("   • Use the model with highest Recall for fraud detection (minimize false negatives)")
print("   • Consider ensemble methods for production deployment")
print("   • Regularly retrain model with new fraud patterns")
print("   • Implement real-time monitoring for fraud detection")

print("\n Analysis Complete!")
print("="*70)