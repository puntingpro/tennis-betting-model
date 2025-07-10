import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)

# ===================================================================
# SETUP: Load the backtest results file
# ===================================================================
RESULTS_PATH = 'data/analysis/backtest_results.csv'

print(f"Loading results from {RESULTS_PATH}...")
df = pd.read_csv(RESULTS_PATH)
print("✅ Results loaded successfully.\n")


# ===================================================================
# PREPARATION: Assign columns to variables
# ===================================================================
# We assume 'winner' holds the true outcome (1 for win, 0 for loss)
# and 'predicted_prob' is the model's probability of that win.

# y_true: The actual, true outcomes from your data.
y_true = df['winner']

# y_pred_proba: The model's predicted probabilities for each match.
y_pred_proba = df['predicted_prob']

# y_pred: The model's binary prediction (1 if probability > 0.5, else 0).
y_pred = (y_pred_proba > 0.5).astype(int)


# ===================================================================
# EVALUATION: Calculate and display performance metrics
# ===================================================================

# 1. Print the classification report (Accuracy, Precision, Recall)
print("--- Classification Report ---")
print(classification_report(y_true, y_pred))

# 2. Display the confusion matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

# 3. Calculate AUC and plot the ROC Curve
auc_score = roc_auc_score(y_true, y_pred_proba)
print(f"\n--- AUC Score --- \n{auc_score:.4f}")

# Plot the curve using the true values and predicted probabilities
RocCurveDisplay.from_predictions(y_true, y_pred_proba)
plt.title('ROC Curve from Backtest Results')
plt.grid(True)
plt.show()