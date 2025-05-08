import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Load models
nb_model = joblib.load("data/naive_bayes_model.pkl")
dt_model = joblib.load("data/decision_tree_model.pkl")
lr_model = joblib.load("data/logistic_regression_model.pkl")
rf_model = joblib.load("data/random_forest_model.pkl")

models = {
    "Naive Bayes": nb_model,
    "Decision Tree": dt_model,
    "Logistic Regression": lr_model,
    "Random Forest": rf_model
}

# Store metrics and confusion matrices
metrics_dict = {}
confusion_matrices = {}

# Evaluation function
def evaluate(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} Evaluation ---")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)

    metrics_dict[model_name] = [acc, prec, rec, f1]
    confusion_matrices[model_name] = cm

# Predict and evaluate each model
for name, model in models.items():
    preds = model.predict(X_test)
    evaluate(y_test, preds, name)

# Plotting evaluation metrics
def plot_metrics(metrics):
    df = pd.DataFrame(metrics, index=["Accuracy", "Precision", "Recall", "F1 Score"]).T
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.show()

# Plotting confusion matrices
def plot_confusion_matrices(matrices):
    for model_name, cm in matrices.items():
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

# Plotting ROC curves
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# Plotting feature importances
def plot_feature_importance(model, X_test, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:10]  # Top 10 features
        features = X_test.columns[indices]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=features)
        plt.title(f'{model_name} - Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

def plot_summary_table(metrics):
    df = pd.DataFrame(metrics, index=["Accuracy", "Precision", "Recall", "F1 Score"]).T * 100
    df = df.round(2)

    plt.figure(figsize=(8, 4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
    plt.title("Model Performance Summary (in %)")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Execute all plots
plot_metrics(metrics_dict)
plot_confusion_matrices(confusion_matrices)
plot_roc_curves(models, X_test, y_test)
plot_feature_importance(dt_model, X_test, "Decision Tree")
plot_feature_importance(rf_model, X_test, "Random Forest")
plot_summary_table(metrics_dict)
