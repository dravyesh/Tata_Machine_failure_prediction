import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# Load the training dataset
train = pd.read_csv("data/train.csv")

# Remove ID columns
train = train.drop(columns=["id", "Product ID"], errors="ignore")

# Convert machine type into numerical values
type_mapping = {"L": 0, "M": 1, "H": 2}
train["Type"] = train["Type"].map(type_mapping)

# Create the same engineered features used during training
train["Temp Difference"] = (
    train["Process temperature [K]"]
    - train["Air temperature [K]"]
)

train["Power"] = (
    train["Torque [Nm]"]
    * train["Rotational speed [rpm]"]
)

train["Wear per Speed"] = (
    train["Tool wear [min]"]
    / (train["Rotational speed [rpm]"] + 1e-6)
)


# Separate features and target
X = train.drop(columns=["Machine failure"])

# Remove leakage columns
leakage_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
X = X.drop(columns=leakage_columns, errors="ignore")

y = train["Machine failure"]


# Split the data exactly as used during model development
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Load the saved scaler
scaler = joblib.load("models/scaler.pkl")

# Transform validation data using the saved scaler
X_val_scaled = scaler.transform(X_val)


# Load the final trained model
model = joblib.load("models/machine_failure_model.pkl")


# Generate predictions
predictions = model.predict(X_val_scaled)


# Calculate evaluation metrics
accuracy = accuracy_score(y_val, predictions)
precision = precision_score(y_val, predictions, zero_division=0)
recall = recall_score(y_val, predictions, zero_division=0)
f1 = f1_score(y_val, predictions, zero_division=0)


# Display the evaluation results
print("\n========== FINAL MODEL EVALUATION ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, predictions, zero_division=0))


# Create confusion matrix
cm = confusion_matrix(y_val, predictions)

print("\nConfusion Matrix:")
print(cm)


# Plot confusion matrix
plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Failure", "Failure"]
)

disp.plot()

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

# Save the confusion matrix image
plt.savefig("models/confusion_matrix.png", dpi=300)

plt.show()


# Create a metric comparison chart
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))

plt.bar(metrics, values)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Final Random Forest Model Performance")

# Display values above each bar
for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()

# Save the metric chart
plt.savefig("models/model_performance.png", dpi=300)

plt.show()


# Save evaluation metrics as a CSV file
evaluation_df = pd.DataFrame({
    "Metric": metrics,
    "Score": values
})

evaluation_df.to_csv(
    "models/evaluation_metrics.csv",
    index=False
)

print("\nEvaluation files saved successfully.")