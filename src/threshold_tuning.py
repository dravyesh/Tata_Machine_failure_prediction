import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


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


# Split the data using stratification
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Create the tuned Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train_scaled, y_train)


# Get failure probabilities instead of direct predictions
failure_probability = model.predict_proba(X_val_scaled)[:, 1]


# Test multiple classification thresholds
thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]

results = []

for threshold in thresholds:

    # Convert probabilities into binary predictions
    predictions = (failure_probability >= threshold).astype(int)

    # Calculate evaluation metrics
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    # Store the results
    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })


# Create a results DataFrame
results_df = pd.DataFrame(results)

# Display threshold comparison
print("\n========== THRESHOLD COMPARISON ==========")
print(results_df.to_string(index=False))


# Select the threshold with the highest F1-score
best_result = results_df.loc[
    results_df["F1-Score"].idxmax()
]

best_threshold = float(best_result["Threshold"])

print("\n========== BEST THRESHOLD ==========")
print(f"Best Threshold: {best_threshold:.2f}")
print(f"Precision: {best_result['Precision']:.2f}")
print(f"Recall: {best_result['Recall']:.2f}")
print(f"F1-Score: {best_result['F1-Score']:.2f}")


# Save the selected threshold
joblib.dump(best_threshold, "models/threshold.pkl")

print("\nThreshold saved successfully.")