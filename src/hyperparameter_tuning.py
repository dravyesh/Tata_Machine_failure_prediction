import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the training dataset
train = pd.read_csv("data/train.csv")

# Remove ID columns
train = train.drop(columns=["id", "Product ID"], errors="ignore")

# Convert machine type into numerical values
type_mapping = {"L": 0, "M": 1, "H": 2}
train["Type"] = train["Type"].map(type_mapping)

# Create temperature difference feature
train["Temp Difference"] = train["Process temperature [K]"] - train["Air temperature [K]"]

# Create power feature
train["Power"] = train["Torque [Nm]"] * train["Rotational speed [rpm]"]

# Create wear per speed feature
train["Wear per Speed"] = train["Tool wear [min]"] / (train["Rotational speed [rpm]"] + 1e-6)


# Separate features and target
X = train.drop(columns=["Machine failure"])

# Remove leakage columns
leakage_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
X = X.drop(columns=leakage_columns, errors="ignore")

y = train["Machine failure"]


# Split the dataset while preserving the failure ratio
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Scale the features using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Define Random Forest parameter combinations
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"]
}


# Create Random Forest
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# Perform GridSearchCV using F1 score
rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# Train Random Forest GridSearch
rf_grid.fit(X_train_scaled, y_train)

# Get the best Random Forest model
best_rf = rf_grid.best_estimator_

print("\n========== BEST RANDOM FOREST ==========")
print("Best Parameters:", rf_grid.best_params_)

rf_pred = best_rf.predict(X_val_scaled)

print("Accuracy:", accuracy_score(y_val, rf_pred))
print(classification_report(y_val, rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, rf_pred))


# Define Gradient Boosting parameter combinations
gb_params = {
    "n_estimators": [100, 150],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
    "subsample": [0.8, 1.0]
}


# Create Gradient Boosting
gb = GradientBoostingClassifier(
    random_state=42
)


# Perform GridSearchCV using F1 score
gb_grid = GridSearchCV(
    estimator=gb,
    param_grid=gb_params,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# Train Gradient Boosting GridSearch
gb_grid.fit(X_train_scaled, y_train)

# Get the best Gradient Boosting model
best_gb = gb_grid.best_estimator_

print("\n========== BEST GRADIENT BOOSTING ==========")
print("Best Parameters:", gb_grid.best_params_)

gb_pred = best_gb.predict(X_val_scaled)

print("Accuracy:", accuracy_score(y_val, gb_pred))
print(classification_report(y_val, gb_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, gb_pred))


# Compare the F1 scores of both tuned models
rf_f1 = rf_grid.best_score_
gb_f1 = gb_grid.best_score_

print("\n========== MODEL COMPARISON ==========")
print("Random Forest CV F1:", rf_f1)
print("Gradient Boosting CV F1:", gb_f1)


# Select the model with the better cross-validation F1 score
if rf_f1 >= gb_f1:
    final_model = best_rf
    final_model_name = "Random Forest"
else:
    final_model = best_gb
    final_model_name = "Gradient Boosting"


# Save the selected final model
joblib.dump(final_model, "models/machine_failure_model.pkl")

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save the feature names
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("\nFinal Model:", final_model_name)
print("Final model saved successfully.")