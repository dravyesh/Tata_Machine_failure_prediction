import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the training dataset
train = pd.read_csv("data/train.csv")

# Remove columns that are not useful for prediction
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


# Separate input features and target variable
X = train.drop(columns=["Machine failure"])

# Remove failure-type columns because they directly reveal the failure condition
leakage_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
X = X.drop(columns=leakage_columns, errors="ignore")

# Store the target variable
y = train["Machine failure"]

# Display the final feature list
print("Final Features:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)


# Scale the numerical features using only the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Create the Logistic Regression model with balanced class weights
logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train Logistic Regression
logistic_model.fit(X_train_scaled, y_train)

# Generate validation predictions
logistic_pred = logistic_model.predict(X_val_scaled)

print("\n========== Logistic Regression ==========")
print("Accuracy:", accuracy_score(y_val, logistic_pred))
print(classification_report(y_val, logistic_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, logistic_pred))


# Create the Random Forest model with balanced class weights
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train Random Forest
random_forest_model.fit(X_train_scaled, y_train)

# Generate validation predictions
random_forest_pred = random_forest_model.predict(X_val_scaled)

print("\n========== Random Forest ==========")
print("Accuracy:", accuracy_score(y_val, random_forest_pred))
print(classification_report(y_val, random_forest_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, random_forest_pred))


# Create the Gradient Boosting model
gradient_boosting_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train Gradient Boosting
gradient_boosting_model.fit(X_train_scaled, y_train)

# Generate validation predictions
gradient_boosting_pred = gradient_boosting_model.predict(X_val_scaled)

print("\n========== Gradient Boosting ==========")
print("Accuracy:", accuracy_score(y_val, gradient_boosting_pred))
print(classification_report(y_val, gradient_boosting_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, gradient_boosting_pred))


# Save the Gradient Boosting model and scaler for later prediction
joblib.dump(gradient_boosting_model, "models/machine_failure_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save the final feature names so prediction uses the same feature order
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("\nModel saved successfully.")
print("Scaler saved successfully.")
print("Feature list saved successfully.")