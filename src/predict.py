import pandas as pd
import joblib


# Load the saved model
model = joblib.load("models/machine_failure_model.pkl")

# Load the saved scaler
scaler = joblib.load("models/scaler.pkl")

# Load the feature names used during training
features = joblib.load("models/features.pkl")


# Create sample machine data
sample_data = pd.DataFrame({
    "Type": [1],
    "Air temperature [K]": [300.0],
    "Process temperature [K]": [310.0],
    "Rotational speed [rpm]": [1500],
    "Torque [Nm]": [40.0],
    "Tool wear [min]": [100]
})


# Create the same engineered features used during training
sample_data["Temp Difference"] = (
    sample_data["Process temperature [K]"]
    - sample_data["Air temperature [K]"]
)

sample_data["Power"] = (
    sample_data["Torque [Nm]"]
    * sample_data["Rotational speed [rpm]"]
)

sample_data["Wear per Speed"] = (
    sample_data["Tool wear [min]"]
    / (sample_data["Rotational speed [rpm]"] + 1e-6)
)


# Arrange the columns in exactly the same order as training
sample_data = sample_data[features]

# Scale the input data using the saved scaler
sample_scaled = scaler.transform(sample_data)

# Generate the prediction
prediction = model.predict(sample_scaled)[0]

# Generate the probability of machine failure
failure_probability = model.predict_proba(sample_scaled)[0][1]


# Display the prediction
print("\n========== MACHINE FAILURE PREDICTION ==========")

if prediction == 1:
    print("Prediction: Machine Failure")
else:
    print("Prediction: No Machine Failure")

print(f"Failure Probability: {failure_probability:.2%}")