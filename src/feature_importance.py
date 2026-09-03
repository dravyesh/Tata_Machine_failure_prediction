import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load the trained final model
model = joblib.load("models/machine_failure_model.pkl")

# Load the final feature names
features = joblib.load("models/features.pkl")

# Get feature importance values from the Random Forest model
importance = model.feature_importances_

# Create a DataFrame containing feature names and importance values
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort features from highest to lowest importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display feature importance values
print("Feature Importance:")
print(importance_df.to_string(index=False))


# Create a horizontal bar chart
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

# Display the most important features at the top
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()