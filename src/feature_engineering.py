import pandas as pd

# Load the cleaned datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Remove ID columns because they are not useful for prediction
train = train.drop(columns=["id", "Product ID"], errors="ignore")
test = test.drop(columns=["id", "Product ID"], errors="ignore")

# Convert machine type into numerical values
type_mapping = {"L": 0, "M": 1, "H": 2}
train["Type"] = train["Type"].map(type_mapping)
test["Type"] = test["Type"].map(type_mapping)

# Create temperature difference to capture the thermal gap between process and air temperature
train["Temp Difference"] = train["Process temperature [K]"] - train["Air temperature [K]"]
test["Temp Difference"] = test["Process temperature [K]"] - test["Air temperature [K]"]

# Create power feature to represent the approximate operating load of the machine
train["Power"] = train["Torque [Nm]"] * train["Rotational speed [rpm]"]
test["Power"] = test["Torque [Nm]"] * test["Rotational speed [rpm]"]

# Create wear per speed to represent tool wear relative to machine speed
train["Wear per Speed"] = train["Tool wear [min]"] / (train["Rotational speed [rpm]"] + 1e-6)
test["Wear per Speed"] = test["Tool wear [min]"] / (test["Rotational speed [rpm]"] + 1e-6)

# Display the new features
print("New Features Created:")
print("- Temp Difference")
print("- Power")
print("- Wear per Speed")

# Display the updated dataset shapes
print("\nTrain Shape:", train.shape)
print("Test Shape:", test.shape)

# Display the first five rows of the engineered features
print("\nFeature Engineering Preview:")
print(train[[
    "Temp Difference",
    "Power",
    "Wear per Speed"
]].head())