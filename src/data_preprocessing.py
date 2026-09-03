import pandas as pd

# Load the training and testing datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Display the original dataset shape
print("Original Train Shape:", train.shape)
print("Original Test Shape:", test.shape)

# Remove ID columns because they do not provide useful information for prediction
train = train.drop(columns=["id", "Product ID"], errors="ignore")
test = test.drop(columns=["id", "Product ID"], errors="ignore")

# Convert machine type from categorical values into numerical values
type_mapping = {"L": 0, "M": 1, "H": 2}
train["Type"] = train["Type"].map(type_mapping)
test["Type"] = test["Type"].map(type_mapping)

# Check missing values after conversion
print("\nMissing values in Train:")
print(train.isnull().sum())

print("\nMissing values in Test:")
print(test.isnull().sum())

# Display the cleaned dataset shape
print("\nCleaned Train Shape:", train.shape)
print("Cleaned Test Shape:", test.shape)