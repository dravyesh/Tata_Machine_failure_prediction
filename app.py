import streamlit as st
import pandas as pd
import joblib

threshold = joblib.load("models/threshold.pkl")


# Load the trained model, scaler and feature list
model = joblib.load("models/machine_failure_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")


# Configure the Streamlit page
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)


# Display the application title
st.title("⚙️ Machine Failure Prediction")

# Display a short description
st.write(
    "Enter the machine operating parameters below to predict "
    "whether a machine failure is likely to occur."
)


# Create input fields for machine parameters
machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=295.0,
    max_value=305.0,
    value=300.0
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=305.0,
    max_value=315.0,
    value=310.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=1000,
    max_value=3000,
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=80.0,
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100
)


# Create the prediction button
if st.button("Predict Machine Failure"):

    # Convert machine type into the numerical value used during training
    type_mapping = {"L": 0, "M": 1, "H": 2}
    type_value = type_mapping[machine_type]

    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame({
        "Type": [type_value],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # Create the same engineered features used during model training
    input_data["Temp Difference"] = (
        input_data["Process temperature [K]"]
        - input_data["Air temperature [K]"]
    )

    input_data["Power"] = (
        input_data["Torque [Nm]"]
        * input_data["Rotational speed [rpm]"]
    )

    input_data["Wear per Speed"] = (
        input_data["Tool wear [min]"]
        / (input_data["Rotational speed [rpm]"] + 1e-6)
    )

    # Arrange the columns in the same order used during training
    input_data = input_data[features]

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Calculate the probability of machine failure
    failure_probability = model.predict_proba(input_scaled)[0][1]

    # Apply the selected classification threshold
    prediction = int(failure_probability >= threshold)

    # Display the prediction result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted")
    else:
        st.success("✅ No Machine Failure Predicted")

    # Display the probability
    st.write(
        f"Estimated Failure Probability: **{failure_probability:.2%}**"
    )