import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained LSTM model
model = tf.keras.models.load_model('lstm_model.h5')

# Load the data
train_data = pd.read_csv('PM_train.txt', delimiter=' ', header=None)
truth_data = pd.read_csv('PM_truth.txt', header=None)

# Data preprocessing
train_data.dropna(axis=1, inplace=True)  # Remove extra spaces or NaN columns
train_data.columns = ['UnitNumber', 'TimeInCycles'] + [f'OpSet{i}' for i in range(1, 4)] + \
                     [f'Sensor{i}' for i in range(1, 22)]

# Check if the number of features matches what the model expects
expected_features = 24  # This should match your model's expected number of features
actual_features = len(train_data.columns) - 2  # Subtracting 'UnitNumber' and 'TimeInCycles'

if actual_features != expected_features:
    st.error(f"Feature mismatch: Model expects {expected_features} features, but found {actual_features} features.")
    st.stop()  # Stop execution if there's a mismatch

# Normalize the sensor data
scaler = StandardScaler()
sensor_columns = [col for col in train_data.columns if 'Sensor' in col]
train_data[sensor_columns] = scaler.fit_transform(train_data[sensor_columns])

# Streamlit Dashboard
st.title("Equipment Health Monitoring and RUL Prediction Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select an option", ["Data Overview", "Predict RUL", "Maintenance Recommendations"])

if options == "Data Overview":
    st.header("Data Overview")
    st.write("Here is a preview of the training data:")
    st.dataframe(train_data.head())

    st.write("Summary statistics of the data:")
    st.write(train_data.describe())

elif options == "Predict RUL":
    st.header("Remaining Useful Life Prediction")
    unit_number = st.number_input("Enter Unit Number", min_value=1, max_value=int(train_data['UnitNumber'].max()), value=1)

    # Filter data for the selected unit
    unit_data = train_data[train_data['UnitNumber'] == unit_number]
    if unit_data.empty:
        st.warning("No data found for this unit number.")
    else:
        # Prepare the data for the LSTM model
        scaled_unit_data = unit_data[sensor_columns].values
        timesteps = 50  # Adjust this to your model's expected timesteps
        features = scaled_unit_data.shape[1]

        if scaled_unit_data.shape[0] < timesteps:
            padding = np.zeros((timesteps - scaled_unit_data.shape[0], features))
            scaled_unit_data = np.vstack((padding, scaled_unit_data))
        else:
            scaled_unit_data = scaled_unit_data[-timesteps:]

        input_data = np.expand_dims(scaled_unit_data, axis=0)
        
        st.write("Input data shape for the model:", input_data.shape)

        try:
            prediction = model.predict(input_data)
            st.write(f"Predicted RUL for Unit {unit_number}: {int(prediction[0][0])} cycles")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

elif options == "Maintenance Recommendations":
    st.header("Maintenance Recommendations")
    st.write("Based on the current health status, here are some recommendations:")
    st.write("- Perform detailed diagnostics on units with less than 30 cycles RUL.")
    st.write("- Schedule preventive maintenance to avoid unexpected failures.")
    st.write("- Monitor sensor anomalies that deviate significantly from normal ranges.")
