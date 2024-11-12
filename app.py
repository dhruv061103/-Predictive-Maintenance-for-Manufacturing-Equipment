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

# Basic preprocessing
train_data.dropna(axis=1, inplace=True)  # Remove extra spaces or NaN columns
train_data.columns = ['UnitNumber', 'TimeInCycles'] + [f'OpSet{i}' for i in range(1, 4)] + \
                     [f'Sensor{i}' for i in range(1, 22)]

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_data.iloc[:, 2:])  # Scaling the sensor data only

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
    unit_number = st.number_input("Enter Unit Number", min_value=1, max_value=len(truth_data), value=1)
    
    # Prepare data for the selected unit
    unit_data = train_data[train_data['UnitNumber'] == unit_number]
    if unit_data.empty:
        st.warning("No data found for this unit number.")
    else:
        scaled_unit_data = scaler.transform(unit_data.iloc[:, 2:])
        # Reshape data for LSTM model (samples, timesteps, features)
        input_data = np.expand_dims(scaled_unit_data, axis=0)
        prediction = model.predict(input_data)
        st.write(f"Predicted RUL for Unit {unit_number}: {int(prediction[0][0])} cycles")

elif options == "Maintenance Recommendations":
    st.header("Maintenance Recommendations")
    st.write("Based on the current health status, here are some recommendations:")
    st.write("- Perform detailed diagnostics on units with less than 30 cycles RUL.")
    st.write("- Schedule preventive maintenance to avoid unexpected failures.")
    st.write("- Monitor sensor anomalies that deviate significantly from normal ranges.")
