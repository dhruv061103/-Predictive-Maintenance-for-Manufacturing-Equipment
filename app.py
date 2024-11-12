import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained LSTM model
model = tf.keras.models.load_model('lstm_model.h5')

# Load the data
train_data = pd.read_csv('PM_train.txt', delimiter=' ', header=None)
test_data = pd.read_csv('PM_test.txt', delimiter=' ', header=None)  # Adjust this file name if needed
truth_data = pd.read_csv('PM_truth.txt', header=None)

# Data preprocessing
train_data.dropna(axis=1, inplace=True)
test_data.dropna(axis=1, inplace=True)

# Naming columns for easier reference
train_data.columns = ['UnitNumber', 'TimeInCycles'] + [f'OpSet{i}' for i in range(1, 4)] + \
                     [f'Sensor{i}' for i in range(1, 22)]
test_data.columns = ['UnitNumber', 'TimeInCycles'] + [f'OpSet{i}' for i in range(1, 4)] + \
                    [f'Sensor{i}' for i in range(1, 22)]

# Normalize the sensor data
scaler = StandardScaler()
sensor_columns = [col for col in train_data.columns if 'Sensor' in col]
train_data[sensor_columns] = scaler.fit_transform(train_data[sensor_columns])
test_data[sensor_columns] = scaler.transform(test_data[sensor_columns])

# Streamlit Dashboard
st.title("Predictive Maintenance Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select an option", ["Data Overview", "Failure Predictions", "Equipment Status", "Maintenance Scheduler"])

if options == "Data Overview":
    st.header("Data Overview")
    st.write("Preview of the training data:")
    st.dataframe(train_data.head())
    st.write("Preview of the test data:")
    st.dataframe(test_data.head())
    st.write("Summary statistics of the training data:")
    st.write(train_data.describe())

elif options == "Failure Predictions":
    st.header("Failure Predictions")
    unit_number = st.number_input("Enter Unit Number", min_value=1, max_value=int(test_data['UnitNumber'].max()), value=1)

    # Filter data for the selected unit in the test set
    unit_data = test_data[test_data['UnitNumber'] == unit_number]
    if unit_data.empty:
        st.warning("No data found for this unit number.")
    else:
        # Prepare the data for the LSTM model
        scaled_unit_data = unit_data[sensor_columns].values
        timesteps = 50  # Adjust to match your model's expected timesteps
        features = scaled_unit_data.shape[1]

        if scaled_unit_data.shape[0] < timesteps:
            padding = np.zeros((timesteps - scaled_unit_data.shape[0], features))
            scaled_unit_data = np.vstack((padding, scaled_unit_data))
        else:
            scaled_unit_data = scaled_unit_data[-timesteps:]

        # Reshape data for LSTM model (samples, timesteps, features)
        input_data = np.expand_dims(scaled_unit_data, axis=0)

        # Make a prediction
        try:
            prediction = model.predict(input_data)
            predicted_rul = int(prediction[0][0])
            st.write(f"Predicted Remaining Useful Life (RUL) for Unit {unit_number}: {predicted_rul} cycles")

            # Provide maintenance recommendations
            if predicted_rul < 30:
                st.warning("Urgent: Schedule maintenance immediately to avoid failure.")
            elif predicted_rul < 100:
                st.info("Schedule maintenance soon to ensure equipment reliability.")
            else:
                st.success("Equipment is in good condition. Routine maintenance is sufficient.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

elif options == "Equipment Status":
    st.header("Equipment Status Visualization")

    # Select a unit to view status history
    selected_unit = st.selectbox("Select Unit Number", sorted(test_data['UnitNumber'].unique()))
    unit_history = test_data[test_data['UnitNumber'] == selected_unit]

    # Plot sensor readings over time
    plt.figure(figsize=(10, 6))
    for sensor in sensor_columns:
        plt.plot(unit_history['TimeInCycles'], unit_history[sensor], label=sensor)
    plt.title(f"Sensor Readings for Unit {selected_unit}")
    plt.xlabel("Time in Cycles")
    plt.ylabel("Sensor Readings (Normalized)")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    st.pyplot(plt)

elif options == "Maintenance Scheduler":
    st.header("Maintenance Scheduler")

    # Maintenance history (for simplicity, using a sample dataframe)
    maintenance_history = pd.DataFrame({
        'UnitNumber': [1, 2, 3, 4, 5],
        'LastMaintenance': ['2024-01-10', '2024-02-15', '2024-03-20', '2024-04-25', '2024-05-30'],
        'NextScheduled': ['2024-06-10', '2024-07-15', '2024-08-20', '2024-09-25', '2024-10-30']
    })

    st.write("Maintenance History:")
    st.dataframe(maintenance_history)

    # Option to add new maintenance schedule
    st.subheader("Schedule New Maintenance")
    unit_number = st.number_input("Enter Unit Number", min_value=1, max_value=int(test_data['UnitNumber'].max()), value=1)
    next_maintenance_date = st.date_input("Select Next Maintenance Date")

    if st.button("Schedule Maintenance"):
        # Append new maintenance schedule to the history (not persistent)
        new_entry = {'UnitNumber': unit_number, 'LastMaintenance': str(pd.Timestamp.now().date()), 'NextScheduled': str(next_maintenance_date)}
        maintenance_history = maintenance_history.append(new_entry, ignore_index=True)
        st.success(f"Maintenance scheduled for Unit {unit_number} on {next_maintenance_date}.")
        st.write("Updated Maintenance History:")
        st.dataframe(maintenance_history)
