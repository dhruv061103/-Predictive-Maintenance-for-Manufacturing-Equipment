import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Streamlit app
st.title('Predictive Maintenance Dashboard')

# Sidebar: File Uploads
st.sidebar.header("Upload Your Files")

# Upload dataset (txt)
dataset_file = st.sidebar.file_uploader("Upload dataset (.txt format)", type=["txt"])

# Upload model (H5 format for LSTM model)
model_file = st.sidebar.file_uploader("Upload Model (.h5 format)", type=["h5"])

# Acknowledgement of successful uploads
if dataset_file is not None:
    st.sidebar.success("Dataset uploaded successfully!")
    
if model_file is not None:
    st.sidebar.success("Model uploaded successfully!")

# Check if both files are uploaded
if dataset_file is not None and model_file is not None:
    # Save the uploaded model to a temporary file
    model_path = "/tmp/uploaded_model.h5"
    with open(model_path, "wb") as f:
        f.write(model_file.read())
    
    # Load the model from the saved file
    model = load_model(model_path)

    # Load dataset (assuming it's a space-separated .txt file)
    @st.cache
    def load_data(file):
        data = pd.read_csv(file, sep=' ', header=None)
        return data

    data = load_data(dataset_file)

    # Preprocessing dataset (example scaling)
    @st.cache
    def preprocess_data(data):
        # Assume that the last column is the target RUL
        features = data.iloc[:, :-1].values  # all columns except the last one
        target = data.iloc[:, -1].values  # last column (RUL)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        
        return scaled_features, target, scaler

    scaled_features, target, scaler = preprocess_data(data)

    # Prepare sequences for LSTM
    def create_sequences(data, seq_length=50):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    X = create_sequences(scaled_features)
    y = target[50:]  # Assuming RUL is in the 51st column

    # Predict Remaining Useful Life (RUL)
    predictions = model.predict(X)

    # Visualize RUL predictions vs true values
    st.subheader('RUL Predictions vs True Values')

    fig = go.Figure()

    # Add True RUL trace
    fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode='lines+markers', name='True RUL'))

    # Add Predicted RUL trace
    fig.add_trace(go.Scatter(x=np.arange(len(predictions)), y=predictions.flatten(), mode='lines+markers', name='Predicted RUL'))

    fig.update_layout(
        title='RUL Predictions vs True Values',
        xaxis_title='Samples',
        yaxis_title='Remaining Useful Life (RUL)',
        template='plotly_dark'
    )

    st.plotly_chart(fig)

    # Display predictions for specific machine
    machine_id = st.number_input('Enter Machine ID to see failure probability', min_value=1, max_value=len(predictions), value=1)

    # Calculate failure probability (Placeholder)
    failure_prob = 1 - np.exp(-predictions[machine_id-1] / 100)
    st.write(f"Predicted Failure Probability for Machine {machine_id}: {failure_prob * 100:.2f}%")

else:
    st.warning("Please upload both the dataset and model.")
