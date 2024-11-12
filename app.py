import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Function to load and handle the dataset
def load_data():
    # Load the training dataset
    dataset_train = pd.read_csv('PM_train.txt', sep=' ', header=None).dropna(axis=1)
    
    # Dynamically set column names based on the number of columns in dataset
    num_columns = dataset_train.shape[1]
    
    # Define the base columns
    base_cols = ['id', 'cycle', 'setting1', 'setting2', 'setting3']  # Assuming 3 settings at the beginning
    
    # Add sensor columns dynamically based on the number of columns
    sensor_cols = [f's{i}' for i in range(1, num_columns - len(base_cols) + 1)]
    col_names = base_cols + sensor_cols
    
    # Apply dynamic column names
    dataset_train.columns = col_names

    # Load the test dataset with dynamic column handling
    dataset_test = pd.read_csv('PM_test.txt', sep=' ', header=None).dropna(axis=1)
    dataset_test.columns = col_names  # Use the same column names
    
    # Load the truth table and merge it with the dataset
    pm_truth = pd.read_csv('PM_truth.txt', sep=' ', header=None).dropna(axis=1)
    pm_truth.columns = ['more']
    pm_truth['id'] = pm_truth.index + 1
    rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    pm_truth['rtf'] = pm_truth['more'] + rul['max']
    pm_truth.drop('more', axis=1, inplace=True)
    
    dataset_test = dataset_test.merge(pm_truth, on=['id'], how='left')
    dataset_test['ttf'] = dataset_test['rtf'] - dataset_test['cycle']
    dataset_test.drop('rtf', axis=1, inplace=True)

    # Label creation for training and test sets
    dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max) - dataset_train['cycle']
    dataset_train['label_bc'] = dataset_train['ttf'].apply(lambda x: 1 if x <= 30 else 0)

    dataset_test['label_bc'] = dataset_test['ttf'].apply(lambda x: 1 if x <= 30 else 0)

    return dataset_train, dataset_test

# Function to scale features
def scale_features(df_train, df_test, features_col_name):
    sc = MinMaxScaler()
    df_train[features_col_name] = sc.fit_transform(df_train[features_col_name])
    df_test[features_col_name] = sc.transform(df_test[features_col_name])
    return df_train, df_test

# Function to generate sequences for LSTM
def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)

# Function to generate labels for LSTM
def gen_label(id_df, seq_length, seq_cols, label):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label = []
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)

# Function to build and compile LSTM model
def build_lstm_model(X_train, seq_length, nb_features):
    model = Sequential()
    model.add(LSTM(input_shape=(seq_length, nb_features), units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main app for Streamlit
def app():
    # Set the title of the app
    st.title("Predictive Maintenance for Manufacturing Equipment")
    st.markdown("This application uses a machine learning model to predict the probability of failure for equipment based on historical sensor data.")
    
    # Load and preprocess data
    dataset_train, dataset_test = load_data()

    features_col_name = [col for col in dataset_train.columns if col not in ['id', 'cycle', 'ttf', 'label_bc']]  # Dynamically select features
    target_col_name = 'label_bc'

    # Scale features
    dataset_train, dataset_test = scale_features(dataset_train, dataset_test, features_col_name)

    # Reshape data for LSTM
    seq_length = 50
    X_train = np.concatenate([gen_sequence(dataset_train[dataset_train['id'] == id], seq_length, features_col_name)
                              for id in dataset_train['id'].unique()], axis=0)
    y_train = np.concatenate([gen_label(dataset_train[dataset_train['id'] == id], seq_length, features_col_name, target_col_name)
                              for id in dataset_train['id'].unique()], axis=0)
    X_test = np.concatenate([gen_sequence(dataset_test[dataset_test['id'] == id], seq_length, features_col_name)
                             for id in dataset_test['id'].unique()], axis=0)
    y_test = np.concatenate([gen_label(dataset_test[dataset_test['id'] == id], seq_length, features_col_name, target_col_name)
                             for id in dataset_test['id'].unique()], axis=0)

    # Build and train LSTM model
    nb_features = X_train.shape[2]
    model = build_lstm_model(X_train, seq_length, nb_features)

    # Fit the model
    model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=0.05, verbose=1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

    # Evaluate model
    scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)
    st.write(f'Accuracy on training data: {scores[1]:.2f}')

    # Test prediction and evaluation
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)  # Convert to binary output
    accuracy = np.mean(y_pred == y_test)
    st.write(f'Accuracy on test data: {accuracy:.2f}')

    # Additional functionality for probability prediction or failure predictions
    def prob_failure(machine_id):
        machine_df = dataset_test[dataset_test.id == machine_id]
        machine_test = gen_sequence(machine_df, seq_length, features_col_name)
        m_pred = model.predict(machine_test)
        failure_prob = list(m_pred[-1] * 100)[0]
        return failure_prob

    # Get user input for machine ID
    machine_id = st.number_input("Enter machine ID for failure probability prediction:", min_value=1, max_value=100, value=16)

    # Calculate failure probability
    failure_prob = prob_failure(machine_id)
    st.write(f'Probability that machine {machine_id} will fail within 30 days: {failure_prob:.2f}%')

    # Plotting the failure probability with matplotlib
    st.subheader("Failure Probability Visualization")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([f"Machine {machine_id}"], [failure_prob], color='blue')
    ax.set_title("Failure Probability (%)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # Provide suggestions based on the failure probability
    if failure_prob > 80:
        st.warning("⚠️ High probability of failure within 30 days. Immediate maintenance is recommended.")
    elif failure_prob > 50:
        st.info("🔶 Moderate probability of failure. It is recommended to schedule maintenance soon.")
    else:
        st.success("✅ Low probability of failure. Maintenance can be scheduled based on usual cycles.")

# Run the Streamlit app
if __name__ == "__main__":
    app()
