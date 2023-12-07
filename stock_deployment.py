# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model("saved_model.h5")



# Streamlit app
def main():
    st.title('Stock Price Prediction App')

    # Sidebar for user input
    st.sidebar.header('User Input Parameters')

    # Add input elements for Date, Open, Close, High, Low
    
    input_open = st.sidebar.number_input('Enter Open Price', min_value=0.0)
    input_high = st.sidebar.number_input('Enter High Price', min_value=0.0)
    input_low = st.sidebar.number_input('Enter Low Price', min_value=0.0)

    # Data preprocessing for prediction
    user_input = {
        
        'Open': input_open,
        'High': input_high,
        'Low': input_low
    }

    # Make predictions
    # Assuming X_new is your new input data for prediction
    X_new = user_input  # Adjust this based on your actual input data

    # Data preprocessing for prediction
    #X_new_scaled = scaler.fit_transform(X_new)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    feature_names = ["Open",  "High" ,"Low"]
    # Reshape and pad sequences
    X_new = pd.DataFrame([user_input], columns=feature_names)
    scaled_input = scaler.transform(X_new)
    X_new_reshaped = pd.DataFrame(scaled_input, columns=feature_names)
    
    # Assuming X_new_reshaped is your input data
    X_reshaped = X_new_reshaped.to_numpy().reshape(X_new_reshaped.shape[0], 1, X_new_reshaped.shape[1])

    # Make predictions
    predicted_closing_price = model.predict(X_reshaped)
    st.write(predicted_closing_price)



# Save the Streamlit app
if __name__ == '__main__':
    main()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #st.save_to_buffer('app.py', format='python')