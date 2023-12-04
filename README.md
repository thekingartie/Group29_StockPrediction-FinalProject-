# Group29_StockPrediction(Final_Project)

This Stock Prediction Model is developed by Arthur and Princess. The model predicts stock prices using machine learning techniques, specifically Long Short-Term Memory (LSTM) networks. The model is trained on historical stock data, and predictions are made for future stock prices.

## Table of Contents

### Introduction
- Introduction
- Data Collection
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Selection and Engineering
- Data Scaling and Splitting
- Multi-Layer Perceptron (MLP) Model using Functional API
- Model Evaluation
- Model Optimization
- Saving the Final Model


## Introduction

This Stock Prediction Model, developed by Arthur and Princess, utilizes Long Short-Term Memory (LSTM) networks to predict stock prices for Global Beverages Limited. The model is designed to analyze historical stock data and provide predictions for future stock prices. The project covers data collection, preprocessing, exploratory data analysis (EDA), feature selection, model building using the Functional API, model evaluation, model optimization, and saving the final model.

## Data Collection

The dataset is loaded from a CSV file located in the specified Google Drive directory using the pandas library. The dataset includes information such as date, stock prices, and other relevant features.

`dataset = pd.read_csv('/content/drive/MyDrive/stock_data.csv')`


## Data Preprocessing

Columns with 30% or more missing values are dropped, and numerical and categorical features are separated for further analysis. Imputation is performed on missing numerical values, and forward-fill is used for missing categorical values.

`threshold = 30`

`nullpercent = dataset.isnull().mean()`

`dataset = dataset.loc[:, nullpercent < threshold]`



## Exploratory Data Analysis (EDA)
Exploratory Data Analysis involves visualizing stock prices over time to identify trends and patterns. The matplotlib library is used for plotting.

# Sample Graph Plots
`plt.figure(figsize=(16,8))`

`plt.plot(dataset["Close"], label='Close Price history')`

`plt.xlabel('Time Scale')`

`plt.ylabel('Scaled Price')`

`plt.title('Stock Price Chart')`

`plt.legend()`

`plt.show()`



# Feature Selection and Engineering
Feature selection is performed based on the correlation of features with the target variable (closing price). The top features are selected, and the dataset is modified accordingly.

`correlation_matrix = cleaned_dataset.corr()`

`top_features = correlation_matrix['Close'].sort_values(ascending=False)`



# Data Scaling and Splitting
Data scaling is applied using MinMaxScaler, and the dataset is split into training and testing sets.

`scaler = MinMaxScaler(feature_range=(0, 1))`

`scaled_data = scaler.fit_transform(final_dataset)`

`x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`



## Multi-Layer Perceptron (MLP) Model using Functional API
The LSTM model is built using the Sequential model from TensorFlow's Keras API. The architecture includes multiple LSTM layers and a Dense output layer.

`model = Sequential()`

`model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_reshaped.shape[1], x_train_reshaped.shape[2])))`

# Additional LSTM layers...
`model.add(Dense(units=1))`
`model.compile(optimizer='adam', loss='mean_squared_error')`



## Model Evaluation
The model is trained and evaluated on the test data. The mean squared error is used as the loss metric.

`model.fit(x_train_reshaped, y_train_array, epochs=1, batch_size=1, verbose=2)`

`results = model.evaluate(X_test_reshaped, y_test_array, verbose=0)`

`print(f"Fold Loss: {results}")`



## Model Optimization
To optimize the model, the dataset is reshaped and padded, and predictions are made. The model predictions are visualized alongside actual stock prices.

`predicted_closing_price = model.predict(X_test_reshaped)`

`train_data = cleaned_dataset[:987]`

`valid_data = cleaned_dataset[987:]`

`valid_data['Predictions'] = predicted_closing_price`

`plt.plot(train_data["Close"])`

`plt.plot(valid_data[['Close', "Predictions"]])`



# Saving the Final Model
The trained LSTM model is saved for future predictions.

`model.save("saved_model.h5")`

`with open('scaler.pkl', 'wb') as scaler_file:`


    `pickle.dump(scaler, scaler_file)`
