# -*- coding: utf-8 -*-
"""Stock Prediction Model (Arthur And Princess) Last Version

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C6rDhcoU9LNw4MmtWQSNor1ocMRR0OQ-

## ***STOCK PREDICTIONS (Global Beverages Limited)***

**IMPORTATION OF NECCESSARY LIBRARIES AND MOUNTING OF** **DRIVE**
"""

# Commented out IPython magic to ensure Python compatibility.
# Installation of required packages
!pip install tensorflow scikeras scikit-learn
!pip install keras-tuner

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning libraries
import sklearn as sk
from sklearn import metrics
from sklearn.model_selection import (KFold, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, accuracy_score, make_scorer)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
scaler=MinMaxScaler(feature_range=(0,1))

# Keras
import keras
import keras_tuner
from kerastuner import Objective
from kerastuner.tuners import RandomSearch
from keras.models import Model, Sequential
from scikeras.wrappers import KerasClassifier
from keras.layers import Input, Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Data Visualization
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
import seaborn as sns


# Additional Libraries
from google.colab import drive

# Mounts the Google Drive to the specified directory '/content/drive'
# This allows access to files and data stored in your Google Drive within the Colab environment
drive.mount('/content/drive')

"""***DATA COLLECTION & PRE-PROCESSING***

"""

# Reasa CSV file into a Pandas DataFrame
# This file is located in the specified Google Drive directory and is loaded into the variabe 'first_dataset (2022 to 2023)' for further data analysis
dataset = pd.read_csv('/content/drive/MyDrive/stock_data.csv')

dataset.head()

# Analyze the closing prices from dataframe
dataset["Date"]=pd.to_datetime(dataset.Date,format="%Y-%m-%d")
dataset.index = dataset['Date']

plt.figure(figsize=(16,8))
plt.plot(dataset["Close"],label='Close Price history')

# Customize labels and title
plt.xlabel('Time Scale')
plt.ylabel('Scaled Price')
plt.title('Stock Price Chart')
plt.legend()

# Show the plot
plt.show()

# Sort the dataset on date time and filter “Date” and “Close” columns
data = dataset.sort_index(ascending=True,axis=0)
new_dataset = pd.DataFrame(index=range(0,len(dataset)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["Close"][i] = data["Close"][i]

# Select columns with numeric data typesin the Pandas Dataframe
# This filters and returns ONLY the columns that contain numerical data
dataset.select_dtypes(include='number')

# The 'verbose = True' option provides detailed information, including the datatypes,non-null counts, and memory usage
dataset.info(verbose=True)

# This provides a previewof the top 20 rows of the DataFrame for us to examine
dataset.head(20)

"""Removing columns with **30%** or more null values*"""

# Defines a threshold for missing data
threshold = 0.3

# Calculates the proportion of missing data for each column
nullpercent = dataset.isnull().mean()

# Filter the DataFame to retain columns wit missing data proportions less than the specified threshold
dataset = dataset.loc[:,nullpercent<threshold]

# Analyzing the new DataFrame
dataset.info(verbose = True)

# This filters and returns ONLY the columns that DO NOT contain numeric data, excluding integers, floats, etc.
dataset.select_dtypes(exclude=['number'])

# Numeric Splitting (Imputing for Numerical Values)
num1 = dataset.select_dtypes(include=['number'])

# Non-numeric Splitting (Imputing for Categorical Values)
obj1 = dataset.select_dtypes(exclude=['number'])

# This includes statistics like count, mean, standard deviation, minimum, and quartiles for each numeric columns
num1.describe()

# Import the SimpleImputer classfrom scikit-learn to handle missing values
from sklearn.impute import SimpleImputer

# Create a SimpleImputer instance with the strategyof imputing missing values using the mean
imp=SimpleImputer(strategy="mean")

# Calculate the mean values of the numeric columns in the DataFrame and store them as an array
num1.mean().values

# Fit imputer to the data in 'num1' to replace missing valued with the calculated means
imp.fit(num1)

# Use the imputer to fill missing values in the DataFrame 'num1' and store the result in 'X'
X=imp.transform(num1)

# Create a new DataFrame'num1_imputed' with values, preserving column names
num1_imputed=pd.DataFrame(X,columns=num1.columns)

num1 = num1_imputed

num1.isnull().any()

# Count the number of null values
num1.isnull().sum()

# Forward-fill (ffill) propagates the last observed non-null value forward in each column
new_obj1= obj1.ffill()

new_obj1.info(verbose=True)

new_obj1.isnull().sum()

obj1 = new_obj1

obj1.isnull().any()

print(obj1.columns)

# List of column names to be encoded
columns = ['Date', 'Stock']

# Create a dictionary to store LabelEncoder instances for each column
dict_obj = {}

# Iterate through the list of columns and apply LabelEncoder to each
for col in columns:
  obj=LabelEncoder()
  obj1[col]=obj.fit_transform(obj1[col])
dict_obj[col]=obj

obj1

# Reset the index of both DataFrames to ensure unique index values
obj1.reset_index(drop=True, inplace=True)
num1.reset_index(drop=True, inplace=True)

# Concatenate the DataFrames along axis 1
cleaned_dataset = pd.concat([obj1, num1], axis=1)

cleaned_dataset.isnull().sum().sum()

print(cleaned_dataset.columns)

"""***FEATURE ENGINEERING AND IMPORTANCE***

Data Scaling
"""

# Normalize the new filtered dataset
scaler = MinMaxScaler(feature_range=(0,1))
final_dataset = cleaned_dataset.values

train_data = final_dataset[0:987,:]
valid_data = final_dataset[987:,:]

cleaned_dataset.index = cleaned_dataset.Date
cleaned_dataset.drop("Date",axis=1,inplace=True)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

# Calculate the correlation matrix for the Dataframe
correlation_matrix = cleaned_dataset.corr()

# Extract the 'Closing Price' column as the target variable
target = cleaned_dataset['Close']

# Sort and store the features by their correlation with the 'Closing Price' tagreget variable in descending order
top_features = correlation_matrix['Close'].sort_values(ascending = False)

# Create a bar plot to visualize the correlation of the featues with the target variable
top_features.plot(kind='bar')

# Set the title for the plot
plt.title("Correlations with the TargetVariable")

# Label the x-axis with "Correlation Coefficient"
plt.xlabel("Correlation Coefficient")

# Display the plot
plt.show()

# Set a correlation threshold limit of 0.55
limits = 0.55

# Extract the column names with the correlation greater than 0.55 with the 'overall' target variable
corrs = correlation_matrix[correlation_matrix['Close']>limits].index

# Create a newDataFrame containing the columns that meet the correlation threshold
chosen_correlations = data[corrs]

#Get the column names
chosen_correlations.columns

# Assigns
cleaned_dataset = chosen_correlations

cleaned_dataset.columns

"""***BUILDING AND TRAINING MODELS***"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Remove the 'Closing Price' column from the DataFrame 'features'to create the feature matrix 'X'
X = cleaned_dataset.drop("Close", axis=1)

# Extract the 'Closing Price' column as the target variable 'y'
y = cleaned_dataset["Close"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Convert X_train to a NumPy array and reshape
x_train_reshaped = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))

# Convert y_train to a NumPy array
y_train_array = y_train.values

# Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_reshaped.shape[1], x_train_reshaped.shape[2])))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the entire dataset
model.fit(x_train_reshaped, y_train_array, epochs=1, batch_size=1, verbose=2)

# Assuming X_test and y_test are your test data
# Convert X_test to a NumPy array and reshape
X_test_reshaped = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))

# Convert y_test to a NumPy array
y_test_array = y_test.values

# Evaluate the model on the test data
results = model.evaluate(X_test_reshaped, y_test_array, verbose=0)
print(f"Fold Loss: {results}")

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Assuming 'Date' is the index column in your DataFrame
X_test_reshaped = []

for i in range(60, cleaned_dataset.shape[0]):
    X_test_reshaped.append(cleaned_dataset.loc[cleaned_dataset.index[i - 60:i], 'Close'].values)

# Pad sequences to ensure uniform length
X_test_reshaped = pad_sequences(X_test_reshaped, dtype='float32', padding='post', truncating='post')

# Assuming 'Close' is the name of the column you want to predict
X_test_reshaped = np.reshape(X_test_reshaped, (X_test_reshaped.shape[0], X_test_reshaped.shape[1]))

# Assuming 'Close' is the name of the column you want to predict
X_test_reshaped = np.reshape(X_test_reshaped, (X_test_reshaped.shape[0], 1, X_test_reshaped.shape[1]))

predicted_closing_price = model.predict(X_test_reshaped)
# predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

# Visualize the predicted stock costs with actual stock costs
train_data = cleaned_dataset[:987]
valid_data = cleaned_dataset[987:]
valid_data['Predictions'] = predicted_closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])

# Save the LSTM model
model.save("saved_model.h5")