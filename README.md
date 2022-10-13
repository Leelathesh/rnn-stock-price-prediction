# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

## Neural Network Model

![image](https://github.com/Leelathesh/rnn-stock-price-prediction/blob/26fefa90e6aa2078b4531a0379ea7e7827242eae/WhatsApp%20Image%202022-10-13%20at%2018.11.56.jpeg)
## DESIGN STEPS

### STEP 1:
import the neccesary tensorflow modules

### STEP 2:
load the stock dataset
### STEP 3:
fit the model and then predict

## PROGRAM
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

dataset_train.tail()

dataset_train.iloc[1256:1258]

train_set = dataset_train.iloc[:,1:2].values

type(train_set)

train_set.shape

plt.plot(np.arange(0,1259),train_set)

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

x_train_array = []
y_train_array = []
for i in range(60,1259):
 x_train_array.append(training_set_scaled[i-60:i,0]) 
 y_train_array.append(training_set_scaled[i,0])
 x_train,y_train = np.array(x_train_array),np.array(y_train_array)
 x_train1 = x_train.reshape((x_train.shape[0],x_train.shape[1],1))

x_train.shape

x_train1.shape

length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(length,n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam',loss='mse')

model.summary()

model.fit(x_train1,y_train,epochs=100,batch_size=32)

dataset_test = pd.read_csv('testset.csv')

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

x_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
Include your code here

## OUTPUT

### True Stock Price, Predicted Stock Price vs time
![image](https://github.com/Leelathesh/rnn-stock-price-prediction/blob/e93d1c3cf7c28f09f5bcb7557d7c65b317272ff3/WhatsApp%20Image%202022-10-13%20at%2018.03.16(1).jpeg)
### Mean Square Error
![image](https://github.com/Leelathesh/rnn-stock-price-prediction/blob/1f7b54af5eb2ff686e7dfa9390f3145b2a004d15/WhatsApp%20Image%202022-10-13%20at%2018.03.46.jpeg)

## RESULT
Thus a recurrence neural network model is developed
