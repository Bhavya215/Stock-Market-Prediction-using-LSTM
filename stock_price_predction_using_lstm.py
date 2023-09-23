!pip install -q yfinance

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
# %matplotlib inline

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()

from datetime import datetime


# The tech stocks we'll use for this analysis
company_data = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# Set up End and Start times for data grab
company_data = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for stock in company_data:
    globals()[stock] = yf.download(stock, start, end)
    

company_list = [AAPL, GOOG, MSFT, AMZN]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)
df.tail(10)

AAPL.describe()

# General info
AAPL.info()

plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot(color = 'green')
    plt.ylabel('Closing Price')
    plt.xlabel(None)
    plt.title(f"Price at Closing of {company_data[i - 1]}")
    
plt.tight_layout()

ma_day = [10, 20, 50]

for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()
        

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0], color = ['purple', 'pink', 'orange', 'magenta'])
axes[0,0].set_title('Company - Apple')

GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1], color = ['purple', 'pink', 'orange', 'magenta'])
axes[0,1].set_title('Company - Google')

MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0], color = ['purple', 'pink', 'orange', 'magenta'])
axes[1,0].set_title('Company - Microsoft')

AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1], color = ['purple', 'pink', 'orange', 'magenta'])
axes[1,1].set_title('Company - Amazon')

fig.tight_layout()

c_df = pdr.get_data_yahoo(company_data, start=start, end=end)['Adj Close']

stats = c_df.pct_change()
stats.head()

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.heatmap(stats.corr(), annot=True, cmap='ocean')
plt.title('Correlation of Returns')

plt.subplot(2, 2, 2)
sns.heatmap(c_df.corr(), annot=True, cmap='ocean')
plt.title('Correlation of Closing Price of Stock')

dataframe = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())

dataframe

# Fetching the data based on the 'Close' column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_length = int(np.ceil( len(dataset)))

training_data_length

# Scaling
sc_var = MinMaxScaler(feature_range=(0,1))
data_sc = sc_var.fit_transform(dataset)

# Training dataset
trained_dataset = data_sc[0:int(training_data_length), :]


X_train = []
Y_train = []

for i in range(70, len(trained_dataset)):
    X_train.append(trained_dataset[i-70:i, 0])
    Y_train.append(trained_dataset[i, 0])
 
X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Model

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, Y_train, batch_size=1, epochs=1)

#Testing Dataset

testing_dataset = data_sc[training_data_length - 70: , :]

X_test = []
Y_test = dataset[training_data_length:, :]
for i in range(70, len(testing_dataset)):
    X_test.append(testing_dataset[i-70:i, 0])
    

X_test = np.array(X_test)
X_test

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))

 
predict_data = model.predict(X_test)
predict_data = sc_var.inverse_transform(predict_data)


rmse = np.sqrt(np.mean(((predict_data - Y_test) ** 2)))
print("Rmse: ",rmse/10)

# Plot the data
train = data[:training_data_length]
validated_data = data[training_data_length:]
validated_data['Predicted Data'] = predict_data

plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(validated_data[['Close', 'Predicted Data']])
plt.legend(['Train', 'Val', 'Predicted Data'], loc='lower right')
plt.show()

# Show the validated_data and predicted prices
validated_data