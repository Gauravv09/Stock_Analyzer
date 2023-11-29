# Define the list of tickers
import numpy as np
import os
import yfinance as yfin
from sklearn.ensemble import RandomForestClassifier
import ta as ta
from sklearn.model_selection import train_test_split, ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, r2_score, classification_report
#from sklearn.metrics import plot_roc_curve
#from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd


tickers =   [
'AAPL',
'MSFT',
'AMZN',
'TSLA',
'BRK-B',
'UNH',
'GOOGL',
'XOM',
'JNJ',
'JPM',
]

#Download data
stocks = []
for ticker in tickers:
    stocks.append(yfin.download(ticker, start="2020-3-2", end="2021-2-09"))
#Add technical indicators
feature_names = ['RSI', 'SO', 'SMA', 'MACD', 'ROC', 'OBV']
for data in stocks:  
    #RSI Indicator: Relative Strength Index
    data['RSI'] = ta.momentum.RSIIndicator(data['Adj Close'], window=14, fillna=True).rsi()
    #Stochastic Oscillator
    data['SO'] = ta.momentum.StochRSIIndicator(data['Adj Close'], window=14, fillna=True).stochrsi()
    #Simple MOving Average
    data['SMA'] = ta.trend.sma_indicator(data['Adj Close'], window=14, fillna=True)
    #MACD
    data['MACD'] = ta.trend.macd(data['Adj Close'], window_slow = 26, window_fast = 12, fillna=True)
    # ROC Rate of Change (momentum)
    data['ROC'] = ta.momentum.ROCIndicator(data['Adj Close'], window=14, fillna=True).roc()
    #OBV: On Balance Volume
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=data['Adj Close'], volume=data['Volume'], fillna=True).on_balance_volume()

# Add labels:
# Buy (1): Positive change in price (>= 0)  
# Sell (0): Negative change in price (< 0)   
for stock in stocks:
    stock['5d_change_in_price'] = stock['Adj Close'].diff(-5)*-1
    stock['5d_label'] = stock['5d_change_in_price'].apply(lambda x: 1 if x >= 0 else 0)
    stock['20d_change_in_price'] = stock['Adj Close'].diff(-20)*-1
    stock['20d_label'] = stock['20d_change_in_price'].apply(lambda x: 1 if x >= 0 else 0)

# ...

# Preprocessing and Scaling: 5 day prediction
X_train_5d = []
y_train_5d = []
X_test_5d = []
y_test_5d = []

for stock in stocks:
    # Any row that has a NaN value will be dropped.
    stock = stock.dropna()

    # Check if there are enough samples for the split
    if len(stock) > 1:
        # Train-test split
        X = stock[feature_names]
        y = stock['5d_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Store data for each stock
        X_train_5d.append(X_train)
        y_train_5d.append(y_train)
        X_test_5d.append(X_test)
        y_test_5d.append(y_test)

# Preprocessing and Scaling: 20 day prediction
X_train_20d = []
y_train_20d = []
X_test_20d = []
y_test_20d = []

for stock in stocks:
    # Any row that has a NaN value will be dropped.
    stock = stock.dropna()

    # Check if there are enough samples for the split
    if len(stock) > 1:
        # Train-test split
        X = stock[feature_names]
        y = stock['20d_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Store data for each stock
        X_train_20d.append(X_train)
        y_train_20d.append(y_train)
        X_test_20d.append(X_test)
        y_test_20d.append(y_test)



import matplotlib
print(matplotlib._version) #3.2.2 - OLD
y_5d = [stock['5d_label'] for stock in stocks]

# Simple distribution counts
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4), sharey=True)
for stock, stock_y5d, ax in zip(tickers, y_5d, axes.flatten()):
    distribution_5d = stock_y5d.value_counts()
    ax = sns.barplot(x=distribution_5d.index, y=distribution_5d.values, ax=ax)
    ax.set_title(stock)
    # ax.set_xticks([0,1], labels=['Sell', 'Buy']) # [0,1] # not compatible with matplotlib version 3.2.2
    ax.set_xticks([0,1])
    ax.set_xticklabels(labels=['Sell', 'Buy'])
    # print(distribution_5d)

# fig.suptitle('Label Distribution') # eithr collab or matplotlib 3.2.2 doesn't like this
fig.tight_layout()

plt.show()





# Distribution over time
fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(12,12))
for stock, stock_y_5d, ax in zip(tickers, y_5d, axes.flatten()):
    y_up = stock_y_5d[stock_y_5d == 1]
    xs_up = y_up.index
    ys_up = np.ones(len(xs_up))
    sns.scatterplot(x=xs_up, y=ys_up, color='orange', ax=ax)

    y_down = stock_y_5d[stock_y_5d == 0]
    xs_down = y_down.index
    ys_down = np.zeros(len(xs_down))
    sns.scatterplot(x=xs_down, y=ys_down, color='blue', ax=ax)

    ax.set_title(stock)
    ax.set_yticks([0,1])
    ax.set_yticklabels(labels=['Sell', 'Buy'])

fig.tight_layout()
plt.show()















models_random_forest_5d = []
models_random_forest_20d = []
scores_5d = []
scores_20d = []

#5 day predictions
for i, stock in enumerate(stocks):
    X_train = X_train_5d[i]
    y_train = y_train_5d[i]
    X_test = X_test_5d[i]
    y_test = y_test_5d[i]
    
    # Create a Random Forest Classifier
    # rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 42)
    rand_frst_clf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [3,5,8], 'max_features': [4, 8]}
    CV = GridSearchCV(estimator=rand_frst_clf, param_grid=param_grid, cv= 5, n_jobs=-1, verbose=2)
    # CV = RandomizedSearchCV(estimator=rand_frst_clf, param_grid=param_grid, cv= 5,verbose=2)
    CV.fit(X_train, y_train)
    # rand_frst_clf_best = RandomForestClassifier(n_estimators=CV.best_params_['n_estimators'],max_depth=CV.best_params_['max_depth'], max_features=CV.best_params_['n_estimators'], random_state=42)
    rf_best = CV.best_estimator_

    # Fit the data to the model
    # rand_frst_clf.fit(X_train, y_train)
    # rand_frst_clf_best.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_best.predict(X_test)
    # y_pred = rand_frst_clf_best.predict(X_test)
    
    # Print the Accuracy of our Model.
    score = accuracy_score(y_test, y_pred, normalize = True) * 100.0
    scores_5d.append(score)
    print('5d Random Forest Accuracy (%) for', tickers[i],":", score)
    # print('Accuracy (%): ', accuracy_score(y_test, rand_frst_clf_best.predict(X_test), normalize = True) * 100.0)

    #save models
    models_random_forest_5d.append(rf_best)

#20 day predictions [TBD]
#5 day predictions
for i, stock in enumerate(stocks):
    X_train = X_train_20d[i]
    y_train = y_train_20d[i]
    X_test = X_test_20d[i]
    y_test = y_test_20d[i]
    
    # Create a Random Forest Classifier
    # rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 42)
    rand_frst_clf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [3,5,8], 'max_features': [4, 8]}
    CV = GridSearchCV(estimator=rand_frst_clf, param_grid=param_grid, cv= 5, n_jobs=-1, verbose=2)
    # CV = RandomizedSearchCV(estimator=rand_frst_clf, param_grid=param_grid, cv= 5,verbose=2)
    CV.fit(X_train, y_train)
    # rand_frst_clf_best = RandomForestClassifier(n_estimators=CV.best_params_['n_estimators'],max_depth=CV.best_params_['max_depth'], max_features=CV.best_params_['n_estimators'], random_state=42)
    rf_best = CV.best_estimator_

    # Fit the data to the model
    # rand_frst_clf.fit(X_train, y_train)
    # rand_frst_clf_best.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_best.predict(X_test)
    # y_pred = rand_frst_clf_best.predict(X_test)
    
    # Print the Accuracy of our Model.
    score = accuracy_score(y_test, y_pred, normalize = True) * 100.0
    scores_20d.append(score)
    print('20d Random Forest Accuracy (%) for', tickers[i],":", score)

    # print('Accuracy (%): ', accuracy_score(y_test, rand_frst_clf_best.predict(X_test), normalize = True) * 100.0)

    #save models
    models_random_forest_20d.append(rf_best)



for i in range(len(tickers)):
  y_pred = models_random_forest_5d[i].predict(X_test_5d[i])
  print("5d Classification Report for", tickers[i])
  report = classification_report(y_true = y_test_5d[i], y_pred = y_pred, target_names = ['down','up'], output_dict = True)
  report_df = pd.DataFrame(report).transpose()
  print(report_df)
  print()



    

labels = tickers

x = np.arange(len(labels))
width = 0.35  # the width of the bars
plt.figure(figsize=(10,7))
rects1 = plt.bar(x - width/2, scores_5d, width, label='5d prediction accuracy')
rects2 = plt.bar(x + width/2, scores_20d, width, label='20d prediction accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Classification Accuracy Score')
plt.title('Random Forest Accuracy by Stock')
plt.xticks(x, labels)
plt.legend()

plt.show()
     



