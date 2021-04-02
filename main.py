from datetime import datetime
import time
import modelling as md
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import os
from ta.volatility import BollingerBands
from ta import add_all_ta_features
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import ADXIndicator
from ta.volume import AccDistIndexIndicator
#from numba import jit, cuda, vectorize, njit

training_suffix = "_training.csv"
test_suffix = "_test.csv"
target_column = "target"
id_column = "id"
date_column = "time"
name_column = "name"
date_time_format = "%Y-%m-%dT%H:%M"
start_date_ts = time.mktime(datetime.strptime("2018-05-04T08:12", date_time_format).timetuple())

data_files = ["BTCUSDT","ETHUSDT","LTCUSDT","XRPUSDT"]

print(data_files)
'''
Basic flow; for each training file:
- split target column & drop target column
- drop unnecessary columns
- change data format to be number of minutes after 2018-05-04T08:12
- train & predict
- write to csv
'''


def transform_date(date_time_str):
    index = date_time_str.rindex(":")
    date_time_str = date_time_str[:index]
    date_time = datetime.strptime(date_time_str, date_time_format)
    ts = time.mktime(date_time.timetuple())
    return int(ts - start_date_ts) / 60


def get_date_time():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

def data_preprocessing(data):
	data = data[data.time >= '2020-07-01T00:00:00Z']  
	data.drop(id_column, inplace=True, axis=1)
	data.drop(name_column, inplace=True, axis=1)
	data[date_column] = data[date_column].apply(transform_date)
	return data

def get_X_y(data,train=True):
	y = []
	if(train):
		y = data[target_column]
		data.drop(target_column, inplace=True, axis=1)
	#feature engineering
	# data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
	# indicator_ao = AwesomeOscillatorIndicator(high=data["High"],low=data["Low"],fillna=True)
	# indicator_adx = ADXIndicator(high=data["High"],low=data["Low"],close=data["Close"],fillna=True)
	# indicator_adi = AccDistIndexIndicator(high=data["High"],low=data["Low"],close=data["Close"],volume=data["Volume"],fillna=True)
	# data['ao'] = indicator_ao.awesome_oscillator()
	# data['adx'] = indicator_adx.adx()
	# data['adx_neg'] = indicator_adx.adx_neg()
	# data['adx_pos'] = indicator_adx.adx_pos()
	# data['adi'] = indicator_adi.acc_dist_index()
	X = data
	return X,y

def local_train(data,ratio=0.8):
	data = data[data.time >= '2020-07-01T00:00:00Z']  
	len_train = int(data.shape[0]*ratio)
	target = data[target_column]
	data.drop(target_column, inplace=True, axis=1)
	data.drop(id_column, inplace=True, axis=1)
	data.drop(name_column, inplace=True, axis=1)
	data[date_column] = data[date_column].apply(transform_date)
	#feature engineering
	# data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
	# indicator_ao = AwesomeOscillatorIndicator(high=data["High"],low=data["Low"],fillna=True)
	# indicator_adx = ADXIndicator(high=data["High"],low=data["Low"],close=data["Close"],fillna=True)
	# indicator_adi = AccDistIndexIndicator(high=data["High"],low=data["Low"],close=data["Close"],volume=data["Volume"],fillna=True)
	# data['ao'] = indicator_ao.awesome_oscillator()
	# data['adx'] = indicator_adx.adx()
	# data['adx_neg'] = indicator_adx.adx_neg()
	# data['adx_pos'] = indicator_adx.adx_pos()
	# data['adi'] = indicator_adi.acc_dist_index()

	RFECV_transform = md.fit_RFECV(data,target)
	return RFECV_transform.transform(data[0:len_train]) , RFECV_transform.transform(data[len_train:]), target[0:len_train], target[len_train:]
	#PCA_transform = md.fit_PCA(data)
	#return PCA_transform.transform(data[0:len_train]) , PCA_transform.transform(data[len_train:]), target[0:len_train], target[len_train:]
	#return data[0:len_train] , data[len_train:], target[0:len_train], target[len_train:]

for file in data_files:
    file_name = "data/" + file + training_suffix
    print("Working on: " + file_name + " - started at " + get_date_time())
    data = pd.read_csv(file_name)
    data = data_preprocessing(data)
    X_train, y_train = get_X_y(data)
    # RFECV feature selection
    RFECV_transform = md.fit_RFECV(X_train,y_train)
    X_train = RFECV_transform.transform(X_train)

    # PCA feature selection
    #PCA_transform = md.fit_PCA(X_train)
    #X_train = PCA_transform.transform(X_train)

    xgboost_model = md.fit_xgboost(X_train,y_train)
    
    test_data = pd.read_csv("data/" + file + test_suffix)
    results = pd.DataFrame()
    results[id_column] = test_data[id_column]
    test_data = data_preprocessing(test_data)
    X_test, y_test = get_X_y(test_data,train=False)
    X_test = RFECV_transform.transform(X_test)
    #X_test = PCA_transform.transform(X_test)
    predict = xgboost_model.predict(X_test)

    results[target_column] = predict
    results.to_csv(file + "results.csv")
	
	
	#local train
    # X_train, X_test, y_train, y_test = local_train(data)
    # xgboost_model = md.fit_xgboost(X_train,y_train)
    # predict = xgboost_model.predict(X_test)
    # print(mean_squared_error(y_test,predict,squared=False))

    print("end time: " + get_date_time())