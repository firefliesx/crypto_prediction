from datetime import datetime
import time
import modelling as md
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import os
from numba import jit, cuda, vectorize, njit

training_suffix = "_training.csv"
test_suffix = "_test.csv"
target_column = "target"
id_column = "id"
date_column = "time"
name_column = "name"
date_time_format = "%Y-%m-%dT%H:%M"
start_date_ts = time.mktime(datetime.strptime("2018-05-04T08:12", date_time_format).timetuple())

data_files = set()

for f in os.listdir("data"):
    f = f.split("_")
    data_files.add(f[0])

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


for file in data_files:
    file_name = "data/" + file + training_suffix
    print("Working on: " + file_name + " - started at " + get_date_time())
    data = pd.read_csv(file_name)
    target = data[target_column]
    data.drop(target_column, inplace=True, axis=1)
    data.drop(id_column, inplace=True, axis=1)
    data.drop(name_column, inplace=True, axis=1)
    data[date_column] = data[date_column].apply(transform_date)
    ada = md.fit_ada_boost(data, target, True)

    print(ada)
    test_data = pd.read_csv("data/" + file + test_suffix)
    test_data.drop(name_column, inplace=True, axis=1)
    results = pd.DataFrame()
    results[id_column] = test_data[id_column]
    test_data.drop(id_column, inplace=True, axis=1)
    test_data[date_column] = test_data[date_column].apply(transform_date)
    results[target_column] = ada.predict(test_data)
    results.to_csv(file + "results")
    print("end time: " + get_date_time())
