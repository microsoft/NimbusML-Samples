# -----------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -----------------------------------------------------------------------------------------

# Sample demonstration of NimbusML and SsaForecaster in single_step forecasting of weather
# timeseries data
# Usage: python3 ssa_time_series_forecasting_sample_single_step.py

#  ----------------------- Imports & Set Parameters START --------------------------------
import os
import pandas as pd

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from nimbusml import Pipeline
from nimbusml.timeseries import SsaForecaster
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

TRAIN_SPLIT = 300000
#  ----------------------- Imports & Set Parameters END ----------------------------------

#  ----------------------- Helper functions START ----------------------------------------
def create_time_steps(length):
	time_steps = []
	for i in range(-length, 0, 1):
		time_steps.append(i)
	return time_steps

def show_plot(plot_data, delta, title):
	labels = ['History', 'True Future', 'Model Prediction']
	marker = ['.-', 'rx', 'go']
	time_steps = create_time_steps(plot_data[0].shape[0])
	if delta:
		future = delta
	else:
		future = 0

	plt.title(title)
	for i, x in enumerate(plot_data):
		if i:
			plt.plot(future, plot_data[i], marker[i], markersize=10,
							 label=labels[i])
		else:
			plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
	plt.legend()
	plt.xlim([time_steps[0], (future+5)*2])
	plt.xlabel('Time-Step')
	return plt

def baseline(history):
	return np.mean(history)

def prepare_univariate_raw_data():
	# Load weather time series data from GitHub
	url_to_csv = "https://raw.githubusercontent.com/mstfbl/NimbusML-Samples/Issue-22/datasets/max_planck_weather_time_series_dataset.csv"
	df_train = pd.read_csv(filepath_or_buffer=url_to_csv, sep = ",",)

	# Extract temperature data from dataset
	data_univariate = df_train['T (degC)']
	data_univariate.index = df_train['Date Time']

	# Normalize dataset 
	data_univariate = data_univariate.values
	data_univariate_mean = data_univariate[:TRAIN_SPLIT].mean()
	data_univariate_std = data_univariate[:TRAIN_SPLIT].std()
	data_univariate = (data_univariate-data_univariate_mean)/data_univariate_std

	return data_univariate

def prepare_univariate_train_and_validation_data(dataset, start_index, end_index, history_size, target_size):
	data = []
	labels = []

	start_index = start_index + history_size
	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i)
		data.append(np.reshape(dataset[indices], (history_size, 1)))
		labels.append(dataset[i+target_size])
	return np.array(data), np.array(labels)

def get_uni_standard_trained_on_entire_data_ssa_pipeline():
	return Pipeline([
		SsaForecaster(series_length=6,
									train_size=20,
									window_size=3,
									horizon=1,
									columns={'T (degC)_fc': 'T (degC)'})
])
#  ----------------------- Helper functions END ----------------------------------

def single_step_prediction(data_univariate):
	# Divide dataset for training and validating
	univariate_past_history = 20
	univariate_future_target = 1

	x_train_uni = np.array(data_univariate[:TRAIN_SPLIT])

	x_val_uni, y_val_uni = prepare_univariate_train_and_validation_data(data_univariate, TRAIN_SPLIT, None,
																								univariate_past_history,
																								univariate_future_target)

	print("Size of the training dataset for SSA Single-Step Forecasting on NimbusML: {} datapoints, equal to around {} days of data sampled every 10 minutes.".format(x_train_uni.size, round(x_train_uni.size/(6*24))))
	# Training
	x_train_ssa_big = pd.Series(x_train_uni, name="T (degC)")
	pipeline = get_uni_standard_trained_on_entire_data_ssa_pipeline()
	print("Training SSA Forecasting model...")
	pipeline.fit(x_train_ssa_big)
	# Validating
	print("Commencing validation...")
	mapes_total = 0
	iter_count = 0
	for i in range(len(y_val_uni)):
		# Predicting
		predicted_val_uni = pipeline.transform(pd.Series(x_val_uni[i].flatten(), name="T (degC)")).drop("T (degC)", 1)
		predicted_val = predicted_val_uni['T (degC)_fc.0'][19]
		# Performace
		# mean absolute percentage error = mean(((y_true - y_pred) / y_true)) * 100)
		mapes_total += abs(y_val_uni[i] - predicted_val) / abs(y_val_uni[i])[0]
		iter_count += 1
		if iter_count % 1000 == 0:
			print("Iteration {}/{} - Current MAPE: {}%".format(iter_count, len(y_val_uni), mapes_total*100/iter_count))
	print("Mean absolute percentage error of {} samples for single-Step SSA forecasting: {}%.".format(iter_count, mapes_total*100/iter_count))

def main():
	print("Demonstration of SSA Single-Step Forecasting on NimbusML")
	data_univariate = prepare_univariate_raw_data()
	single_step_prediction(data_univariate)

if __name__ == "__main__":
		main()