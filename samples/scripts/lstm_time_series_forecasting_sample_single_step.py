# -----------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -----------------------------------------------------------------------------------------

# Sample demonstration of TensorFlow and LSTMs in single_step forecasting of weather
# timeseries data
# Usage: python3 lstm_time_series_forecasting_sample_single_step.py

#  ----------------------- Imports & Set Parameters START --------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

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
	zip_path = tf.keras.utils.get_file(
		origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
		fname='jena_climate_2009_2016.csv.zip',
		extract=True)
	csv_path, _ = os.path.splitext(zip_path)
	df = pd.read_csv(csv_path)

	# Extract temperature data from dataset
	uni_data = df['T (degC)']
	uni_data.index = df['Date Time']

	# Normalize dataset 
	uni_data = uni_data.values
	uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
	uni_train_std = uni_data[:TRAIN_SPLIT].std()
	uni_data = (uni_data-uni_train_mean)/uni_train_std

	return uni_data

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
#  ----------------------- Helper functions END ----------------------------------

def single_step_prediction(data_univariate):
	# Divide dataset for training and validating
	univariate_past_history = 20
	univariate_future_target = 0

	x_train_uni, y_train_uni = prepare_univariate_train_and_validation_data(data_univariate, 0, TRAIN_SPLIT,
																						univariate_past_history,
																						univariate_future_target)
	x_val_uni, y_val_uni = prepare_univariate_train_and_validation_data(data_univariate, TRAIN_SPLIT, None,
																				univariate_past_history,
																				univariate_future_target)
	BATCH_SIZE = 256
	BUFFER_SIZE = 10000

	train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
	train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
	val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

	simple_lstm_model = tf.keras.models.Sequential([
			tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
			tf.keras.layers.Dense(1)
	])

	simple_lstm_model.compile(optimizer='adam', loss='mae')

	print("Size of the training dataset for LSTM Single-Step Forecasting on TensorFlow: {} datapoints, equal to around {} days of data sampled every 10 minutes.".format(x_train_uni.size, round(x_train_uni.size/(6*24))))


	EVALUATION_INTERVAL = 200
	EPOCHS = 10

	simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
												steps_per_epoch=EVALUATION_INTERVAL,
												validation_data=val_univariate, validation_steps=50)

	# Validating
	mape_total = 0
	iter_count = 0
	for x, y in val_univariate.__iter__():
			prediction = simple_lstm_model.predict(x)[0]
			mape_total += abs(y[0].numpy() - prediction) / abs(y[0])
			iter_count += 1
			if iter_count % 1000 == 0:
					print("Iteration {}/{} - Current MAPE: {}%".format(iter_count, len(x_val_uni), mape_total*100/iter_count))
	print("Mean absolute percentage error of {} samples for single-step SSA forecasting: {}%.".format(len(iter_count), mape_total*100/iter_count))

def main():
	print("Demonstration of LSTM Single-Step Forecasting on TensorFlow")
	# For replicability
	tf.random.set_seed(13)
	data_univariate = prepare_univariate_raw_data()
	single_step_prediction(data_univariate)

if __name__ == "__main__":
		main()