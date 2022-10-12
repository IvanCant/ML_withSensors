import os
import pathlib
import numpy as np
import tensorflow
from tensorflow.keras import Model
#from tensorflow.keras.datasets import cifar10 #Not the dataset we are using
from tensorflow.keras.layers import Add, GlobalAveragePooling2D,\
	Dense, Flatten, Conv2D, Lambda,	Input, BatchNormalization, Activation
from tensorflow.keras.optimizers import schedules, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

def model_configuration():
	"""
	variables of configuration for the model
	"""
	# Load dataset for computing dataset size
	(input_train, _), (_, _) = load_dataset()

	width, height, channels = 32, 32, 3 # Will test different options

	batch_size = ?
	num_classes = ?  # Depending on datasets found
	validation_split = 0.1
	verbose = 1
	n = 3
	init_fm_dim = 16
	shortcut_type = "identity" # or: projection

	#Data Size
	train_size = (1 - validation_split) * len(input_train)
	val_size = validation_split * len(input_train)

	# Number of epoch depends on batch size
	maximum_number_iterations = ??????
	steps_per_epoch = tensorflow.math.floor(train_size / batch_size)
	val_steps_per_epoch = tensorflow.math.floor(val_size / batch_size)
	epochs = tensorflow.cast(tensorflow.math.floor(maximum_number_iterations / steps_per_epoch),\
			dtype=tensorflow.int64)

	#loss function
	loss = tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True)

	#learing rate
	boudaries = [?,?]
	values = [0.1, 0.01, 0.001]
	lr_schedule = schedules.PiecewiseConstantDecay(boundaries, values)

	# Set layer init
	initializer = tensorflow.keras.initializers.HeNormal()

	# Define optimizer
	optimizer_momentum = 0.9
	optimizer_additional_metrics = ["accuracy"]
	optimizer = SGD(learning_rate=lr_schedule, momentum=optimizer_momentum)

	# Load Tensorboard callback
	tensorboard = TensorBoard(
  	  log_dir=os.path.join(os.getcwd(), "logs"),
  	  histogram_freq=1,
  	  write_images=True
  	)

	# Saves a model checkpoint after every epoch
	checkpoint = ModelCheckpoint(os.path.join(os.getcwd(), "model_checkpoint"),
  				 save_freq="epoch"
	)

	# Add callbacks to list
	callbacks = [tensorboard, checkpoint]

	# Create config dictionary
	config = {
			"width": width,
			"height": height,
			"dim": channels,
			"batch_size": batch_size,
			"num_classes": num_classes,
			"validation_split": validation_split,
			"verbose": verbose,
			"stack_n": n,
			"initial_num_feature_maps": init_fm_dim,
			"training_ds_size": train_size,
			"steps_per_epoch": steps_per_epoch,
			"val_steps_per_epoch": val_steps_per_epoch,
			"num_epochs": epochs,
			"loss": loss,
			"optim": optimizer,
			"optim_learning_rate_schedule": lr_schedule,
			"optim_momentum": optimizer_momentum,
			"optim_additional_metrics": optimizer_additional_metrics,
			"initializer": initializer,
			"callbacks": callbacks,
			"shortcut_type": shortcut_type

	}

	return config


def load_dataset():

	data_dir = './Traffic Signs/'
	data_dir = pathlib.Path(data_dir)

	return data_dir.load_data()

def random_crop(img, random_crop_size):
	# Note: image_data_format is 'channel_last'
    # SOURCE: https://jkjung-avt.github.io/keras-image-cropping/
	assert img.shape[2] == 3
	height, width = img.shape[0], img.shape[1]
	dy, dx = random_crop_size
	x = np.random.randint(0, width - dx + 1)
	y = np.random.randint(0, height - dy + 1)
	return img[y:(y+dy), x:(x+dx), :]

def crop_generator(batches, crop_length):

	while True:
		batch_x, batch_y = next(batches)
		batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
		for i in range(batch_x.shape[0]):
			batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
		yield (batch_crops, batch_y)



