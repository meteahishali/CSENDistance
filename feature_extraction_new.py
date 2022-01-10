"""
Feature extraction using deep models over the KITTI dataset.
Author: Mete Ahishali,
Tampere University, Tampere, Finland.

Modified by: Tim Rosenkranz
Goethe University, Frankfurt am Main, Germany
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
import cv2
import scipy.io as sio
import pandas as pd
import argparse
import random

from tqdm import tqdm


argparser = argparse.ArgumentParser(description='Feature extraction.')
argparser.add_argument('-m', '--model', help='Model name: DenseNet121, VGG19, or ResNet50.')
args = argparser.parse_args()
modelName = args.model

# Path for the data and annotations.
kittiData = 'kitti-data/'
# Note that KITTI provides annotations only for the training since it is a challenge dataset.
imagePath = kittiData + 'training/image_2/'

# Load a random sample
n = 141981 #  Number of records in file
s = 10000 # Desired sample size
skip = sorted(random.sample(range(n),n-s))[1:]

df = pd.read_csv(kittiData + 'annotations_inc_id.csv')
df_pairs = pd.read_csv(kittiData + 'annotations_interdistance.csv', skiprows=skip)

visualize = False # Visualization of the objects.
objectFeatures = []
gtd = []

# Load the model.
function_name = 'tf.keras.applications.' + modelName
model = eval(function_name + "(include_top=False, weights='imagenet', input_shape=(64, 64, 3), pooling='max')")

pbar = tqdm(total=df_pairs.shape[0], position=1)

for idx, row in df_pairs.iterrows():
	pbar.update(1)

	imageName = kittiData + 'training/image_2/' + row['filename'].replace('txt', 'png')
	im = cv2.imread(imageName) # Load the image.

	obj_ids = row['object_ids'].replace('(','').replace(')','').split(', ')
	id0 = int(obj_ids[0])
	id1 = int(obj_ids[1])

	obj0 = df.iloc[id0]
	obj1 = df.iloc[id1]

	# Object Location.
	x01 = int(obj0['xmin'])
	y01 = int(obj0['ymin'])
	x02 = int(obj0['xmax'])
	y02 = int(obj0['ymax'])

	x11 = int(obj1['xmin'])
	y11 = int(obj1['ymin'])
	x12 = int(obj1['xmax'])
	y12 = int(obj1['ymax'])

	# Feature extraction.
	# Crop images to 64x64
	Object0 = cv2.resize(im[y01:y02, x01:x02, :], (64, 64))
	Object1 = cv2.resize(im[y11:y12, x11:x12, :], (64, 64))

	# Expand dimensions
	Object0 = np.expand_dims(cv2.cvtColor(Object0, cv2.COLOR_BGR2RGB), axis=0)
	Object1 = np.expand_dims(cv2.cvtColor(Object1, cv2.COLOR_BGR2RGB), axis=0)

	# Process NN
	function_name = 'tf.keras.applications.' + modelName[:8].lower() + '.preprocess_input'

	Object0 = eval(function_name + '(Object0)')
	Object1 = eval(function_name + '(Object1)')

	features0 = model.predict(Object0)
	features1 = model.predict(Object1)
	objectFeatures.append([features0, features1])
	
	gtd.append([obj0['observation angle'], obj1['observation angle'], row['obj distance 2D']])
		
	if visualize:
		cv2.rectangle(im, (x01, y01), (x02, y02), (0, 255, 0), 3)
		cv2.rectangle(im, (x11, y11), (x12, y12), (0, 255, 0), 3)
		string = "({}, {}, {})".format(obj0['observation angle'], obj1['observation angle'], row['obj distance 2D'])
		cv2.putText(im, string, (int((x01+x02)/2), int((y01+y02)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

		cv2.imshow("detections", im)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

# Record features.
if not os.path.exists('features/'): os.makedirs('features/')
sio.savemat('features/features_new_' + modelName + '.mat',
			{'objectFeatures' : objectFeatures, 'gtd' : gtd})