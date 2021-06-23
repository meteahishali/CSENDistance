"""
Feature extraction using deep models over the KITTI dataset.
Author: Mete Ahishali,
Tampere University, Tampere, Finland.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
import cv2
import scipy.io as sio
import pandas as pd
import argparse

from tqdm import tqdm


argparser = argparse.ArgumentParser(description='Feature extraction.')
argparser.add_argument('-m', '--model', help='Model name: DenseNet121, VGG19, or ResNet50.')
args = argparser.parse_args()
modelName = args.model

# Path for the data and annotations.
kittiData = 'kitti-data/'
# Note that KITTI provides annotations only for the training since it is a challenge dataset.
imagePath = kittiData + 'training/image_2/'
df = pd.read_csv(kittiData + 'annotations.csv')

visualize = False # Visualization of the objects.
objectFeatures = []
gtd = []

# Load the model.
function_name = 'tf.keras.applications.' + modelName
model = eval(function_name + "(include_top=False, weights='imagenet', input_shape=(64, 64, 3), pooling='max')")

pbar = tqdm(total=df.shape[0], position=1)

for idx, row in df.iterrows():
	pbar.update(1)

	imageName = kittiData + 'training/image_2/' + row['filename'].replace('txt', 'png')
	im = cv2.imread(imageName) # Load the image.

	# Object Location.
	x1 = int(row['xmin'])
	y1 = int(row['ymin'])
	x2 = int(row['xmax'])
	y2 = int(row['ymax'])

	# Feature extraction.
	Object = cv2.resize(im[y1:y2, x1:x2, :], (64, 64))
	Object = np.expand_dims(cv2.cvtColor(Object, cv2.COLOR_BGR2RGB), axis=0)
	function_name = 'tf.keras.applications.' + modelName[:8].lower() + '.preprocess_input'
	Object = eval(function_name + '(Object)')
	features = model.predict(Object)
	objectFeatures.append(features)
	
	gtd.append([row['observation angle'], row['zloc']])
		
	if visualize:
		cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
		string = "({}, {})".format(row['observation angle'], row['zloc'])
		cv2.putText(im, string, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

		cv2.imshow("detections", im)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

# Record features.
if not os.path.exists('features/'): os.makedirs('features/')
sio.savemat('features/features_max_' + modelName + '.mat', 
			{'objectFeatures' : objectFeatures, 'gtd' : gtd})