'''
The Convolutional Support Estimator Network (CSEN) and
Compressive Learning CSEN (CL-CSEN) implementations.

The competing regrossor implementation: Support Vector Regressor (SVR).

Author: Mete Ahishali,
Tampere University, Tampere, Finland.
'''
import os
import numpy as np
import argparse
import scipy.io
import cv2
import pandas as pd
import tqdm
import pickle
import tensorflow as tf

import model

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
np.random.seed(10)

# INITIALIZATION
# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('--method', default='CL-CSEN', 
				help="Method for the regression: CL-CSEN, CL-CSEN-1D.")
ap.add_argument('--feature_type', default = 'ResNet50', help="Features extracted by the network (DenseNet121, VGG19, ResNet50).")
args = vars(ap.parse_args())
modelType = args['method']
feature_type = args['feature_type']

MR = 0.5

# Path for the data and annotations.
kittiData = '../kitti-data/'
# Note that KITTI provides annotations only for the training since it is a challenge dataset.
imagePath = kittiData + 'training/image_2/'
df = pd.read_csv(kittiData + 'annotations.csv') # Kitti dataset annotations.


# For the results.
if not os.path.exists('results/' + modelType): os.makedirs('results/' + modelType)
if modelType == 'CL-CSEN':
	with open(feature_type + '_scaler.pkl','rb') as f:
		sc = pickle.load(f)
	
	phi = scipy.io.loadmat('phi_' + feature_type + '.mat') # Run 5 measurement matrix.
	phi = phi['phi']

	if feature_type == 'DenseNet121':
		modelFold = model.get_CL_CSEN(feature_size = 512, imageSizeM = 80, imageSizeN = 15)
	elif feature_type == 'VGG19':
		modelFold = model.get_CL_CSEN(feature_size = 256, imageSizeM = 80, imageSizeN = 15)
	elif feature_type == 'ResNet50':
		modelFold = model.get_CL_CSEN(feature_size = 1024, imageSizeM = 80, imageSizeN = 15)
	outName  = 'results/CL-CSEN'

elif modelType == 'CL-CSEN-1D':
	with open(feature_type + '_1D_scaler.pkl','rb') as f:
		sc = pickle.load(f)

	phi = scipy.io.loadmat('phi_1D_' + feature_type + '.mat') # Run 1 measurement matrix.
	phi = phi['phi']

	if feature_type == 'DenseNet121':
		modelFold = model.get_CL_CSEN_1D(feature_size = 512, imageSizeM = 80, imageSizeN = 15)
	elif feature_type == 'VGG19':
		modelFold = model.get_CL_CSEN_1D(feature_size = 256, imageSizeM = 80, imageSizeN = 15)
	elif feature_type == 'ResNet50':
		modelFold = model.get_CL_CSEN_1D(feature_size = 1024, imageSizeM = 80, imageSizeN = 15)
	outName  = 'results/CL-CSEN-1D'
	

# Record weights.
weightsDir = '../weights/' + modelType + '/'
if not os.path.exists(weightsDir): print('Weights are not available.')
weightPath = weightsDir + feature_type + '_0.5_1' + '.h5'

# Load the model.
function_name = 'tf.keras.applications.' + feature_type
feature_extractor = eval(function_name + "(include_top=False, weights='imagenet', input_shape=(64, 64, 3), pooling='max')")


modelFold.load_weights(weightPath)

pbar = tqdm.tqdm(total=df.shape[0], position=1)

for idx, row in df.iterrows():
	pbar.update(1)
	if row['zloc'] < 0.5 or row['zloc'] > 60.5: # Do not perform prediction for the out of range objects.
		continue

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
	function_name = 'tf.keras.applications.' + feature_type[:8].lower() + '.preprocess_input' # Preprocessing.
	Object = eval(function_name + '(Object)')
	features = feature_extractor.predict(Object)
	features = np.matmul(phi, features.T).T # Dimensionality reduction.
	l_max = np.sqrt(np.sum(features ** 2))
	features = features/np.squeeze(l_max)
	features = sc.transform(features) # Normalization.
	 
	y = modelFold.predict(features)
		
	cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
	string = 'Predicted distance: ' + str(np.round(y[0][0], 2)) + 'm'
	
	if row['occluded'] == 0:
		string2 = 'Occlusion: fully visible' # Easy case.
	elif row['occluded'] == 1:
		string2 = 'Occlusion: partly occluded' # Medium case.
	elif row['occluded'] == 2:
		string2 = 'Occlusion: largely occluded' # Hard case.
	else:
		string2 = 'Occlusion: unknown'
	
	'''
	0 = fully visible, 1 = partly occluded
                     2 = largely occluded
	'''
	cv2.putText(im, string, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
	cv2.putText(im, string2, (int((x1+x2)/2), int((y1+y2)/2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

	cv2.imshow("detections", im)
	k = cv2.waitKey(0) # Press any key to continue with the next object.
	
	if k & 0xFF == ord('q'):
		break