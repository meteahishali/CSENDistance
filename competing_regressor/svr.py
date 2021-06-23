import os
import numpy as np
import tensorflow as tf
tf.random.set_seed(10)
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_approximation import Nystroem

from competing_regressor.utils import *

class model:
    def __init__(self):
       
        self.model = None
        self.x_train = None
        self.x_test = None
        self.x_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.n_componenets = None

    def loadData(self, feature_type, set, MR):
        
        # Check if the data is available.
        if not os.path.exists('competing_splits/'): exit('competing_splits is not prepared!')
        data = 'competing_splits/' + feature_type
        dataPath = data + '_mr_' + MR + '_run' + str(set) + '.mat'

        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = loadData(dataPath)
        
        self.n_componenets = int(np.round(self.x_train.shape[1] * float(MR)))

        '''
        # Perform PCA for dimensionality reduction.
        print('Applying PCA.')
        pca = PCA(n_components = int(np.round(self.x_train.shape[1] * float(MR))))
        pca.fit(self.x_train)
        self.x_train = pca.transform(self.x_train)
        self.x_val = pca.transform(self.x_val)
        self.x_test = pca.transform(self.x_test)
        '''

    def train(self, weightPath, epochs = 100, batch_size = 16):

        print('SVR parameter search.')

        kernel = {'linear', 'rbf', 'poly'}
        gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        degree = [2, 3, 4]
        C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

        bestScore = -1e3
        bestParam = {'kernel':None, 'gamma':None, 'degree':None, 'C':None}

        n_components = self.n_componenets

        n_jobs = int(os.cpu_count() / 8)
        
        for k in kernel: # Loop through kernels.
            if k is 'linear':
                for c in C:
                    print('\nKernel: ' + k + ', C: ' + str(c))
                    clf = LinearSVR(C = c, random_state=1, loss='squared_epsilon_insensitive', dual=False)
                    clf.fit(self.x_train, np.squeeze(self.y_train))
                    score = clf.score(self.x_val, np.squeeze(self.y_val))
                    print('Score achieved: ' + str(score))

                    if score > bestScore:
                        bestScore = score
                        self.model = clf
                        print('***Best score achieved: ' + str(bestScore))
                        bestParam['kernel'] = k
                        bestParam['gamma'] = None
                        bestParam['degree'] = None
                        bestParam['C'] = c

                continue

            # If not linear.
            for g in gamma: # Loop through gammas.

                if k is 'poly':
                    for d in degree: # Loop through the degrees.
                        feat_map = Nystroem(kernel = k, gamma = g, degree = d,
                                            n_jobs = n_jobs, n_components=n_components, random_state=1)
                        feat_map.fit(self.x_train)
                        x_train_trans = feat_map.transform(self.x_train)
                        x_val_trans = feat_map.transform(self.x_val)

                        for c in C:
                            print('\nKernel: ' + k + ', Degree: ' + str(d) + ', Gamma: ' + str(g) + ', C: ' + str(c))
                            clf = LinearSVR(C = c, random_state=1, loss='squared_epsilon_insensitive', dual=False)
                            clf.fit(x_train_trans, np.squeeze(self.y_train))
                            score = clf.score(x_val_trans, np.squeeze(self.y_val))
                            print('Score achieved: ' + str(score))

                            if score > bestScore:
                                bestScore = score
                                self.model = clf
                                print('***Best score achieved: ' + str(bestScore))
                                bestParam['kernel'] = k
                                bestParam['gamma'] = g
                                bestParam['degree'] = d
                                bestParam['C'] = c

                    
                    continue

                # If not a  polynomial kernel.
                feat_map = Nystroem(kernel = k, gamma = g,
                                    n_jobs = n_jobs, n_components=n_components, random_state=1)
                feat_map.fit(self.x_train)
                x_train_trans = feat_map.transform(self.x_train)
                x_val_trans = feat_map.transform(self.x_val)

                for c in C:
                    print('\nKernel: ' + k + ', Gamma: ' + str(g) + ', C: ' + str(c))
                    clf = LinearSVR(C = c, random_state=1, loss='squared_epsilon_insensitive', dual=False)
                    clf.fit(x_train_trans, np.squeeze(self.y_train))
                    score = clf.score(x_val_trans, np.squeeze(self.y_val))
                    print('Score achieved: ' + str(score))

                    if score > bestScore:
                        bestScore = score
                        self.model = clf
                        print('***Best score achieved: ' + str(bestScore))
                        bestParam['kernel'] = k
                        bestParam['gamma'] = g
                        bestParam['degree'] = None
                        bestParam['C'] = c



        print('\nFound best parameters:')        
        print(bestParam)

        print('SVR Train')
        if bestParam['kernel'] is not 'linear':
            feat_map = Nystroem(kernel = bestParam['kernel'], degree = bestParam['degree'],
                                gamma = bestParam['gamma'], n_jobs = n_jobs,
                                n_components=n_components, random_state=1)
            feat_map.fit(self.x_train)
            self.x_train = feat_map.transform(self.x_train)
            self.x_test = feat_map.transform(self.x_test)
        
        self.model = LinearSVR(C = bestParam['C'], random_state=1, loss='squared_epsilon_insensitive', dual=False)
        self.model.fit(self.x_train, np.squeeze(self.y_train))
        print('SVR Train Finished')

        '''
        self.model = SVR(kernel = bestParam['kernel'], degree = bestParam['degree'],
                        C = bestParam['C'], gamma = bestParam['gamma'], random_state = 1)
        '''
        

    def load_weights(self, weightPath):
        #This is only for network based methods.
        print('Skipping loading weights for the SVR method.')

    def predict(self):
        if self.model is None:
            exit('\n **Error: This is only for network based methods. For SVR methods, set --weights False.')
        y_pred = np.expand_dims(self.model.predict(self.x_test), axis = -1)

        return y_pred
######