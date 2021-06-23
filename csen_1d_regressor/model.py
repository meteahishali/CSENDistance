import os
import tensorflow as tf
tf.random.set_seed(10)
import numpy as np

from csen_1d_regressor.utils import *

class model:
    def __init__(self):
        self.imageSizeM = 1200
        self.model = None
        self.history = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def loadData(self, feature_type, set, MR):
        # Check if the CSEN data is available.
        if not os.path.exists('CSENdata-1D/'): exit('CSENdata-1D is not prepared!')
        data = 'CSENdata-1D/' + feature_type
        dataPath = data + '_mr_' + MR + '_run' + str(set) + '.mat'
        dic_label = scipy.io.loadmat('CSENdata-1D/dic_label' + '.mat')["ans"]

        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = loadData(dataPath)

    def getModel(self):
        input_shape = (self.imageSizeM, 1)

        input = tf.keras.Input(input_shape, name='input')
        x_0 = tf.keras.layers.Conv1D(64, 25, padding = 'same', activation = 'relu')(input)
        x_0 = tf.keras.layers.MaxPooling1D(pool_size=(20))(x_0) # Sparse code shapes.
        x_0 = tf.keras.layers.Conv1D(1, 25, padding = 'same', activation = 'relu')(x_0)
        
        y = tf.keras.layers.Flatten()(x_0)
        y = tf.keras.layers.Dense(1, activation = 'softplus')(y)
        
        self.model = tf.keras.models.Model(input, y, name='CSEN')
        self.model.summary()

    def train(self, weightPath, epochs = 100, batch_size = 16):
        adam = tf.keras.optimizers.Adam(
            lr=0.001, beta_1=0.9, beta_2=0.999, 
            epsilon=None, decay=0.0, amsgrad=True)

        checkpoint_csen = tf.keras.callbacks.ModelCheckpoint(
            weightPath, monitor='val_loss', verbose=1,
            save_best_only=True, mode='min')
        
        callbacks_csen = [checkpoint_csen]

        while True:
            self.getModel()
            self.model.compile(loss = tf.compat.v1.losses.huber_loss,
                optimizer = adam, metrics=['mae', 'mse'] )
            # Training.
            self.history = self.model.fit(self.x_train, self.y_train,
                validation_data=(self.x_val, self.y_val), 
                epochs = epochs, batch_size = batch_size,
                shuffle = True, callbacks=callbacks_csen)

            if self.model.history.history['loss'][2] < 8: # If it is converged.
                break

    def load_weights(self, weightPath):
        self.getModel()
        self.model.load_weights(weightPath)

    def predict(self):
        y_pred = self.model.predict(self.x_test)
        return y_pred
######