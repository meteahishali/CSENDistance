from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.io
######################################## FUNCTIONS
# Data loading.
def loadData(dataPath):

    Data = scipy.io.loadmat(dataPath)

    x_train = Data['xx_train'].astype('float32')
    x_test = Data['xx_test'].astype('float32')
    y_train = Data['yy_train'].astype('float32')
    y_test = Data['yy_test'].astype('float32')

    print('\n\n\n')
    print('Loaded dataset:')
    print(len(x_train), ' train')
    print(len(x_test), ' test')

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.188, random_state = 1)

    # Data normalization.
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    print("\n")
    print('Partitioned.')
    print(len(x_train), ' Train')
    print(len(x_val), ' Validation')
    print(len(x_test), ' Test\n')
    print("\n\n\n")

    return x_train, x_val, x_test, y_train, y_val, y_test

######################################## END FUNCTIONS