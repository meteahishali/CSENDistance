from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.io
######################################## FUNCTIONS
# Data loading.
def loadData(dataPath):

    Data = scipy.io.loadmat(dataPath)

    x_dic = Data['x_dic'].astype('float32')
    x_train = Data['x_train'].astype('float32')
    x_test = Data['x_test'].astype('float32')
    y_dic = Data['dicRealLabel'].astype('float32')
    y_train = Data['trainRealLabel'].astype('float32')
    y_test = Data['testRealLabel'].astype('float32')


    print('\n\n\n')
    print('Loaded dataset:')
    print(len(x_train), ' train')
    print(len(x_test), ' test')
    
    # Partition for the validation.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1)

    # Data normalization.
    m =  x_train.shape[1]
    n =  x_train.shape[2]

    x_dic = np.reshape(x_dic, [len(x_dic), m * n])
    x_train = np.reshape(x_train, [len(x_train), m * n])
    x_val = np.reshape(x_val, [len(x_val), m * n])
    x_test = np.reshape(x_test, [len(x_test), m * n])
    
    scaler = StandardScaler().fit(np.concatenate((x_dic, x_train), axis = 0))
    x_dic = scaler.transform(x_dic)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    x_dic = np.reshape(x_dic, [len(x_dic), m, n])
    x_train = np.reshape(x_train, [len(x_train), m, n])
    x_val = np.reshape(x_val, [len(x_val), m, n])
    x_test = np.reshape(x_test, [len(x_test), m, n])

    x_train = np.concatenate((x_dic, x_train), axis = 0)
    y_train = np.concatenate((y_dic, y_train), axis = 0)


    print("\n")
    print('Partitioned.')
    print(len(x_train), ' Train')
    print(len(x_val), ' Validation')
    print(len(x_test), ' Test\n')
    print("\n\n\n")

    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_val, x_test, y_train, y_val, y_test

######################################## END FUNCTIONS