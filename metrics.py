import numpy as np

class metrics:
    def __init__(self, sets, test_size):

        self.th = np.zeros([len(sets), test_size]) # Threshold.
        self.ard = np.zeros([len(sets), ]) # Absolute relative distance.
        self.srd = np.zeros([len(sets), ]) # Squared relative distance.
        self.rmse = np.zeros([len(sets), ]) # Root mean squared error.
        self.rmseLog = np.zeros([len(sets), ]) # RMSE logarithmic.

        self.y_preds = np.zeros([len(sets), test_size]) # Predictions.
        self.y_tests = np.zeros([len(sets), test_size]) # Actual distances.

    def compute(self, set, y_pred, y_test):

        y_pred = np.squeeze(y_pred)
        y_test = np.squeeze(y_test)
    
        self.y_preds[set-1, :] = y_pred
        self.y_tests[set-1, :] = y_test


        self.ard[set-1] = np.sum(np.divide(np.absolute(y_pred - y_test), y_test)) / len(y_test)
        self.srd[set-1] = np.sum(np.divide(np.square(y_pred - y_test), y_test)) / len(y_test)
        self.th[set-1, :] = np.maximum(np.divide(y_test, y_pred), np.divide(y_pred, y_test))

        self.rmse[set-1] = np.sqrt(np.sum(np.square(y_pred - y_test)) / len(y_test))

        
        y_pred[y_pred <= 0] = 1
        
        self.rmseLog[set-1] = np.sqrt(np.sum(np.square(np.log(y_pred) - np.log(y_test))) / len(y_test))

    def display(self, set):
        print('\n\nARD: ' + str(self.ard[set-1]) )
        print('\n\nSRD: ' + str(self.srd[set-1]) )
        print('\n\nRMSE: ' + str(self.rmse[set-1]) )
        print('\n\nRMSELog: ' + str(self.rmseLog[set-1]) )