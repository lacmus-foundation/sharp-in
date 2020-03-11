from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
from pathlib import Path

class BatchGenerator(Sequence):
    
    def __init__(self, dataset_path, list_IDs, labels=[], batch_size=32, dim=(512,512), n_channels=3, shuffle=True):
        
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dataset_path = dataset_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
       
        X = np.empty((self.batch_size, *self.dim, self.n_channels)).astype('float64')
        y = np.empty((self.batch_size, *self.dim, self.n_channels)).astype('float64')
        
        for i, ID in enumerate(list_IDs_temp):
            
            X[i,] = cv2.imread(str(Path(self.dataset_path, 'X', ID)))/255.
            y[i,] = cv2.imread(str(Path(self.dataset_path, 'y', ID)))/255.

        return X, y