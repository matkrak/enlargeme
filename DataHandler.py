import numpy as np
from PIL import Image
import struct
import os

MNIST_TRAIN_LABELS = 'data/train-labels-idx1-ubyte/data'
MNIST_TRAIN_IMAGES = 'data/train-images-idx3-ubyte/data'

class DataHandler:
    def __init__(self):
        self.images = None
        self.labels = None

        self.size = None
        self.shape = None

        self.normalizationParameters = None

        self.current_iter = None



    def __getitem__(self, item):
        return self.images[item]

    def __iter__(self):
        self.current_iter = 0
        return self

    def __next__(self):
        if self.current_iter == self.size:
            raise StopIteration
        else:
            self.current_iter += 1
            return self.images[self.current_iter - 1]


    def readMNISTData(self, img_path, lab_path=None):
        with open(img_path, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.images = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows, cols)
        self.size = self.images.shape[0]
        self.shape = (rows, cols)
        if lab_path is not None:
            with open(lab_path, 'rb') as flab:
                magic, num = struct.unpack(">II", flab.read(8))
                self.labels = np.fromfile(flab, dtype=np.int8)

    def normalize(self):
        self.images = self.images.astype(np.float32)
        self.normalizationParameters = (self.images.mean(), self.images.std())
        self.images -= self.normalizationParameters[0]
        self.images /= self.normalizationParameters[1]



if __name__ == '__main__':
    dh = DataHandler()
    dh.readMNISTData(MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS)

    i = 0
    for img in dh:
        i += 1
        print(i)

    i = 0
    for img in dh:
        i += 1
        print(i)


