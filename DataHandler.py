import numpy as np
from PIL import Image
# import struct
import scipy.io as sio

import logging
logger = logging.getLogger('root.' + __name__)
logger.addHandler(logging.NullHandler())


MNIST_TRAIN_LABELS = 'data/train-labels-idx1-ubyte/data'
MNIST_TRAIN_IMAGES = 'data/train-images-idx3-ubyte/data'
BRAINWEB_IMAGES = 'data/BrainWeb_data.mat'


class DataHandler:
    def __init__(self):
        self.train = None
        self.test = None

        self.shape = None

        self.normalParams = {}

        self.tr_size = None
        self.te_size = None

        self.current_iter = None



    # def readMNISTData(self, img_path=MNIST_TRAIN_IMAGES, lab_path=None):
    #     with open(img_path, 'rb') as fimg:
    #         magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    #         self.train = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows, cols)
    #     self.size = self.images.shape[0]
    #     self.shape = (rows, cols)
    #     if lab_path is not None:
    #         with open(lab_path, 'rb') as flab:
    #             magic, num = struct.unpack(">II", flab.read(8))
    #             self.labels = np.fromfile(flab, dtype=np.int8)

    def readBrainWebData(self, img_path=BRAINWEB_IMAGES, resize=None, shape=None, train_test_ratio=20):
        #hardcoded for now
        matfile = sio.loadmat(img_path)

        data = matfile.get('dataset_T1')
        if resize is None:
            dims = (data.shape[2], data.shape[0], data.shape[1])
        else:
            if shape is None:
                logger.error('[readBrainWebData] When resize is not None size arg must be provided!')
                raise ValueError('When resize is not None size arg must be provided!')
            dims = (data.shape[2], shape[0], shape[1])

        self.te_size = int(data.shape[2]/(train_test_ratio + 1))
        self.tr_size = data.shape[2] - self.te_size

        self.train = []
        self.test = []

        shuffling = list(range(data.shape[2]))

        img_no = 0
        if resize is None:
            while img_no < self.tr_size:
                self.train.append(data[:, :, shuffling.pop(np.random.randint(0, len(shuffling)))])
                img_no += 1

            while img_no < dims[0]:
                self.test.append(data[:, :, shuffling.pop(np.random.randint(0, len(shuffling)))])
                img_no += 1
        else:
            while img_no < self.tr_size:
                self.train.append(np.asarray(Image.fromarray(data[:, :, shuffling.pop(np.random.randint(0, len(shuffling)))]).resize(shape)))
                img_no += 1

            while img_no < dims[0]:
                self.test.append(np.asarray(Image.fromarray(data[:, :, shuffling.pop(np.random.randint(0, len(shuffling)))]).resize(shape)))
                img_no += 1

        self.shape = (dims[1], dims[2])

        print('Loaded Brainweb data. Image size: ' + str(self.shape))
        print('Trainset size: ' + str(self.tr_size))
        print('Tetset size:   ' + str(self.te_size))
        logger.info('Loaded Brainweb data. Image size: {}, train set size: {}, test set size: {}'.format(self.shape, self.tr_size, self.te_size))

    def normalize(self):

        tmp_tr = np.ndarray(shape=(self.tr_size, *self.shape))
        tmp_te = np.ndarray(shape=(self.te_size, *self.shape))
        self.normalParams['means_tr'] = []
        self.normalParams['means_te'] = []

        # remember mean for each image and subtract it!
        for i in range(self.tr_size):
            self.normalParams['means_tr'].append(self.train[i].mean())
            self.train[i] = self.train[i] - self.normalParams['means_tr'][i]
        for i in range(self.te_size):
            self.normalParams['means_te'].append(self.test[i].mean())
            self.test[i] = self.test[i] - self.normalParams['means_te'][i]

        # compute std for each dataset, remember it and divide!
        for i in range(self.tr_size):
            tmp_tr[i, :, :] = np.copy(self.train[i])
        self.normalParams['std_tr'] = tmp_tr.std()

        for i in range(self.tr_size):
            self.train[i] = self.train[i] / self.normalParams['std_tr']

        for i in range(self.te_size):
            tmp_te[i, :, :] = np.copy(self.test[i])
        self.normalParams['std_te'] = tmp_te.std()

        for i in range(self.te_size):
            self.test[i] = self.test[i] / self.normalParams['std_te']

        logger.info('Successfully normalized data from DataHandler object.')

if __name__ == '__main__':
    print('Tell me what to do')




