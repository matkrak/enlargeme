import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import struct
import scipy.io as sio

import logging
logger = logging.getLogger('root.' + __name__)
logger.addHandler(logging.NullHandler())


MNIST_TRAIN_LABELS = 'data/train-labels-idx1-ubyte/data'
MNIST_TRAIN_IMAGES = 'data/train-images-idx3-ubyte/data'
BRAINWEB_IMAGES = 'data/BrainWeb_data.mat'

BRAINWEB_NPY = 'data/t1_filtered.npy'

class DataHandler:
    def __init__(self):
        self.train = None
        self.test = None

        self.shape = None

        self.normalParams = dict()
        self.normalParams['normalized'] = None

        self.tr_size = None
        self.te_size = None

        self.current_iter = None

    def __str__(self):
        res = "images shape: (H, W) = {}, train set # {}, test set # {}, normalized: {}".format(self.shape,
                                                                                                   self.tr_size,
                                                                                                   self.te_size,
                                                                                                   self.normalParams['normalized'])
        return res



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
        """
        Read BrainWeb data mat file. Create test and train sets.
        :param img_path: Path to mat file
        :param resize: resize data?
        :param shape: tuple, shape should be (Height, Width).
        :param train_test_ratio: how many images in train/test set?
        """
        #hardcoded for now
        matfile = sio.loadmat(img_path)

        data = matfile.get('dataset_T1')
        if resize is not True:
            dims = (data.shape[2], data.shape[0], data.shape[1])
        else:
            if shape is None:
                logger.error('[readBrainWebData] When resize is not None shape arg must be provided!')
                raise ValueError('When resize is not None shape arg must be provided!')
            dims = (data.shape[2], shape[0], shape[1])

        self.te_size = int(data.shape[2]/(train_test_ratio + 1))
        self.tr_size = data.shape[2] - self.te_size

        self.train = []
        self.test = []

        shuffling = list(range(data.shape[2]))

        img_no = 0
        if resize is not True:
            while img_no < self.tr_size:
                self.train.append(data[:, :, shuffling.pop(np.random.randint(0, len(shuffling)))])
                img_no += 1

            while img_no < dims[0]:
                self.test.append(data[:, :, shuffling.pop(np.random.randint(0, len(shuffling)))])
                img_no += 1
        else:
            while img_no < self.tr_size:
                self.train.append(np.asarray(Image.fromarray(data[:, :, shuffling.pop(np.random.randint(0, len(shuffling)))]).resize((shape[1], shape[0]))))
                img_no += 1

            while img_no < dims[0]:
                self.test.append(np.asarray(Image.fromarray(data[:, :, shuffling.pop(np.random.randint(0, len(shuffling)))]).resize((shape[1], shape[0]))))
                img_no += 1

        self.shape = (dims[1], dims[2])
        self.normalParams['normalized'] = False

        print('Loaded Brainweb data. Image shape: (H, W) = ' + str(self.shape))
        print('Trainset size: ' + str(self.tr_size))
        print('Tetset size:   ' + str(self.te_size))
        logger.info('Loaded Brainweb data. Image shape: (H, W) = {}, train set size: {}, test set size: {}'.format(self.shape, self.tr_size, self.te_size))

    def readnpy(self, img_path=BRAINWEB_NPY, resize=None, shape=None, train_test_ratio=20): # TODO double check normalization (make it +-1)
        data = np.load(img_path)

        if resize is not True:
            dims = data.shape
        else:
            if shape is None:
                logger.error('[readnpy] When resize is not None shape arg must be provided!')
                raise ValueError('When resize is not None shape arg must be provided!')
            dims = (data.shape[0], shape[0], shape[1])

        self.te_size = int(dims[0] / (train_test_ratio + 1))
        self.tr_size = dims[0] - self.te_size

        shuffling = list(range(dims[0]))
        self.train = []
        self.test = []

        if resize is not True:
            for i in range(self.tr_size):
                self.train.append(data[shuffling.pop(np.random.randint(0, len(shuffling))), :, :])
            for i in range(self.te_size):
                self.test.append(data[shuffling.pop(np.random.randint(0, len(shuffling))), :, :])

        else:
            for i in range(self.tr_size):
                self.train.append(np.asarray(Image.fromarray(data[shuffling.pop(np.random.randint(0, len(shuffling))), :, :]).resize((shape[1], shape[0]))))
            for i in range(self.te_size):
                self.test.append(np.asarray(Image.fromarray(data[shuffling.pop(np.random.randint(0, len(shuffling))), :, :]).resize((shape[1], shape[0]))))


        self.shape = (dims[1], dims[2])
        self.normalParams['normalized'] = False

        print('Loaded numpy data. Image shape: (v, h) = ' + str(self.shape))
        print('Train set size: ' + str(self.tr_size))
        print('Tet set size:   ' + str(self.te_size))
        logger.info(
            'Loaded numpy data. Image shape: (v, h) = {}, train set size: {}, test set size: {}'.format(self.shape,
                                                                                                   self.tr_size,
                                                                                                   self.te_size))

    def normalize(self, force=False):
        if self.normalParams['normalized'] and not force:
            raise RuntimeWarning('Data already normalized! Use force is necessary ;)')

        # tmp_tr = np.ndarray(shape=(self.tr_size, *self.shape))    # * expressions wont work with lower python verions
        # tmp_te = np.ndarray(shape=(self.te_size, *self.shape))
        tmp_tr = np.ndarray(shape=(self.tr_size, self.shape[0], self.shape[1]))
        tmp_te = np.ndarray(shape=(self.te_size, self.shape[0], self.shape[1]))

        self.normalParams['means_tr'] = []
        self.normalParams['means_te'] = []

        # remember mean for each image and subtract it!
        for i in range(self.tr_size):
            self.normalParams['means_tr'].append(self.train[i].mean())
            self.train[i] = self.train[i] - self.normalParams['means_tr'][i]
        for i in range(self.te_size):
            self.normalParams['means_te'].append(self.test[i].mean())
            self.test[i] = self.test[i] - self.normalParams['means_te'][i]

        # compute std for each data set, remember it and divide!
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

        self.normalParams['normalized'] = True
        logger.info('Successfully normalized data object.')

    def getimg(self, idx, dataset='train', normalize=False):
        if dataset not in ('train', 'test'):
            raise ValueError('Only test or train set here')

        usetrainset = True if dataset == 'train' else False
        im = np.copy(self.train[idx] if usetrainset else self.test[idx])

        if normalize and self.normalParams['normalized']:
                im *= self.normalParams['std_tr' if usetrainset else 'std_tr']
                im += self.normalParams['means_tr' if usetrainset else 'means_te'][idx]

        return im

    def imshow(self, idx, dataset='train', normalize=True):
        Image.fromarray(np.ndarray.astype(self.getimg(idx, dataset, normalize), 'int8')).show()

    def displayImages(self, dataset='train', grid=(3, 3), normalize=True, startidx=0, title=True):
        if (self.train if dataset == 'train' else self.test) is None:
            raise RuntimeWarning('Trying to display from empty dh')
        f = plt.figure()
        for i in range(grid[0] * grid[1]):
            if startidx + i >= (self.tr_size if dataset=='train' else self.te_size):
                logger.warning('[DataHandler.displayImages] index out of range')
                raise RuntimeWarning('[DataHandler.displayImages] index out of range')
                break
            plt.subplot(*grid, i + 1)
            plt.imshow(Image.fromarray(self.getimg(startidx + i, dataset, normalize)))
            if title:
                plt.title('#' + str(startidx + i))
        f.show()





if __name__ == '__main__':
    print('Tell me what to do')




