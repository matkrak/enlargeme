import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import pickle
import DataHandler


def flipped(matrix):
    result = np.ndarray(matrix.shape, dtype=matrix.dtype)
    for i in range(matrix.size):
        x = int(i / matrix.shape[1])
        y = i % matrix.shape[1]
        result[x][y] = matrix[matrix.shape[0] - x - 1][matrix.shape[1] - y - 1]
    return result


def sigmoid(x):
        return 1 / (1 + np.exp(-x))


def sigmoid2(x):
    """More stable but slower version of sigmoid function
    :param x: ndarray
    :return: ndarray of sigmoids
    """
    res = np.ndarray(x.shape)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            if x[i][j] > 30:
                res[i][j] = 1
            elif x[i][j] < -30:
                res[i][j] = 0
            else:
                res[i][j] = 1 / (1 + np.exp(-x[i][j]))
    return res


class CCRBM:
    def __init__(self, size_v, size_h, filters_no, conv_kernel, typeB='scalar', typeC='matrix'):
        # RBM parameters
        self.insize_v = size_v
        self.insize_h = size_h
        self.filters_no = filters_no
        self.conv_kernel = conv_kernel
        # neurons, weigths and biases
        self.v = np.ndarray((size_v, size_h), dtype=np.float32) #int32?
        self.h = [np.ndarray((size_v - conv_kernel[0] + 1, size_h - conv_kernel[1] + 1),
                             dtype=np.int8) for i in range(filters_no)]
        self.W = [np.random.normal(0, 0.01, conv_kernel) for i in range(filters_no)]

        if typeB not in ('scalar', 'matrix') or typeC not in ('scalar', 'matrix'):
            raise ValueError('Wrong input arguments. typeB and typeC must be either \'scalar\' or \'matrix\'')
        self.typeB = typeB
        self.typeC = typeC

        if self.typeB == 'scalar':
            self.b = [np.random.normal(0, 0.01) for i in range(filters_no)]
        else:
            self.b = [np.random.normal(0, 0.01, (size_v - conv_kernel[0] + 1, size_h - conv_kernel[1] + 1)) for i in range(filters_no)]

        if self.typeC == 'scalar':
            self.c = np.random.normal(0, 0.01)
        else:
            self.c = np.random.normal(0, 0.01, (size_v, size_h))

        self.dh = DataHandler.DataHandler()
        self.imgInfo = None

        self.iterations = 0
        self.mse = []

    def sample_h_given_v(self):
        for feature_map in range(self.filters_no):
            tmp = convolve2d(self.v, flipped(self.W[feature_map]), mode='valid') + self.b[feature_map]
            self.h[feature_map] = np.random.binomial(1, sigmoid(tmp))

    def sample_v_given_h(self):
        tmp = np.zeros((self.insize_v, self.insize_h))
        for feature_map in range(self.filters_no):
            tmp += convolve2d(self.h[feature_map], self.W[feature_map])
        tmp += self.c
        self.v = np.random.normal(tmp, 0.01)

    def prob_h_given_v(self):
        for feature_map in range(self.filters_no):
            self.h[feature_map] = sigmoid(convolve2d(self.v, flipped(self.W[feature_map]), mode='valid') + self.b[feature_map])

    def prob_v_given_h(self):
        tmp = np.zeros((self.insize_v, self.insize_h))
        for feature_map in range(self.filters_no):
            tmp += convolve2d(self.h[feature_map], self.W[feature_map])
        self.v = tmp + self.c

    def batchMSE(self, batchSize=None, steps=3):
        if self.dh.train is None:
            raise ValueError('Data handler was not initialised, no source for images')
        if batchSize is None:
            batchSize = self.dh.te_size
        mse = 0
        for i in range(batchSize):
            self.loadImage(i, dataset='test')
            v0 = np.copy(self.v)
            for j in range(steps):
                self.sample_h_given_v()
                self.sample_v_given_h()
            mse += ((self.v - v0)**2).mean()
        return mse/batchSize

    def contrastiveDivergence(self, iterations, lrate, batchSize=10, monitor=10):
        # bshape = (self.insize_v - self.conv_kernel[0] + 1, self.insize_h - self.conv_kernel[1] + 1)
        # cshape = (self.insize_v, self.insize_h)

        print('Starting Contrastive Divergence with following parameters:\n'\
              'iterations = {}, learnig rate = {}, batch size = {}, monitor = {}'.format(iterations, lrate, batchSize, monitor))
        imgcounter=0

        for it in range(self.iterations, self.iterations + iterations):

            dW = [np.zeros(shape=self.W[0].shape, dtype=np.float32) for i in range(self.filters_no)]
            db = [0 for i in range(self.filters_no)]
            dc = 0

            for batchidx in range(batchSize):
                if imgcounter == self.dh.tr_size:
                    print('All dataset has been used, staring from 0 again.')
                    imgcounter = 0

                self.loadImage(imgcounter)
                imgcounter += 1

                v0 = np.copy(self.v)
                #print('MSE before update: {}'.format(self.msError(image)))

                pH0 = [sigmoid(convolve2d(self.v, flipped(self.W[k]), mode='valid') + self.b[k]) for k in range(self.filters_no)]
                grad0 = [convolve2d(self.v, flipped(pH0[k]), mode='valid') for k in range(self.filters_no)]
                self.h = [np.random.binomial(1, pH0[k]) for k in range(self.filters_no)]

                self.sample_v_given_h()

                pH1 = [sigmoid(convolve2d(self.v, flipped(self.W[k]), mode='valid') + self.b[k]) for k in range(self.filters_no)]
                grad1 = [convolve2d(self.v, flipped(pH1[k]), mode='valid') for k in range(self.filters_no)]

                for k in range(self.filters_no):
                    dW[k] += (grad0[k] - grad1[k])
                    if self.typeB == 'scalar':
                        db[k] += (pH0[k] - pH1[k]).sum()
                    else:
                        db[k] += (pH0[k] - pH1[k])
                if self.typeC == 'scalar':
                    dc += (v0 - self.v).sum()
                else:
                    dc += (v0 - self.v)

            for k in range(self.filters_no):
                self.W[k] += (lrate / batchSize) * dW[k]
                self.b[k] += (lrate / batchSize) * db[k]
            self.c += (lrate / batchSize) * dc

            if not it % monitor:
                if not self.mse:
                    self.mse.append((it, self.batchMSE(steps=1)))
                elif self.mse[-1][0] != it:
                    self.mse.append((it, self.batchMSE(steps=1)))
                print('Iter: {}   MSE: {}'.format(*self.mse[-1]))
        self.iterations += iterations
        self.mse.append((self.iterations, self.batchMSE(steps=1)))
        print('Iter: {}   MSE: {}'.format(*self.mse[-1]))

    def persistantCD(self, iterations, lrate, pcdSteps=5, monitor=10):
        # bshape = (self.insize_v - self.conv_kernel[0] + 1, self.insize_h - self.conv_kernel[1] + 1)
        # cshape = (self.insize_v, self.insize_h)
        # mse = []
        print('Starting Persistant Contrastive Divergence with following parameters:\n' \
              'iterations = {}, learning rate = {}, pcd steps = {}, monitor = {}'.format(iterations, lrate, pcdSteps, monitor))
        imgcounter = 0
        for it in range(self.iterations, self.iterations + iterations):

            dW = [np.zeros(shape=self.W[0].shape, dtype=np.float32) for i in range(self.filters_no)]
            db = [0 for i in range(self.filters_no)]
            dc = 0

            if imgcounter == self.dh.tr_size:
                print('All dataset has been used, staring from 0 again.')
                imgcounter = 0

            self.loadImage(imgcounter)
            imgcounter += 1

            for pcd in range(pcdSteps):
                if pcd == 0:
                    v0 = np.copy(self.v)
                # print('MSE before update: {}'.format(self.msError(image)))

                pH0 = [sigmoid(convolve2d(v0, flipped(self.W[k]), mode='valid') + self.b[k]) for k in range(self.filters_no)]
                grad0 = [convolve2d(v0, flipped(pH0[k]), mode='valid') for k in range(self.filters_no)]
                self.h = [np.random.binomial(1, pH0[k]) for k in range(self.filters_no)]

                self.sample_v_given_h()

                pH1 = [sigmoid(convolve2d(self.v, flipped(self.W[k]), mode='valid') + self.b[k]) for k in range(self.filters_no)]
                grad1 = [convolve2d(self.v, flipped(pH1[k]), mode='valid') for k in range(self.filters_no)]

                # print('W:{} grad0:{} grad1:{}'.format(self.W[0].shape, grad0[0].shape, grad1[0].shape))
                for k in range(self.filters_no):
                    # if k ==1 and pcd == 0 : print('Iter {} delta.mean(k=1): {}, W.mean(k=1) : {}'.format(iter, delta.mean(), self.W[k].mean()))
                    dW[k] += (grad0[k] - grad1[k])
                    if self.typeB == 'scalar':
                        db[k] += (pH0[k] - pH1[k]).sum()
                    else:
                        db[k] += (pH0[k] - pH1[k])
                if self.typeC == 'scalar':
                    dc += (v0 - self.v).sum()
                else:
                    dc += (v0 - self.v)

            for k in range(self.filters_no):
                self.W[k] += (lrate / pcdSteps) * dW[k]
                self.b[k] += (lrate / pcdSteps) * db[k]
            self.c += (lrate / pcdSteps) * dc

            if not it % monitor:
                if not self.mse:
                    self.mse.append((it, self.batchMSE(steps=1)))
                elif self.mse[-1][0] != it:
                    self.mse.append((it, self.batchMSE(steps=1)))
                print('Iter: {}   MSE: {}'.format(*self.mse[-1]))
        self.iterations += iterations
        self.mse.append((self.iterations, self.batchMSE(steps=1)))
        print('Iter: {}   MSE: {}'.format(*self.mse[-1]))

    def loadV(self, image):
        if image.shape != self.v.shape:
            raise ValueError
        self.v = image
        self.imgInfo = None

    def loadImage(self, imgNo, dataset='train'):
        if dataset == 'train':
            image = self.dh.train[imgNo]
        elif dataset == 'test':
            image = self.dh.test[imgNo]
        else:
            raise ValueError

        if image.shape != self.v.shape:
            raise ValueError
        self.v = image
        self.imgInfo = (dataset, imgNo)

    def displayV(self, normalize=True):
        if normalize:
            if self.imgInfo is not None:
                if self.imgInfo[0] == 'train':
                    keys = ('means_tr', 'std_tr')
                elif self.imgInfo[0] == 'test':
                    keys = ('means_te', 'std_te')
                else:
                    raise ValueError
                im = Image.fromarray(self.v * self.dh.normalParams[keys[1]] + self.dh.normalParams[keys[0]][self.imgInfo[1]])
            else:
                print('Normalization parameters were not provided, displaying visible layer without normalization')
                im = Image.fromarray(self.v)
        else:
            im = Image.fromarray(self.v)
        im.show()

    def displayFilters(self, fshape, itpl=False):
        fig = plt.figure()

        plt.subplot(fshape[0], fshape[1], 1)
        for i in range(len(self.W)):
            plt.subplot(fshape[0], fshape[1], i + 1)
            if itpl:
                plt.imshow(self.W[i], cmap='gray', interpolation='bilinear')
            else:
                plt.imshow(self.W[i], cmap='gray')
        fig.show()

    def displayC(self):
        if self.typeC == 'scalar':
            print('C layer is a scalar! c = ' + str(self.c))
            return
        tmp = np.copy(self.c)
        tmp -= tmp.min()
        tmp = tmp * 255 / tmp.max()
        Image.fromarray(tmp).show()

    def plotMSE(self):
        if not self.mse:
            print('MSE list is empty!')

        f = plt.figure()
        plt.plot([arg[0] for arg in self.mse], [arg[1] for arg in self.mse])
        f.show()

    def saveToFile(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def getRbm(imsize1=64, imsize2=64, filters=40, cfilter=(5, 5), loadBWdata=True):
    rbm = CCRBM(imsize1, imsize2, filters, cfilter)
    if loadBWdata:
        rbm.dh.readBrainWebData(resize=True, shape=(imsize1, imsize2))
        rbm.dh.normalize()
    return rbm


def loadFromFile(filename):
    with open(filename, 'rb') as f:
        rbm = pickle.load(f)
    return rbm


if __name__ == '__main__':
    print('What can I do for you?')



