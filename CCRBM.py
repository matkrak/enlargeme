import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
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


class CCRBM:
    def __init__(self, size_v, size_h, filters_no, conv_kernel):
        # RBM parameters
        self.insize_v = size_v
        self.insize_h = size_h
        self.filters_no = filters_no
        self.conv_kernel = conv_kernel
        # neurons, weigths and biases
        self.v = np.ndarray((size_v, size_h), dtype=np.float32) #int32?
        self.h = [np.ndarray((size_v - conv_kernel[0] + 1, size_h - conv_kernel[1] + 1),
                             dtype=np.int8) for i in range(filters_no)]
        self.W = [np.random.normal(0, 0.05, conv_kernel) for i in range(filters_no)]
        #self.W[0] = np.zeros(conv_kernel)
        #self.b = [np.random.normal(0, 0.05, (size_v - conv_kernel[0] + 1, size_h - conv_kernel[1] + 1)) for i in range(filters_no)]
        self.b = [np.random.normal(0, 0.05, (size_v - conv_kernel[0] + 1, size_h - conv_kernel[1] + 1)) for i in range(filters_no)]
        self.c = np.random.normal(0, 0.05, (size_v, size_h))

        self.dh = DataHandler.DataHandler()

    def sample_h_given_v(self):
        for feature_map in range(self.filters_no):
            tmp =  convolve2d(self.v, flipped(self.W[feature_map]), mode='valid') + self.b[feature_map]
            self.h[feature_map] = np.random.binomial(1, sigmoid(tmp))


    def sample_v_given_h(self):
        tmp = np.zeros((self.insize_v, self.insize_h))
        for feature_map in range(self.filters_no):
            tmp += convolve2d(self.h[feature_map], self.W[feature_map])
        tmp += self.c
        self.v = np.random.normal(tmp, 0.01)

    def MSError(self, image=None, steps=3):
        if image is not None:
            self.loadImage(image)

        v0 = np.copy(self.v)
        for i in range(steps):
            self.sample_h_given_v()
            self.sample_v_given_h()
        return ((self.v - v0)**2).sum()

    def BatchMSE(self, batchSize=10, steps=3):
        if self.dh.images is None:
            raise ValueError('Data handler was not initialised, no source for images')
        mse = 0
        for i in range(batchSize):
            self.loadImage(self.dh[np.random.randint(0, self.dh.size)])
            v0 = np.copy(self.v)
            for j in range(steps):
                self.sample_h_given_v()
                self.sample_v_given_h()
            mse += ((self.v - v0)**2).sum()
        return mse/batchSize

    def contrastiveDivergence(self, iterations, lrate, batchSize=10):
        bshape = (self.insize_v - self.conv_kernel[0] + 1, self.insize_h - self.conv_kernel[1] + 1)
        cshape = (self.insize_v, self.insize_h)
        print('Start    MSE: {}'.format(self.BatchMSE(steps=1)))
        imgcounter=0
        for iter in range(iterations):

            dW = [np.zeros(shape=self.W[0].shape, dtype=np.float32) for i in range(self.filters_no)]
            db = [np.zeros(shape=(self.insize_v - self.conv_kernel[0] + 1, self.insize_h - self.conv_kernel[1] + 1),
                           dtype=np.float32) for i in range(self.filters_no)]
            dc = np.zeros(shape=cshape)

            for batchidx in range(batchSize):
                if imgcounter == self.dh.size:
                    print('All dataset has been used, staring from 0 again.')
                    imgcounter = 0

                self.loadImage(self.dh[imgcounter])
                imgcounter += 1

                v0 = np.copy(self.v)
                #print('MSE before update: {}'.format(self.MSError(image)))

                pH0 = [sigmoid(convolve2d(self.v, flipped(self.W[k]), mode='valid') + self.b[k]) for k in range(self.filters_no)]
                grad0 = [convolve2d(self.v, flipped(pH0[k]), mode='valid') for k in range(self.filters_no)]
                self.h = [np.random.binomial(1, pH0[k]) for k in range(self.filters_no)]

                self.sample_v_given_h()

                pH1 = [sigmoid(convolve2d(self.v, flipped(self.W[k]), mode='valid') + self.b[k]) for k in range(self.filters_no)]
                grad1 = [convolve2d(self.v, flipped(pH1[k]), mode='valid') for k in range(self.filters_no)]

                #print('W:{} grad0:{} grad1:{}'.format(self.W[0].shape, grad0[0].shape, grad1[0].shape))
                for k in range(self.filters_no):
                    # if k ==1 and not iter % 50: print('Iter {} delta(k=1): {}'.format(iter, delta))
                    dW[k] += (grad0[k] - grad1[k])
                    db[k] += (pH0[k] - pH1[k])
                dc += (v0 - self.v)

            for k in range(self.filters_no):
                self.W[k] += (lrate / batchSize) * dW[k]
                self.b[k] += (lrate / batchSize) * db[k]
            self.c += (lrate / batchSize) * dc

            #print('MSE after update: {}'.format(self.MSError(image)))
            if not iter % 10:
                print('Iter: {}   MSE: {}'.format(iter, self.BatchMSE(steps=1)))

    def PersistantCD(self, iterations, pcdSteps, lrate, monitor=10):
        bshape = (self.insize_v - self.conv_kernel[0] + 1, self.insize_h - self.conv_kernel[1] + 1)
        cshape = (self.insize_v, self.insize_h)
        print('Start   MSE: {}'.format(self.BatchMSE(steps=1)))
        imgcounter = 0
        for iter in range(iterations):

            dW = [np.zeros(shape=self.W[0].shape, dtype=np.float32) for i in range(self.filters_no)]
            db = [np.zeros(shape=(self.insize_v - self.conv_kernel[0] + 1, self.insize_h - self.conv_kernel[1] + 1),
                           dtype=np.float32) for i in range(self.filters_no)]
            dc = np.zeros(shape=cshape)

            if imgcounter == self.dh.size:
                print('All dataset has been used, staring from 0 again.')
                imgcounter = 0

            self.loadImage(self.dh[imgcounter])
            imgcounter += 1

            for pcd in range(pcdSteps):
                if pcd == 0:
                    v0 = np.copy(self.v)
                #print('MSE before update: {}'.format(self.MSError(image)))

                pH0 = [sigmoid(convolve2d(v0, flipped(self.W[k]), mode='valid') + self.b[k]) for k in range(self.filters_no)]
                grad0 = [convolve2d(v0, flipped(pH0[k]), mode='valid') for k in range(self.filters_no)]
                self.h = [np.random.binomial(1, pH0[k]) for k in range(self.filters_no)]

                self.sample_v_given_h()

                pH1 = [sigmoid(convolve2d(self.v, flipped(self.W[k]), mode='valid') + self.b[k]) for k in range(self.filters_no)]
                grad1 = [convolve2d(self.v, flipped(pH1[k]), mode='valid') for k in range(self.filters_no)]

                # print('W:{} grad0:{} grad1:{}'.format(self.W[0].shape, grad0[0].shape, grad1[0].shape))
                for k in range(self.filters_no):
                    #if k ==1 and pcd == 0 : print('Iter {} delta.mean(k=1): {}, W.mean(k=1) : {}'.format(iter, delta.mean(), self.W[k].mean()))
                    dW[k] += (grad0[k] - grad1[k])
                    db[k] += (pH0[k] - pH1[k])
                dc += (v0 - self.v)

            for k in range(self.filters_no):
                self.W[k] += (lrate / pcdSteps) * dW[k]
                self.b[k] += (lrate / pcdSteps) * db[k]
            self.c += (lrate / pcdSteps) * dc

            if not iter % monitor:
                print('Iter: {}   MSE: {}'.format(iter, self.BatchMSE(steps=1)))
        if iter % monitor:
            print('Iter: {}   BatchMSE: {}'.format(iter, self.BatchMSE(steps=1)))

    def loadImage(self, image):
        if image.shape != self.v.shape:
            raise ValueError
        self.v = image

    def displayV(self, normalize=True):
        if normalize:
            im = Image.fromarray(self.v * self.dh.normalizationParameters[1] + self.dh.normalizationParameters[0])
        else:
            im = Image.fromarray(self.v)
        im.show()

    def saveToFile(self, filename):
        toBeSaved = []
        toBeSaved.append(self.insize_v)
        toBeSaved.append(self.insize_h)
        toBeSaved.append(self.filters_no)
        toBeSaved.append(self.conv_kernel[0])
        toBeSaved.append(self.conv_kernel[1])

        np.save(filename + 'META', toBeSaved)
        np.save(filename + 'W', self.W)
        np.save(filename + 'B', self.b)
        np.save(filename + 'C', self.c)


def loadCcrbmFromFile(filename):
    data = np.load(filename + 'META.npy')

    rbm = CCRBM(data[0], data[1], data[2], (data[3], data[4]))
    rbm.W = np.load(filename + 'W.npy')
    rbm.b = np.load(filename + 'B.npy')
    rbm.c = np.load(filename + 'C.npy')

    return rbm

def displayFilters(rbm, fshape, itpl=False):
    fig = plt.figure()

    plt.subplot(fshape[0], fshape[1], 1)
    for i in range(len(rbm.W)):
        plt.subplot(fshape[0], fshape[1], i + 1)
        if itpl:
            plt.imshow(rbm.W[i], cmap='gray', interpolation='bilinear')
        else:
            plt.imshow(rbm.W[i], cmap='gray')
    fig.show()

def testRun():
    rbm1 = CCRBM(64, 64, 40, (5, 5))
    rbm1.dh.readBrainWebData(resize=True, size=(64, 64))
    rbm1.dh.normalize()

    Image.fromarray(rbm1.dh[0]).show()
    rbm1.loadImage(rbm1.dh[0])
    rbm1.sample_h_given_v()
    rbm1.sample_v_given_h()
    rbm1.displayV()

    for i in range(2):
        rbm1.PersistantCD(rbm1.dh.size, 10, 1e-7)
    for i in range(2):
        rbm1.PersistantCD(rbm1.dh.size, 3, 1e-6)
    for i in range(2):
        rbm1.PersistantCD(rbm1.dh.size, 3, 1e-7)

    #rbm1.contrastiveDivergence(100, 1e-7)

    rbm1.loadImage(rbm1.dh[0])
    rbm1.sample_h_given_v()
    rbm1.sample_v_given_h()
    rbm1.displayV()

    fig = plt.figure()
    plt.subplot(6, 5, 1)
    for i in range(30):
        plt.subplot(6, 5, i + 1)
        plt.imshow(rbm1.W[i], cmap='gray')
    fig.show()

    fig2 = plt.figure()
    plt.subplot(6, 5, 1)
    for i in range(30):
        plt.subplot(6, 5, i + 1)
        plt.imshow(rbm1.W[i], cmap='gray', interpolation='bilinear')
    fig2.show()

    input()
    return rbm1

def cdBatchTest():
    rbm1 = CCRBM(64, 64, 40, (5, 5))
    rbm1.dh.readBrainWebData(resize=True, size=(64, 64))
    rbm1.dh.normalize()

    rbm1.contrastiveDivergence(50, 1e-6, 10)
    return rbm1 #for ipython testing purpose

def pcdBatchTest():
    rbm1 = CCRBM(64, 64, 40, (5, 5))
    rbm1.dh.readBrainWebData(resize=True, size=(64, 64))
    rbm1.dh.normalize()

    rbm1.PersistantCD(50, 10, 1e-7)
    rbm1.PersistantCD(100, 5, 1e-6)
    rbm1.PersistantCD(50, 10, 1e-7)

    displayFilters(rbm1, (8, 5), False)
    displayFilters(rbm1, (8, 5), True)

    return rbm1 #for ipython testing purpose

if __name__ == '__main__':

    testRun()


