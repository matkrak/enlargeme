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
        self.W = [np.random.normal(0, 0.1, conv_kernel) for i in range(filters_no)]
        self.W[0] = np.zeros(conv_kernel)
        self.b = [np.random.normal(0, 0.1, (size_v - conv_kernel[0] + 1, size_h - conv_kernel[1] + 1)) for i in range(filters_no)]
        self.c = np.random.normal(0, 0.1, (size_v, size_h))

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

    def contrastiveDivergence(self, iterations, lrate):
        iter = 1
        print('Iter: {}   MSE: {}'.format(iter, self.BatchMSE(steps=1)))
        for image in rbm1.dh:
            iter += 1
            if iter > iterations: break
            rbm1.loadImage(image)
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
                delta = lrate * (grad0[k] - grad1[k])
                # if k ==1 and not iter % 50: print('Iter {} delta(k=1): {}'.format(iter, delta))
                self.W[k] += delta
                self.b[k] += lrate * (pH0[k] - pH1[k])
            self.c += lrate * (v0 - self.v)

            #print('MSE after update: {}'.format(self.MSError(image)))
            if not iter % 100:
                print('Iter: {}   MSE: {}'.format(iter, self.BatchMSE(steps=1)))
        if iter % 100:
            print('Iter: {}   BatchMSE: {}'.format(iter, self.BatchMSE()))

    def PersistantCD(self, iterations, pcdSteps, lrate):
        iter = 0
        print('Iter: {}   MSE: {}'.format(iter, self.BatchMSE(steps=1)))
        for image in rbm1.dh:
            iter += 1
            if iter > iterations: break
            rbm1.loadImage(image)
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

                #print('W:{} grad0:{} grad1:{}'.format(self.W[0].shape, grad0[0].shape, grad1[0].shape))
                for k in range(self.filters_no):
                    delta = lrate * (grad0[k] - grad1[k])
                    # if k ==1 and pcd == 0 : print('Iter {} delta.mean(k=1): {}, W.mean(k=1) : {}'.format(iter, delta.mean(), self.W[k].mean()))
                    self.W[k] += delta
                    self.b[k] += lrate * (pH0[k] - pH1[k])
                self.c += lrate * (v0 - self.v)

                #print('MSE after update: {}'.format(self.MSError(image)))
                # if not pcd % 10:
                #     print('pcd: {}   MSE: {}'.format(pcd, self.BatchMSE(steps=1)))
            if not iter % 50:
                print('Iter: {}   MSE: {}'.format(iter, self.BatchMSE(steps=1)))
        if iter % 50:
            print('Iter: {}   BatchMSE: {}'.format(iter, self.BatchMSE()))

    def loadImage(self, image):
        if image.shape != self.v.shape:
            raise ValueError
        self.v = image

    def displayV(self):
        im = Image.fromarray(self.v)
        im.show()


if __name__ == '__main__':

    rbm1 = CCRBM(28, 28, 16, (7, 7))
    rbm1.dh.readMNISTData(DataHandler.MNIST_TRAIN_IMAGES)
    rbm1.dh.normalize()

    Image.fromarray(rbm1.dh[0]).show()
    rbm1.loadImage(rbm1.dh[0])
    rbm1.sample_h_given_v()
    rbm1.sample_v_given_h()
    rbm1.displayV()
    #
    # print('W')
    # print(rbm1.W[0].shape)
    #
    # print('b')
    # print(rbm1.b[0].shape)
    #
    # print('c')
    # print(rbm1.c.shape)

    fig = plt.figure()
    plt.subplot(4, 4, 1)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(rbm1.W[i], cmap='gray')

    fig.show()
    input()

    # rbm1.contrastiveDivergence(1000, 2e-5)

    # rbm1.contrastiveDivergence(10000, 1e-5)
    # rbm1.contrastiveDivergence(10000, 1e-6)

    rbm1.PersistantCD(200, 50, 2e-5)
    rbm1.PersistantCD(200, 50, 2e-6)
    rbm1.PersistantCD(200, 50, 1e-6)

    rbm1.loadImage(rbm1.dh[0])
    rbm1.sample_h_given_v()
    rbm1.sample_v_given_h()
    rbm1.displayV()
    #
    # print('W')
    # print(rbm1.W[0])
    #
    # print('b')
    # print(rbm1.b[0])
    #
    # print('c')
    # print(rbm1.c)

    fig2 = plt.figure()
    plt.subplot(4, 4, 1)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(rbm1.W[i], cmap='gray')

    fig2.show()
    input()

