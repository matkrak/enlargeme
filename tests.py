import CCRBM
from PIL import Image
import matplotlib.pyplot as plt


def testRun():
    rbm1 = CCRBM.getRbm()

    Image.fromarray(rbm1.dh[0]).show()
    rbm1.loadImage(rbm1.dh[0])
    rbm1.sample_h_given_v()
    rbm1.sample_v_given_h()
    rbm1.displayV()

    for i in range(2):
        rbm1.PersistantCD(rbm1.dh.size,  1e-7, 10)
    for i in range(2):
        rbm1.PersistantCD(rbm1.dh.size, 1e-6, 5)
    for i in range(2):
        rbm1.PersistantCD(rbm1.dh.size, 1e-7, 3)

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
    rbm1 = CCRBM.CCRBM(64, 64, 40, (5, 5))
    rbm1.dh.readBrainWebData(resize=True, size=(64, 64))
    rbm1.dh.normalize()

    rbm1.contrastiveDivergence(50, 1e-6, 10)
    return rbm1 #for ipython testing purpose

def pcdBatchTest():
    rbm1 = CCRBM.CCRBM(64, 64, 40, (5, 5))
    rbm1.dh.readBrainWebData(resize=True, size=(64, 64))
    rbm1.dh.normalize()

    rbm1.PersistantCD(50, 1e-7, 10)
    rbm1.PersistantCD(100, 1e-6, 10)
    rbm1.PersistantCD(50, 1e-7, 10)

    rbm1.displayFilters((8, 5), False)
    rbm1.displayFilters((8, 5), True)

    return rbm1 #for ipython testing purpose




if __name__ == '__main__':
    print('Use me in Ipython or tell me what to do')
