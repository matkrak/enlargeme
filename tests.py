import CCRBM
import DataHandler
import matplotlib.pyplot as plt


def compareMatrixScalar():
    rbms = []
    rbms.append((CCRBM.CCRBM(78, 64, 40, (5, 5), 'scalar', 'scalar'), 'SS'))
    rbms.append((CCRBM.CCRBM(78, 64, 40, (5, 5), 'scalar', 'matrix'), 'SM'))
    rbms.append((CCRBM.CCRBM(78, 64, 40, (5, 5), 'matrix', 'matrix'), 'MM'))
    rbms.append((CCRBM.CCRBM(78, 64, 40, (5, 5), 'matrix', 'scalar'), 'MS'))

    for r in rbms:
        # r[0].dh.readBrainWebData(resize=True, shape=(64, 64))
        r[0].dh.readnpy(resize=True, shape=(78, 64))
        r[0].dh.normalize()
        r[0].contrastiveDivergence(200, 1e-6, 0, 10, 10)
        r[0].saveToFile('compare2' + r[1])


def compareFiltersNo():
    rbms = []
    for i in range(6):
        rbms.append((CCRBM.CCRBM(78, 64, i * 10 + 10, (5, 5)), str(i * 10 + 10)))
        # rbms[i][0].dh.readBrainWebData(resize=True, shape=(64, 64))
        rbms[i][0].dh.readnpy(resize=True, shape=(78, 64))
        rbms[i][0].dh.normalize()
        rbms[i][0].contrastiveDivergence(200, 1e-6, 0, 10, 20)
        rbms[i][0].saveToFile('compare2' + rbms[i][1])


def compareConvFilterSize():
    rbms = []
    filters = ((3, 3), (5, 5), (7, 7), (9, 9))
    for i in range(len(filters)):
        rbms.append((CCRBM.CCRBM(78, 64, 40, filters[i]), str(filters[i][0])))
        # rbms[i][0].dh.readBrainWebData(resize=True, shape=(64, 64))
        rbms[i][0].dh.readnpy(resize=True, shape=(78, 64))
        rbms[i][0].dh.normalize()
        rbms[i][0].contrastiveDivergence(200, 1e-6, 0, 10, 20)
        rbms[i][0].saveToFile('compareConvFilters' + rbms[i][1])


def compareMiniBatchSize():
    rbms = []
    mb = [1, 10, 20, 30, 40, 50]

    for i in range(6):
        rbms.append((CCRBM.CCRBM(78, 64, 40, (5, 5)), str(mb[i])))
        # rbms[i][0].dh.readBrainWebData(resize=True, shape=(64, 64))
        rbms[i][0].dh.readnpy(resize=True, shape=(78, 64))
        rbms[i][0].dh.normalize()
        rbms[i][0].contrastiveDivergence(200, 1e-6, 0, mb[i], 10)
        rbms[i][0].saveToFile('compareMB' + rbms[i][1])


def compareLRandMomentum():
    rbms = []

    rbms.append((CCRBM.CCRBM(78, 64, 40, (5, 5)), 'LRM1'))
    rbms[-1][0].dh.readnpy(resize=True, shape=(78, 64))
    rbms[-1][0].dh.normalize()
    rbms[-1][0].contrastiveDivergence(300, 1e-6, 0, 10, 10)
    rbms[-1][0].saveToFile('compareMB' + rbms[-1][1])

    rbms.append((CCRBM.CCRBM(78, 64, 40, (5, 5)), 'LRM2'))
    rbms[-1][0].dh.readnpy(resize=True, shape=(78, 64))
    rbms[-1][0].dh.normalize()
    rbms[-1][0].contrastiveDivergence(100, 1e-7, 0, 10, 10)
    rbms[-1][0].contrastiveDivergence(100, 1e-6, 0, 10, 10)
    rbms[-1][0].contrastiveDivergence(100, 1e-6, 0, 10, 10)
    rbms[-1][0].saveToFile('compareMB' + rbms[-1][1])

    rbms.append((CCRBM.CCRBM(78, 64, 40, (5, 5)), 'LRM3'))
    rbms[-1][0].dh.readnpy(resize=True, shape=(78, 64))
    rbms[-1][0].dh.normalize()
    rbms[-1][0].contrastiveDivergence(100, 1e-7, 0.9, 10, 10)
    rbms[-1][0].contrastiveDivergence(100, 1e-6, 0.9, 10, 10)
    rbms[-1][0].contrastiveDivergence(100, 1e-7, 0.9, 10, 10)
    rbms[-1][0].saveToFile('compareMB' + rbms[-1][1])

    rbms.append((CCRBM.CCRBM(78, 64, 40, (5, 5)), 'LRM4'))
    rbms[-1][0].dh.readnpy(resize=True, shape=(78, 64))
    rbms[-1][0].dh.normalize()
    rbms[-1][0].contrastiveDivergence(100, 1e-6, 0.5, 10, 10)
    rbms[-1][0].contrastiveDivergence(100, 1e-6, 0.9, 10, 10)
    rbms[-1][0].contrastiveDivergence(100, 1e-7, 0.9, 10, 10)
    rbms[-1][0].saveToFile('compareMB' + rbms[-1][1])


def compareMSE(rbms, labels=None):
    f = plt.figure()
    if labels is not None:
        for item in zip(rbms, labels):
            plt.plot([arg[0] for arg in item[0].mse], [arg[1] for arg in item[0].mse], label=item[1])
            plt.legend()
    else:
        for item in rbms:
            plt.plot([arg[0] for arg in item.mse], [arg[1] for arg in item.mse])
    plt.xlabel('Iterations')
    plt.ylabel('BatchMSE')
    f.show()


if __name__ == '__main__':
    print('Here you can find some test methods. Import this to main and use with a logger!')
