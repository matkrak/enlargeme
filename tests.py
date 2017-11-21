import CCRBM
import DataHandler


def compareMatrixScalar():
    rbms = []
    rbms.append((CCRBM.CCRBM(64, 64, 40, (5, 5), 'scalar', 'scalar'), 'SS'))
    rbms.append((CCRBM.CCRBM(64, 64, 40, (5, 5), 'scalar', 'matrix'), 'SM'))
    rbms.append((CCRBM.CCRBM(64, 64, 40, (5, 5), 'matrix', 'matrix'), 'MM'))
    rbms.append((CCRBM.CCRBM(64, 64, 40, (5, 5), 'matrix', 'scalar'), 'MS'))

    for r in rbms:
        r[0].dh.readBrainWebData(resize=True, shape=(64, 64))
        r[0].dh.normalize()
        r[0].contrastiveDivergence(200, 2e-6, 10, 20)
        r[0].saveToFile('compare' + r[1])


def compareFiltersNo():
    rbms = []
    for i in range(5):
        rbms.append((CCRBM.CCRBM(64, 64, i * 10 + 10, (5, 5)), str(i * 10 + 10)))
        rbms[i][0].dh.readBrainWebData(resize=True, shape=(64, 64))
        rbms[i][0].dh.normalize()
        rbms[i][0].contrastiveDivergence(100, 1e-6, 10, 20)
        rbms[i][0].saveToFile('compare' + rbms[i][1])


if __name__ == '__main__':
    print('Here you can find some test methods. Import this to main and use with a logger!')