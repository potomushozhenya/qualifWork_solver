import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import sqrt
from scipy.optimize import fsolve
mpl.use('Qt5Agg')
k = -0.64
coef = np.sqrt(-1 * k)

def realSol(x):
    return [2*np.exp(x) - 1/np.exp(x), -np.cos(x) + 2*np.exp(x) + 1/np.exp(x)]
    #return [np.cos(x), np.sin(x)]


def f1(x, y):
    return np.cos(x) + y
    #return -y


def f2(x, y):
    return np.sin(x) + y
    #return y


class MonoImplicitSolver:

    def __init__(self):
        self.n1 = 1
        self.n2 = 1
        self.m1 = 3.
        self.m2 = 2
        self.h = np.array([], dtype=np.float64)
        self.x = np.array([], dtype=np.float64)
        self.y1 = np.array([], dtype=np.float64)
        self.y2 = np.array([], dtype=np.float64)
        self.diff = np.array([], dtype=np.float64)
        self.diffY1 = np.array([], dtype=np.float64)
        self.diffY2 = np.array([], dtype=np.float64)
        # Параметр
        x1322 = 0
        # Жесткие коэф
        self.v12 = np.array([1, ((1219*sqrt(3) - 2127)*sqrt(2) + 1747*sqrt(3) - 3048)/(18*(4*sqrt(3) - 7)*(16*sqrt(2) + 23)), (((-48*x1322 + 13)*sqrt(3) + 72*x1322 - 23)*sqrt(2) + (-72*x1322 + 17)*sqrt(3) + 108*x1322 - 30)/(6*(2*sqrt(3) - 3)*(2*sqrt(2) + 3))], dtype=np.float64)
        self.v21 = np.array([2/3 - sqrt(3)/6, ((216*sqrt(3) - 382)*sqrt(2) + 319*sqrt(3) - 564)/(6*(4*sqrt(3) - 7)*(16*sqrt(2) + 23))], dtype=np.float64)
        self.b1 = np.array([(-1 - sqrt(2))/(2*sqrt(2) + 5), 3/4, (6*sqrt(2) + 9)/(8*sqrt(2) + 20)], dtype=np.float64)
        self.b2 = np.array([1/2, 1/2], dtype=np.float64)
        self.c1 = np.array([1, (87*sqrt(2) + 124)/(96*sqrt(2) + 138), (4517*sqrt(2) + 6388)/(19164*sqrt(2) + 27102)], dtype=np.float64)
        self.c2 = np.array([1/2 - sqrt(3)/6, (5*sqrt(3) - 9)/(24*sqrt(3) - 42)], dtype=np.float64)
        self.x1 = np.array([[0, 0], [-sqrt(3)*(-3 + sqrt(2))/18, 0], [(((318936*x1322 - 99512)*sqrt(3) - 552384*x1322 + 172369)*sqrt(2) + (451050*x1322 - 140728)*sqrt(3) - 781200*x1322 + 243761)/(6*(97*sqrt(3) - 168)*(548*sqrt(2) + 775)), x1322]], dtype=np.float64)
        self.x2 = np.array([[-1/6, 0, 0], [(62*sqrt(2) + 87)/(96*sqrt(2) + 138), -1, 0]], dtype=np.float64)

    def res(self, y):
        k1 = np.array([], dtype=np.float64)
        k2 = np.array([], dtype=np.float64)
        y1 = y[0]
        y2 = y[1]
        y0_1 = self.y1[-1]
        y0_2 = self.y2[-1]
        x = self.x[-1]
        k1 = np.append(k1, f1(x + self.c1[0] * self.h[-1], (1 - self.v12[0]) * y0_2 + self.v12[0] * y2))
        k2 = np.append(k2, f2(x + self.c2[0] * self.h[-1],
                              (1 - self.v21[0]) * y0_1 + self.v21[0] * y1 + self.h[-1] * k1[0] * self.x2[0, 0]))
        k1 = np.append(k1, f1(x + self.c1[1] * self.h[-1],
                              (1 - self.v12[1]) * y0_2 + self.v12[1] * y2 + self.h[-1] * self.x1[1][0] * k2[0]))
        k2 = np.append(k2, f2(x + self.c2[1] * self.h[-1], (1 - self.v21[1]) * y0_1 + self.v21[1] * y1 + self.h[-1] * (
                    self.x2[1][0] * k1[0] + self.x2[1][1] * k1[1])))
        k1 = np.append(k1, f1(x + self.c1[2] * self.h[-1],
                              (1 - self.v12[2]) * y0_2 + self.v12[2] * y2 + self.h[-1] * np.dot(self.x1[2], k2)))
        return [y[0] - y0_1 - self.h[-1] * np.dot(self.b1, k1), y[1] - y0_2 - self.h[-1] * np.dot(self.b2, k2)]

    def solve(self, x0, xFin, y0: np.ndarray, h0: float):
        i = 0
        self.y1 = np.append(self.y1, y0[0])
        self.y2 = np.append(self.y2, y0[1])
        self.x = np.append(self.x, x0)
        self.h = np.append(self.h, h0)
        self.diff = np.append(self.diff, np.inf)
        self.diffY1 = np.append(self.diffY1, np.inf)
        self.diffY2 = np.append(self.diffY2, np.inf)

        while self.x[-1] < xFin:
            self.x = np.append(self.x, self.x[-1] + self.h[-1])
            yRK = np.array(fsolve(self.res, y0, xtol=10**-12), dtype=np.float64)
            while yRK[0] is None or yRK[1] is None:
                self.h[-1] = self.h[-1]/2
                self.x[-1] = self.x[-1] - self.h[-1]
                yRK = fsolve(self.res, y0)
            self.y1 = np.append(self.y1, yRK[0])
            self.y2 = np.append(self.y2, yRK[1])
            sol = realSol(self.x[-1])
            d = np.sqrt((self.y1[-1] - sol[0]) ** 2 + (self.y2[-1] - sol[1]) ** 2)
            self.diff = np.append(self.diff, d)
            self.diffY1 = np.append(self.diffY1, np.abs(self.y1[-1] - sol[0]))
            self.diffY2 = np.append(self.diffY2, np.abs(self.y2[-1] - sol[1]))
            i += 1
        print(i)
        return self.y1, self.y2, self.diff

    def plot(self):
        fig, axs = plt.subplots()
        axs.plot(self.x, self.diffY1)
        axs.plot(self.x, self.diffY2)
        plt.show()

    #


task = MonoImplicitSolver()
#print(task.solve( 0, 5, np.array([1, 0]), 0.001))
print(task.solve( 0, 5, np.array([1, 2]), 0.0001))
task.plot()
