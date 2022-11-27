import intvalpy as ip
import numpy as np
import matplotlib.pyplot as plt


def line(ax, coefs, res, x, y, mod):
    if coefs[0] == 0 and coefs[1] == 0:
        return
    if coefs[1] == 0:
        ax.plot([res / coefs[0]] * len(y), y, mod)
    else:
        ax.plot(x, (res - coefs[0] * x) / coefs[1], mod)


def linear_system_plot(A, b, title):
    colors = ['r', 'g', 'b', 'k']
    x = np.linspace(-1, 5, 100)
    y = [-1, 3]
    fig, ax = plt.subplots()
    for coefs, res, color in zip(A, b, colors):
        line(ax, coefs.mid, res.mid, x, y, color + '-')
    plt.title(title)
    plt.grid()
    plt.show()


def tol_plot(A, b, title, needVe=False):
    x, y = np.mgrid[-1:5:100j, -0.5:3:35j]
    z = np.zeros(x.shape)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            z[i][j] = ip.linear.Tol(A, b, [x[i][j], y[i][j]])
    max = ip.linear.Tol(A, b, maxQ=True)

    fig, ax = plt.subplots()
    cs = ax.contour(x, y, z, levels = 20)
    fig.colorbar(cs, ax=ax)
    ax.clabel(cs)
    ax.plot(max[1][0], max[1][1], 'r*', label='Максимум ({}, {}), значение: {}'.format(max[1][0], max[1][1], max[2]))
    if needVe:
        ive = ip.linear.ive(A, b)
        rve = ive * np.linalg.norm(b.mid) / np.linalg.norm([max[1][0], max[1][1]])
        print("ive: {}".format(ive))
        print("rve: {}".format(rve))
        iveRect = plt.Rectangle((max[1][0] - ive, max[1][1] - ive), 2 * ive, 2 * ive, edgecolor='red', facecolor='none', label='Брус ive')
        plt.gca().add_patch(iveRect)
        rveRect = plt.Rectangle((max[1][0] - rve, max[1][1] - rve), 2 * rve, 2 * rve, edgecolor='blue', facecolor='none', label='Брус rve')
        plt.gca().add_patch(rveRect)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    return max[2]


def b_correction(b, K, weights):
    return b + K * ip.Interval(-1, 1) * weights


midA = np.array([[1, 2], [1, -3], [1, 0], [0, 1]])
radA = np.array([[1, 1], [0, 1], [0, 0], [0, 0]])
A = ip.Interval(midA, radA, midRadQ=True)

# b1, b2 = np.random.uniform(1, 5), np.random.uniform(1, 5)
b1, b2 = 4, 1.8
print("b values: {}, {}".format(b1, b2))
midb = np.array([5, 0, b1, b2])
radb = np.array([2, 0.5, 0.25, 0.25])
b = ip.Interval(midb, radb, midRadQ=True)

linear_system_plot(A, b, 'mid исходной СЛАУ')
maxTol = tol_plot(A, b, 'Tol для исходной системы')

weights = np.ones(len(b))
K = 1.5 * maxTol
bCorrected = b_correction(b, K, weights)
print("corrected b: {}".format(bCorrected))
maxTolBCorrected = tol_plot(A, bCorrected, 'Tol для системы со скорректированной правой частью', True)
