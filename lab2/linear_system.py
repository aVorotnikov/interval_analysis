import intvalpy as ip
import numpy as np
import matplotlib.pyplot as plt


def tol_plot(A, b, title):
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
    ax.plot(max[1][0], max[1][1], 'r*', label='Максимум, значение: {}'.format(max[2]))
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


midA = np.array([[1, 2], [1, -3], [1, 0], [0, 1]])
radA = np.array([[1, 1], [0, 1], [0, 0], [0, 0]])
A = ip.Interval(midA, radA, midRadQ=True)

# b1, b2 = np.random.uniform(1, 5), np.random.uniform(1, 5)
b1, b2 = 4, 1.8
print("b values: {}, {}".format(b1, b2))
midb = np.array([5, 0, b1, b2])
radb = np.array([2, 0.5, 0.25, 0.25])
b = ip.Interval(midb, radb, midRadQ=True)

x = np.linspace(-1, 5, 100)
fig, ax = plt.subplots()
ax.plot(x, (5 - x) / 2, 'r-')
ax.plot(x, x / 3, 'g-')
ax.plot([b1, b1], [-0.5, 3], 'b-')
ax.plot([b2, b2], [-0.5, 3], 'k-')
plt.title("СЛАУ, образованная mid")
plt.grid()
plt.show()

tol_plot(A, b, "Немодифицированная система")
