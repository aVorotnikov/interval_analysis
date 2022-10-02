import numpy as np
import intvalpy as ip
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm


def glob_opt(func, eps, x0):
    yList = []
    y = x0
    fY = func(x0)
    L = [(y, fY)]
    while fY.wid >= eps:
        l = 0
        for i in range(1, len(y)):
            if y[i].wid > y[l].wid:
                l = i
        y1 = deepcopy(y)
        y1[l] = ip.Interval(y[l].a, y[l].mid)
        fY1 = func(y1)
        y2 = deepcopy(y)
        y2[l] = ip.Interval(y[l].mid, y[l].b)
        fY2 = func(y2)
        L = L[1:]
        L.append((y1, fY1))
        L.append((y2, fY2))
        L.sort(key=lambda tup : tup[1].a)
        yList.append(y.mid)
        y, fY = L[0]
    yList.append(y.mid)
    return y, yList


def rosenbrock(x):
    sum = ip.Interval(0, 0)
    for i in range(0, len(x) - 1):
        sum = sum + (100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]))
    return sum

def rastrigin(x):
    sum = ip.Interval(10 * len(x), 0, midRadQ=True)
    for i in range(0, len(x)):
        sum = sum + x[i] * x[i] - 10 * np.cos(2 * np.pi * x[i])
    return sum

def rastrigin1(x, y):
    return 10 * 2 + x * x - 10 * np.cos(2 * np.pi * x) + y * y - 10 * np.cos(2 * np.pi * y)

def himmelblau(x):
    return (x[0] * x[0] + x[1] - 11) * (x[0] * x[0] + x[1] - 11) + (x[0] + x[1] * x[1] - 7) * (x[0] + x[1] * x[1] - 7)

def himmelblau1(x, y):
    return (x * x + y - 11) * (x * x + y - 11) + (x + y * y - 7) * (x + y * y - 7)


mid = np.ones(2)
rad = np.ones(2) * 2
y_opt, list = glob_opt(rastrigin, 0.01, ip.Interval(mid, rad, midRadQ=True))

x, y = np.mgrid[-2:2:100j, -2:2:100j]
z = rastrigin1(x, y)
fig, ax = plt.subplots()

ax.contour(x, y, z, levels = 20)
ax.plot([l[0] for l in list], [l[1] for l in list], 'k', label='Центры брусов')
ax.plot([0], [0], 'r*', label='Минимум')

plt.title('Функция Растрыгина')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title('Функция Растрыгина')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.contour(x, y, z)
ax.plot([l[0] for l in list], [l[1] for l in list], [rastrigin1(float(l[0]), float(l[1])) for l in list], 'k', label='Центры брусов')
ax.plot([0], [0], [rastrigin1(0, 0)], 'r*', label='Минимум')
plt.title('Функция Растрыгина')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()


mid = np.ones(2)
rad = np.ones(2) * 2
y_opt, list = glob_opt(himmelblau, 0.01, ip.Interval(mid, rad, midRadQ=True))

x, y = np.mgrid[-4:4:400j, -4:4:400j]
z = himmelblau1(x, y)
fig, ax = plt.subplots()

ax.contour(x, y, z, levels = 20)
ax.plot([l[0] for l in list], [l[1] for l in list], 'k')
ax.plot([3, -2.805188, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.818426], 'r*', label='Минимумы')

plt.title('Функция Химмельблау')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title('Функция Химмельблау')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.contour(x, y, z)
ax.plot([l[0] for l in list], [l[1] for l in list], [himmelblau1(float(l[0]), float(l[1])) for l in list], 'k', label='Центры брусов')
ax.plot([3, -2.805188, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.818426], [himmelblau1(3, 2)], 'r*', label='Минимумы')
plt.title('Функция Химмельблау')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
