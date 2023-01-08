import csv
from matplotlib import pyplot as plt
from scipy.optimize import linprog
import numpy as np

data = []
EPS = 0
eps = 1e-4 * 1.2

with open("data/Ch1.txt") as file:
    a, b = file.readline().split(" ")
    w = [float(line) for line in file]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()


max_w = max(w)

y = []
x = []
with open("data/ReverseChanel1_800nm_0_03.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=";")
    i = 0
    for line_count, row in enumerate(csv_reader):
        if line_count != 0:
            data.append([float(row[0]) - eps * max_w, float(row[0]) + eps * max_w])
            ax1.vlines(i, data[-1][0], data[-1][1], "b")
            ax2.vlines(i, float(row[0]) - eps * w[i], float(row[0]) + eps * w[i], "b")
            ax3.vlines(i, data[-1][0], data[-1][1], "b")
            y.append(float(row[0]))
            x.append(i)
            i += 1

v = []
for i in range(len(x)):
    for j in range(i + 1, len(x)):
        px1 = [x[i], x[j]]
        py1 = [data[i][0], data[j][0]]

        px2 = [x[i], x[j]]
        py2 = [data[i][0], data[j][1]]

        px3 = [x[i], x[j]]
        py3 = [data[i][1], data[j][0]]

        px4 = [x[i], x[j]]
        py4 = [data[i][1], data[j][1]]

        c1 = np.polyfit(px1, py1, 1)
        c2 = np.polyfit(px2, py2, 1)
        c3 = np.polyfit(px3, py3, 1)
        c4 = np.polyfit(px4, py4, 1)

        isIn1 = True
        isIn2 = True
        isIn3 = True
        isIn4 = True
        for k in x:
            t = c1[1] + c1[0] * k
            if isIn1 and (t < data[k][0] - EPS or t > data[k][1] + EPS):
                isIn1 = False

            t = c2[1] + c2[0] * k
            if isIn2 and (t < data[k][0] - EPS or t > data[k][1] + EPS):
                isIn2 = False

            t = c3[1] + c3[0] * k
            if isIn3 and (t < data[k][0] - EPS or t > data[k][1] + EPS):
                isIn3 = False

            t = c4[1] + c4[0] * k
            if isIn4 and (t < data[k][0] - EPS or t > data[k][1] + EPS):
                isIn4 = False

        if isIn1:
            v.append(c1)

        if isIn2:
            v.append(c2)

        if isIn3:
            v.append(c3)

        #if isIn4:
        #    v.append(c4)

print(v)

obj = [1 for _ in range(len(x) + 2)]
obj[0] = 0
obj[1] = 0
bnd = [(0, float("inf")) for _ in range(len(x) + 2)]
bnd[0] = (float("-inf"), float("inf"))
bnd[1] = (float("-inf"), float("inf"))
lhs_ineq = []
rhs_ineq = []

for k in x:
    coefs = [0 for _ in range(len(x) + 2)]
    coefs[0] = 1
    coefs[1] = k
    coefs[k + 2] = -1
    lhs_ineq.append(coefs)
    rhs_ineq.append(y[k])
    coefs[0] = -1
    coefs[1] = -k
    coefs[k + 2] = -1
    lhs_ineq.append(coefs)
    rhs_ineq.append(-y[k])

opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd, method="highs")
solve = opt.x[:2]
print(opt.x[:2])

for tmp in v:
    polynomial = np.poly1d(tmp)
    x_axis = np.linspace(0, 200, 10)
    y_axis = polynomial(x_axis)
    ax1.plot(x_axis, y_axis)


ks = [tmp[0] for tmp in v]
bs = [tmp[1] for tmp in v]
k_c = np.mean(ks)
b_c = np.mean(bs)
print(k_c, b_c)
polynomial = np.poly1d(np.array([k_c, b_c]))
x_axis = np.linspace(0, 200, 10)
y_axis = polynomial(x_axis)
# ax3.plot(x_axis, y_axis)

max = []
min = []
vals = []
for tmp in v:
    polynomial = np.poly1d(tmp)
    vals.append(polynomial(101.5))
max.append(np.max(vals))
min.append(np.min(vals))

vals = []
for tmp in v:
    polynomial = np.poly1d(tmp)
    vals.append(polynomial(-10))
max.append(np.max(vals))
min.append(np.min(vals))

vals = []
for tmp in v:
    polynomial = np.poly1d(tmp)
    vals.append(polynomial(1000))
max.append(np.max(vals))
min.append(np.min(vals))

plt.figure()
for tmp in v:
    polynomial = np.poly1d(tmp)
    x_axis = np.linspace(95, 110, 5)
    y_axis = polynomial(x_axis)
    plt.plot(x_axis, y_axis)

plt.vlines(101.5, min[0], max[0])
plt.plot(101.5, (min[0] + max[0]) / 2, marker="o", markersize=5, markerfacecolor="red")

plt.figure()
for tmp in v:
    polynomial = np.poly1d(tmp)
    x_axis = np.linspace(-20, 10, 5)
    y_axis = polynomial(x_axis)
    plt.plot(x_axis, y_axis)

plt.vlines(-10, min[1], max[1])
plt.plot(-10, (min[1] + max[1]) / 2, marker="o", markersize=5, markerfacecolor="red")

plt.figure()
for tmp in v:
    polynomial = np.poly1d(tmp)
    x_axis = np.linspace(990, 1010, 5)
    y_axis = polynomial(x_axis)
    plt.plot(x_axis, y_axis)

plt.vlines(1000, min[2], max[2])
plt.plot(1000, (min[2] + max[2]) / 2, marker="o", markersize=5, markerfacecolor="red")

print(max, min)
for m, l in zip(max, min):
    print("[{}, {}]".format(l, m))
    print((m + l) / 2)
    print((m - l) / 2)

plt.figure()

x, y = [], []
for tmp in v:
    x.append(tmp[1])
    y.append(tmp[0])
    plt.plot(tmp[1], tmp[0], marker="o", markersize=5, markerfacecolor="red")
x, y = np.array(x), np.array(y)
order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
plt.fill(x[order], y[order], "g", alpha=0.5)

plt.show()
