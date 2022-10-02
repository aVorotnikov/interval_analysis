import intvalpy as ip
import numpy as np
import matplotlib.pyplot as plt


def det2(A):
    return A[0][0] * A[1][1] - A[1][0] * A[0][1]


def get_det(A, delta):
    rad = np.array([[delta, 0], [delta, 0]])
    A_int = ip.Interval(A, rad, midRadQ=True)
    return det2(A_int)


def bauman(vertices):
    for i in range(len(vertices)):
        for j in range(i + 1):
            if det2(vertices[i]) * det2(vertices[j]) <= 0:
                return False
    return True


A = np.array([[1.05, 1.0], [0.95, 1.0]])
deltas = np.linspace(0, 1, 101)
dets_a = []
dets_b = []
for delta in deltas:
    det = get_det(A, delta)
    dets_a.append(det.a)
    dets_b.append(det.b)

fig, ax = plt.subplots()
ax.plot(deltas, dets_a, 'g-', label='Нижняя граница интервала детерминанта')
ax.plot(deltas, dets_b, 'r-', label='Верхняя граница интервала детерминанта')
ax.plot([0.05], [0], 'b*', label='0.05')
plt.xlabel('delta')
plt.legend()
plt.grid()
plt.show()

for delta in deltas:
    A1 = A.copy()
    A1[0][1] -= delta
    A1[1][1] -= delta
    A2 = A.copy()
    A2[0][1] -= delta
    A2[1][1] += delta
    A3 = A.copy()
    A3[0][1] += delta
    A3[1][1] -= delta
    A4 = A.copy()
    A4[0][1] += delta
    A4[1][1] += delta
    verticesA = [A1, A2, A3, A4]
    print("{:2.4f}\t{}".format(delta, "неособенная" if bauman(verticesA) else "особенная"))
