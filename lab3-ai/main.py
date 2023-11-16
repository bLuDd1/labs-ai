import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

dataset, _ = make_blobs(n_samples=1000, centers=[[3, 2], [4, 3], [6, 8], [7, 5]], random_state=10)

plt.scatter(dataset[:, 0], dataset[:, 1])
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.title('Дані')
plt.show()

n_clusters = 4
centers, u, _, _, jm, _, _ = fuzz.cluster.cmeans(dataset.T, c=n_clusters, m=2, error=0.001, maxiter=100)
max_membership = np.argmax(u, axis=0)

for cluster in range(n_clusters):
    cluster_points = dataset[max_membership == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1])
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='black')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.title('Кластери та їх центри')
plt.show()

plt.plot(jm)
plt.title('Графік зміни значень цільової функції')
plt.xlabel('Номер ітерації')
plt.ylabel('Значення цільової функції')
plt.grid(True)
plt.show()

