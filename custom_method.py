import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.animation import FuncAnimation

data = load_iris()
X = data.data[:, :2]

k = 5
np.random.seed(42)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]
frames = []


def compute_distance(points, centers) -> list:
    return np.linalg.norm(points[:, np.newaxis] - centers, axis=2)


def compute_closest_centroids(distances) -> list:
    labels = []
    distances = distances.tolist()
    for vect in distances:
        labels.append(vect.index(min(vect)))
    return labels


def update_centroids(X, labels, k):
    new_centroids = []

    for i in range(k):
        cluster_points = [X[j] for j in range(len(labels)) if labels[j] == i]
        if cluster_points:
            cluster_mean = [
                sum(coord) / len(coord) for coord in zip(*cluster_points)
            ]
        else:
            cluster_mean = [0] * len(X[0])
        new_centroids.append(cluster_mean)
        

    return new_centroids

centroids = np.array(centroids)
steps = []
for step in range(10):
    distances = compute_distance(X, centroids)
    labels = compute_closest_centroids(distances)
    steps.append((np.array(centroids), np.array(labels)))

    new_centroids = update_centroids(X, labels, k)
    if np.allclose(centroids, new_centroids):
        break
    centroids = np.array(new_centroids)

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    centroids, labels = steps[frame]
    for i in range(k):
        cluster_points = np.array([X[j] for j in range(len(labels)) if labels[j] == i])
        ax.scatter(cluster_points[:, 0],
                   cluster_points[:, 1],
                   label=f'Cluster {i+1}')
        ax.scatter(centroids[i, 0],
                   centroids[i, 1],
                   s=200,
                   c='black',
                   marker='X')
    ax.set_title(f'Шаг {frame + 1}')
    ax.legend()

animation = FuncAnimation(fig, update, frames=len(steps), repeat=False)
animation.save('kmeans_steps.gif', writer='pillow')
plt.show()