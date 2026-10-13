import numpy as np
import torch
from sklearn.cluster import KMeans


def cluster(values, count, seed, n_init):
    array = values.detach().float().cpu().numpy()
    if count > len(array):
        raise ValueError("More clusters requested than available training samples")
    estimator = KMeans(n_clusters=count, random_state=seed, n_init=n_init, algorithm="lloyd")
    labels = estimator.fit_predict(array)
    if len(np.unique(labels)) != count:
        raise ValueError("Too few distinct embeddings to construct the requested nonempty clusters")
    return labels, estimator.cluster_centers_


def medoid_indices(values, count, seed, n_init):
    labels, centers = cluster(values, count, seed, n_init)
    array = values.detach().float().cpu().numpy()
    result = []
    for index in range(count):
        members = np.flatnonzero(labels == index)
        distances = np.square(array[members] - centers[index]).sum(axis=1)
        result.append(int(members[np.argsort(distances, kind="stable")[0]]))
    return torch.tensor(result, dtype=torch.long)


def global_representatives(values, count, representative_count, seed, n_init):
    labels, centers = cluster(values, count, seed, n_init)
    array = values.detach().float().cpu().numpy()
    selected = torch.full((count, representative_count), -1, dtype=torch.long)
    for index in range(count):
        members = np.flatnonzero(labels == index)
        distances = np.square(array[members] - centers[index]).sum(axis=1)
        ranked = members[np.argsort(distances, kind="stable")][:representative_count]
        selected[index, :len(ranked)] = torch.from_numpy(ranked.astype(np.int64))
    return torch.from_numpy(centers.astype(np.float32)), selected
