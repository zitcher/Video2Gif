from sklearn.cluster import KMeans
import numpy as np

def cluster(embeddings, k=12):
    kmeans = KMeans(n_clusters = k, random_state = 0)
    return kmeans.labels_

# Just do this recursively
def hierarchical_clustering(embeddings, d=4, k=12):
    if d == 1:
        return cluster(embeddings, k)
    else:
        labels = np.array(cluster(embeddings, k))
        final_labels = np.array([0]*len(labels))
        for i in range(k):
            embeds = embeddings[labels == i]
            labs = np.array(hierarchical_clustering(embeds, d-1, k))
            labs = labs + (i*k)
            final_labels[labels == i] = labs
    return final_labels



