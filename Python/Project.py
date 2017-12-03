import hdbscan
# import sklearn
import numpy as np
from sklearn.preprocessing import scale
from sklearn import metrics # jaccard_similarity_score, silhouette_score

if __name__ == "__main__":
    # HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    #         gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
    #         metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
    clusterer = hdbscan.HDBSCAN()  #leaf_size=15)

    matrix = np.genfromtxt("../Matrix.csv", delimiter=',')
    tr = np.transpose(matrix)
    X_scaled = scale(tr, axis=1)
    y_true = list(range(4199)) * 15

    clusterer.fit(X_scaled)
    labels = clusterer.labels_
    print("Jaccard Index is ", metrics.jaccard_similarity_score(y_true, labels))
    print("Adjusted Rand Index is", metrics.adjusted_rand_score(y_true, labels))
    # print("Silhouette score is", metrics.silhouette_score(tr, labels))

    # db = sklearn.cluster.DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
    # clusterer.predict()
    # clusterer.fit_predict(X_scaled)

    print("Number of clusters is", clusterer.labels_.max())
    # print(dir(clusterer))
    g = clusterer.condensed_tree_.to_networkx()
    print(g)
    # print(clusterer.condensed_tree_.to_pandas())
    # clusterer.condensed_tree_.plot(select_clusters=True)
    clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
