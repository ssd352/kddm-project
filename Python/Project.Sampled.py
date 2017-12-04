import hdbscan
# import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn import metrics  # jaccard_similarity_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN

if __name__ == "__main__":

    # HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    #         gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
    #         metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
        # leaf_size=15)
    NUM_CLASSES = 4199

    matrix = np.genfromtxt("../Matrix.csv", delimiter=',')
    tr = np.transpose(matrix)
    X_scaled = scale(tr, axis=1)
    y_true = np.asarray(list(range(NUM_CLASSES)) * 15)

    SAMPLE = NUM_CLASSES
    picking_ind = [class_num + i * NUM_CLASSES for class_num in range(SAMPLE) for i in range(15)]
    X_sampled = X_scaled[picking_ind, :]
    y_sampled = y_true[picking_ind]

    print("Silhouette score is", metrics.silhouette_score(X_sampled, y_sampled))

    kmeans = KMeans(n_clusters=SAMPLE).fit(X_sampled)
    print('KMeans-------------------------------->')
    print("K-means Labels:", kmeans.labels_)
    np.save("Kmeans.npy", kmeans.labels_)
    print("Jaccard Index for K-means is ", metrics.jaccard_similarity_score(y_sampled, kmeans.labels_))
    print("Rand Index for K-means is", metrics.adjusted_rand_score(y_sampled, kmeans.labels_))
    print("Silhouette score is", metrics.silhouette_score(X_sampled, kmeans.labels_))
    # print(kmeans.labels_)

    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X_sampled)
    # labels = clusterer.labels_
    print('HDBScan-------------------------------->')
    print("Number of clusters is", clusterer.labels_.max() + 1)
    # print("HDBScan Labels:", clusterer.labels_)
    np.save("HDBScan.npy", clusterer.labels_)
    print("Jaccard Index for HDBScan is ", metrics.jaccard_similarity_score(y_sampled, clusterer.labels_))
    print("Rand Index for HDBScan is", metrics.adjusted_rand_score(y_sampled, clusterer.labels_))
    print("Silhouette score is", metrics.silhouette_score(X_sampled, clusterer.labels_))
    # clusterer.condensed_tree_.plot(select_clusters=True)
    # plt.show()
    # print("Silhouette score is", metrics.silhouette_score(X_sampled, labels))

    print('DBScan-------------------------------->')
    db = DBSCAN(eps=0.1, min_samples=5).fit(X_sampled)
    # print("DBScan Labels:", db.labels_)
    np.save("DBScan.npy", db.labels_)
    print("Number of clusters", db.labels_.max() + 1)
    print("Jaccard Index for DBScan is ", metrics.jaccard_similarity_score(y_sampled, db.labels_))
    print("Rand Index for DBScan is", metrics.adjusted_rand_score(y_sampled, db.labels_))
    print("Silhouette score is", metrics.silhouette_score(X_sampled, db.labels_))
    # clusterer.condensed_tree_.plot(select_clusters=True)
    # clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
