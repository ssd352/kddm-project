import numpy as np
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.cluster import DBSCAN

if __name__ == "__main__":

    NUM_CLASSES = 4199

    matrix = np.genfromtxt("../Matrix.csv", delimiter=',')
    tr = np.transpose(matrix)
    X_scaled = scale(tr, axis=1)
    y_true = np.asarray(list(range(NUM_CLASSES)) * 15)

    SAMPLE = NUM_CLASSES
    picking_ind = [class_num + i * NUM_CLASSES for class_num in range(SAMPLE) for i in range(15)]
    X_sampled = X_scaled[picking_ind, :]
    y_sampled = y_true[picking_ind]

    print('DBScan-------------------------------->')
    print('eps,minPts,Number of Clusters,Jaccard Index,Adjusted Rand Index')
    for min_samples in range(2, 15 + 1):
        for eps in [0.01, 0.05, 0.1, 0.125, 0.25, 0.5, 0.75, 1]:
            # print('epsilon is ', eps, 'min points is', min_samples)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_sampled)
            # np.save("DBScan.npy", db.labels_)
            print("{},{},{},{},{}", eps, min_samples, db.labels_.max() + 1,
                  metrics.jaccard_similarity_score(y_sampled, db.labels_),
                  metrics.adjusted_rand_score(y_sampled, db.labels_))
            # print("Number of clusters", db.labels_.max() + 1)
            # print("Jaccard Index for DBScan is ", metrics.jaccard_similarity_score(y_sampled, db.labels_))
            # print("Rand Index for DBScan is", metrics.adjusted_rand_score(y_sampled, db.labels_))
            # print("Silhouette score is", metrics.silhouette_score(X_sampled, db.labels_))
