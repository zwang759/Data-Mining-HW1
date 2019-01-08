#Zhiheng Wang
import pandas as pd
import numpy as np
import random
import time
from ClusterUtils.SuperCluster import SuperCluster
from ClusterUtils.ClusterPlotter import _plot_kmeans_
from ClusterUtils.InternalValidator import get_sihouette
from IPython import embed

def centroid_init(init, n_clusters, X):
    m,n = X.shape
    if init == 'random':
        c_index = random.sample(range(m), n_clusters)
        centroids = X[c_index]
        pass
    elif init == 'k-means++':
        c_index = []
        first = random.randint(0,m-1)
        c_index.append(first)
        for k in range(1,n_clusters):
            cents = X[c_index]
            dis_matrix = np.zeros((m,1))
            for j in range(m):
                dis = ((cents - X[j])**2).sum(axis=1)
                d = np.min(dis)
                d_squre = d**2
                dis_matrix[j] = d_squre
            total = dis_matrix.sum()
            p_matrix = dis_matrix/total
            p_sum = np.zeros((m,1))
            p_sum[0] = p_matrix[0]
            rand_num = random.random()
            if rand_num<= p_sum[0]:
                c_index.append(0)
            else:
                for j in range(1,m):
                    p_sum[j] = p_matrix[j] + p_sum[j-1]
                    if p_sum[j]>=rand_num and rand_num >= p_sum[j-1]:
                        c_index.append(j)
                        break
        centroids = X[c_index]

    elif init == 'global':
        c_index = []
        for k in range(n_clusters):
            d = np.zeros((m,1))
            for j in range(m):
                temp_c = c_index[:]
                temp_c.append(j)
                cents = X[temp_c][:]
                sum_d = 0
                for q in range(m):
                    sum_d += np.min(((cents-X[q])**2).sum(axis=1))
                d[j] = sum_d
            idx = np.argmin(d)
            c_index.append(idx)
        centroids = X[c_index]
    else:
        c_index = random.sample(range(m), n_clusters)
        centroids = X[c_index]
    return centroids

def getLabels(X,centroids,n_clusters, algorithm, max_iter, verbose):
    m,n = X.shape
    labels = np.zeros((m,1))
    if algorithm == 'lloyds':
        for _ in range(max_iter):
            for i in range(m):
                d = ((centroids - X[i]) * (centroids - X[i])).sum(axis=1)
                labels[i] = d.argmin()
            for k in range(n_clusters):
                r = X[np.where(labels==k)[0]][:]
                if not r.any():
                    centroids[k] = np.zeros((1,n))
                else:
                    new_mean = r.mean(axis=0)
                    centroids[k] = new_mean[:]
    if algorithm == 'hartigans':
        labels = np.array([random.randrange(0,n_clusters) for _ in range(m)]).reshape((m,1))
        for iter in range(max_iter):
            for i in range(m):
                idx = labels[i]
                d_matrix = np.zeros((n_clusters,1))
                for k in range(n_clusters):
                    r = X[np.where(labels==k)[0]]
                    d = ((r-centroids[k])**2).sum()
                    d_matrix[k] = d
                delta_matrix = np.zeros((n_clusters,1))
                change_idx = (X[i]-centroids[idx])**2
                for k in range(n_clusters):
                    delta_matrix[k] = 0 - change_idx.sum() + ((centroids[k]-X[i])**2).sum()
                new_cen_idx = np.argmin(delta_matrix)
                if new_cen_idx != idx:
                    temp =  X[np.where(labels==new_cen_idx)[0]]
                    num = len(temp)
                    old_mean = ((temp.sum(axis=0)) -X[i]) / (num-1)
                    centroids[idx] = old_mean[:]
                    temp2 = X[np.where(labels==new_cen_idx)[0]][:]
                    new_mean = ((temp2.sum(axis=0)) + X[i])/(len(temp2)+1)
                    centroids[new_cen_idx] = new_mean[:]
                    labels[i] = new_cen_idx

    return labels, centroids

def k_means(X, n_clusters=3, init='random', algorithm='lloyds', n_init=1, max_iter=300, verbose=False):

    # Implement.

    # Input: np.darray of samples

    # Return the following:
    #
    # 1. labels: An array or list-type object corresponding to the predicted
    #  cluster numbers,e.g., [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # 2. centroids: An array or list-type object corresponding to the vectors
    # of the centroids, e.g., [[0.5, 0.5], [-1, -1], [3, 3]]
    # 3. inertia: A number corresponding to some measure of fitness,
    # generally the best of the results from executing the algorithm n_init times.
    # You will want to return the 'best' labels and centroids by this measure.
    if n_init==1:
        centroids = centroid_init(init, n_clusters, X)
        labels, centroids = getLabels(X,centroids,n_clusters,algorithm, max_iter, verbose)
        return labels, centroids, None
    
    max_sihouettes = -10
    centroids_return = []
    labels_return = []
    for i in range(n_init):
        centroids = centroid_init(init, n_clusters, X)
        labels, centroids = getLabels(X,centroids,n_clusters,algorithm, max_iter, verbose)
        if n_init==1:
            return labels, centroids, None
        m,n = X.shape
        matrix = np.zeros((m,m))
        for i in range(m):
            matrix[i] = ((X - X[i]) * (X-X[i])).sum(axis=1)
        dis_matrix = np.sqrt(matrix)
        clusters = {}
        for i in range(m):
            l = labels[i][0]
            if l not in clusters:
                clusters[l] = []
            clusters[l].append(i)
        sihouette = get_sihouette(dis_matrix, clusters, m, labels,n_clusters)
        if sihouette > max_sihouettes:
            max_sihouettes = sihouette
            labels_return = labels[:]
            centroids_return = centroids[:]
    return labels_return, centroids_return, max_sihouettes 

    # Implement.

    # Input: np.darray of samples

    # Return the following:
    #
    # 1. labels: An array or list-type object corresponding to the predicted
    #  cluster numbers,e.g., [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # 2. centroids: An array or list-type object corresponding to the vectors
    # of the centroids, e.g., [[0.5, 0.5], [-1, -1], [3, 3]]
    # 3. inertia: A number corresponding to some measure of fitness,
    # generally the best of the results from executing the algorithm n_init times.
    # You will want to return the 'best' labels and centroids by this measure.



# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class KMeans(SuperCluster):
    """
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init : {'random', 'k-means++', 'global', or 'hartigans'}
        Method for initialization, defaults to 'random'.
    n_init : int, default: 1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    csv_path : str, default: None
        Path to file for dataset csv
    keep_dataframe : bool, default: True
        Hold on the results pandas DataFrame generated after each run.
        Also determines whether to use pandas DataFrame as primary internal data state
    keep_X : bool, default: True
        Hold on the results generated after each run in a more generic array-type format
        Use these values if keep_dataframe is False
    verbose: bool, default: False
        Optional log level
    """

    def __init__(self, n_clusters=3, init='random', n_init=1, algorithm='lloyds', max_iter=300,
                 csv_path=None, keep_dataframe=True, keep_X=True, verbose=False):
        self.n_clusters=n_clusters
        self.init = init
        self.algorithm = algorithm # IMPORTANT -- attach new 'algorithm' parameter to self
        self.n_init = n_init
        self.max_iter = max_iter
        self.csv_path = csv_path
        self.keep_dataframe = keep_dataframe
        self.keep_X = keep_X
        self.verbose = verbose

    # X is an array of shape (n_samples, n_features)
    def fit(self, X):
        if self.keep_X:
            self.X = X
        start_time = time.time()
        self.labels, self.centroids, self.inertia = \
            k_means(X, n_clusters=self.n_clusters, init=self.init, algorithm=self.algorithm,
                    n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose)
        print(self.init + " k-means finished in  %s seconds" % (time.time() - start_time))
        return self

    def show_plot(self):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_kmeans_(df=self.DF)
        elif self.keep_X:
            _plot_kmeans_(X=self.X, labels=self.labels, centroids=self.centroids)
        else:
            print('No data to plot.')

    def save_plot(self, name = 'kmeans_plot'):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_kmeans_(df=self.DF, save=True, n=name)
        elif self.keep_X:
            _plot_kmeans_(X=self.X, labels=self.labels,
                            centroids=self.centroids, save=True, n=name)
        else:
            print('No data to plot.')
