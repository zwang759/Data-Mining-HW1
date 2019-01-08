#Zhiheng Wang
import pandas as pd
import time
from ClusterUtils.ClusterPlotter import _plot_cvnn_
from ClusterUtils.ClusterPlotter import _plot_silhouette_
import numpy as np
import collections as cl
from IPython import embed

def build_dismatrix(dataset,cluster_n):
    dataset = dataset.values
    data = dataset[:-1*cluster_n,:-1]
    label = dataset[:-1*cluster_n,-1]
    n = len(data)
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i] = ((data - data[i]) * (data-data[i])).sum(axis=1)
    result_matrix = np.sqrt(matrix)
    clusters = {}
    for i in range(n):
        l = label[i]
        if l not in clusters:
            clusters[l] = []
        clusters[l].append(i)
    return result_matrix, clusters, n, label

def findSep(k, dis_matrix, points):
    n, n = dis_matrix.shape
    ni = len(points)
    weights = []
    for i in points:
        tuple_dis = [(ith,dis_matrix[i][ith]) for ith in range(n)]
        tuple_dis = sorted(tuple_dis, key=lambda x:x[1])
        q = 0
        for p in tuple_dis[1:k+1]:
            if p[0] not in points:
                q += 1
        weight = float(q)/float(k)
        weights.append(weight)
    avg_weight = np.array(weights).mean()
    return avg_weight

def findCom(points, dis_matrix):
    ni = len(points)
    points_incluster = dis_matrix[points][:,points]
    dis_sum = points_incluster.sum()/2
    com = 2*dis_sum/(ni*(ni-1)) if ni > 1 else 0
    return com

def get_sihouette(dis_matrix, clusters, n, label, cluster_num):
    s_matrix = np.zeros((n,1))
    for i in range(n):
        l = label[i]
        a = 0
        b = []
        for k in clusters:
            points = clusters[k]
            r = dis_matrix[i][points]
            size = len(r)
            if k==l:
                a = 0 if size== 1 else r.sum()/(size-1)
            else:
                b.append(r.sum()/size)
        b = min(b) if b else 0
        s = (b-a)/max(a,b)
        s_matrix[i] = s
    s_total = 0
    for k in clusters:
        points = clusters[k]
        s = s_matrix[points]
        s_mean_cluster = np.mean(s)
        s_total += s_mean_cluster
    si = s_total/cluster_num
    return si

def tabulate_silhouette(datasets, cluster_nums):

    # Implement.

    # Inputs:
    # datasets: Your provided list of clustering results.
    # cluster_nums: A list of integers corresponding to the number of clusters
    # in each of the datasets, e.g.,
    # datasets = [np.darray, np.darray, np.darray]
    # cluster_nums = [2, 3, 4]

    # Return a pandas DataFrame corresponding to the results.
    silhouette = []
    for idx in range(len(datasets)):
        dis_matrix, clusters, n, label = build_dismatrix(datasets[idx], cluster_nums[idx])
        si = get_sihouette(dis_matrix, clusters, n, label, cluster_nums[idx])
        silhouette.append(si)
    dfdata = {'CLUSTERS':cluster_nums,'SILHOUETTE_IDX':silhouette}
    df = pd.DataFrame(dfdata)
    return df

def tabulate_cvnn(datasets, cluster_nums, k_vals):

    # Implement.

    # Inputs:
    # datasets: Your provided list of clustering results.
    # cluster_nums: A list of integers corresponding to the number of clusters
    # in each of the datasets, e.g.,
    # datasets = [np.darray, np.darray, np.darray]
    # cluster_nums = [2, 3, 4]

    # Return a pandas DataFrame corresponding to the results.
    ks = len(k_vals)
    k_out = []
    cluster_out = []
    cvnn_out = []

    for k in k_vals:
        separations = []
        compacts = []
        for idx in range(len(cluster_nums)):  ##for each dataset of nc clustering number,
            dis_matrix, clusters, n, _ = build_dismatrix(datasets[idx], cluster_nums[idx])
            nc = cluster_nums[idx]
            sep = []
            coms = []

            for i in clusters:
                points = clusters[i]
                avg_weight = findSep(k, dis_matrix, points)
                sep.append(avg_weight)
                com = findCom(points, dis_matrix)
                coms.append(com)
            separation = max(sep)
            separations.append(separation)

            compact = sum(coms)
            compacts.append(compact)
        sep_max = max(separations)
        sep_norm = (np.array(separations)) / sep_max

        com_max = max(compacts)
        com_norm = (np.array(compacts)) / com_max

        cvnn = sep_norm + com_norm

        k_out += [k] * len(cluster_nums)
        cvnn_out += list(cvnn)
        cluster_out += cluster_nums

    df = {'CLUSTERS':cluster_out,'K':k_out,'CVNN':cvnn_out}
    data_matrix = pd.DataFrame(df)

    return data_matrix


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class InternalValidator:
    """
    Parameters
    ----------
    datasets : list or array-type object, mandatory
        A list of datasets. The final column should cointain predicted labeled results
        (By default, the datasets generated are pandas DataFrames, and the final
        column is named 'CLUSTER')
    cluster_nums : list or array-type object, mandatory
        A list of integers corresponding to the number of clusters used (or found).
        Should be the same length as datasets.
    k_vals: list or array-type object, optional
        A list of integers corresponding to the desired values of k for CVNN
        """

    def __init__(self, datasets, cluster_nums, k_vals=[1, 5, 10, 20]):
        if 'CENTROID' in datasets[0].index:
            self.datasets = list(map(lambda df : df.drop('CENTROID', axis=0), datasets))
        else:
            self.datasets = datasets
        self.cluster_nums = cluster_nums
        self.k_vals = k_vals

    def make_cvnn_table(self):
        start_time = time.time()
        self.cvnn_table = tabulate_cvnn(self.datasets, self.cluster_nums, self.k_vals)
        print("CVNN finished in  %s seconds" % (time.time() - start_time))

    def show_cvnn_plot(self):
        _plot_cvnn_(self.cvnn_table)

    def save_cvnn_plot(self, name='cvnn_plot'):
        _plot_cvnn_(self.cvnn_table, save=True, n=name)

    def make_silhouette_table(self):
        start_time = time.time()
        self.silhouette_table = tabulate_silhouette(self.datasets, self.cluster_nums)
        print("Silhouette Index finished in  %s seconds" % (time.time() - start_time))

    def show_silhouette_plot(self):
        _plot_silhouette_(self.silhouette_table)

    def save_silhouette_plot(self, name='silhouette_plot'):
        _plot_silhouette_(self.silhouette_table, save=True, n=name)

    def save_csv(self, cvnn=False, silhouette=False, name='internal_validator'):
        if cvnn is False and silhouette is False:
            print('Please pass either cvnn=True or silhouette=True or both')
        if cvnn is not False:
            filename = name + '_cvnn_' + (str(round(time.time()))) + '.csv'
            self.cvnn_table.to_csv(filename)
            print('Dataset saved as: ' + filename)
        if silhouette is not False:
            filename = name + '_silhouette_' + (str(round(time.time()))) + '.csv'
            self.silhouette_table.to_csv(filename)
            print('Dataset saved as: ' + filename)
        if cvnn is False and silhouette is False:
            print('No data to save.')
