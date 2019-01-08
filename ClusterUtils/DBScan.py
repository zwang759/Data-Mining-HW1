#Zhiheng Wang
import pandas as pd
import numpy as np
import time
from ClusterUtils.SuperCluster import SuperCluster
from ClusterUtils.ClusterPlotter import _plot_generic_

def dbscan(X, eps=1, min_points=10, verbose=False):
    labels = [0]*len(X)
    C = 0
    for P in range(0, len(X)):
        if not (labels[P] == 0):
            continue
        NeighborPts = regionQuery(X, P, eps)
        if len(NeighborPts) < min_points:
            labels[P] = -1
        else:
           C += 1
           growCluster(X, labels, P, NeighborPts, C, eps, min_points)
    return labels
def growCluster(X, labels, P, NeighborPts, C, eps, min_points):
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
            labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = regionQuery(X, Pn, eps)
            if len(PnNeighborPts) >= min_points:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1
def regionQuery(X, P, eps):
    neighbors = []
    for Pn in range(0, len(X)):
        if np.linalg.norm(X[P] - X[Pn]) < eps:
           neighbors.append(Pn)
    return neighbors

# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class DBScan(SuperCluster):
    """
    Perform DBSCAN clustering from vector array.
    Read more in the :ref:`User Guide <dbscan>`.
    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
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

    def __init__(self, eps=1, min_points=10, csv_path=None, keep_dataframe=True,
                                                    keep_X=True,verbose=False):
        self.eps = eps
        self.min_points = min_points
        self.verbose = verbose
        self.csv_path = csv_path
        self.keep_dataframe = keep_dataframe
        self.keep_X = keep_X

    # X is an array of shape (n_samples, n_features)
    def fit(self, X):
        if self.keep_X:
            self.X = X
        start_time = time.time()
        self.labels = dbscan(X, eps=self.eps, min_points = self.min_points,verbose = self.verbose)
        print("DBSCAN finished in  %s seconds" % (time.time() - start_time))
        return self

    def show_plot(self):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels)
        else:
            print('No data to plot.')

    def save_plot(self, name):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF, save=True, n=name)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels, save=True, n=name)
        else:
            print('No data to plot.')
