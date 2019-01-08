from ClusterUtils import DBScan
from ClusterUtils import KMeans
from ClusterUtils import InternalValidator
from ClusterUtils import ExternalValidator
from ClusterUtils import KernelKM
from ClusterUtils import Spectral
from IPython import embed
import numpy as np

#1 Lloyd's kmeans
km1 = KMeans(init='random', n_clusters=5, csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/ZhihengWang_HW1_Skeleton/Datasets/three_globs.csv')
km1.fit_from_csv()
km1.show_plot()
km1.save_plot('KMrandom')
km1.save_csv()

km2 = KMeans(init='k-means++', n_clusters=5, csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/ZhihengWang_HW1_Skeleton/Datasets/three_globs.csv')
km2.fit_from_csv()
km2.show_plot()
km2.save_plot('KM++')
km2.save_csv()

km3 = KMeans(init='global', n_clusters=5, csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/ZhihengWang_HW1_Skeleton/Datasets/three_globs.csv')
km3.fit_from_csv()
km3.show_plot()
km3.save_plot('KMGlobal')
km3.save_csv()

#2 Hartigans
km4 = KMeans(algorithm = 'hartigans', init='k-means++', n_clusters=5,  csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/ZhihengWang_HW1_Skeleton/Datasets/three_globs.csv')
km4.fit_from_csv()
km4.show_plot()
km4.save_plot('hartigans_Kmean++')
km4.save_csv()

#3. internal 
km5 = KMeans(init='k-means++', csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/ZhihengWang_HW1_Skeleton/Datasets/image_segmentation.csv')
dfs = []
cs = []
for i in range(2, 10):
    km5.n_clusters = i # IMPORTANT -- Update the number of clusters to run.
    dfs.append(km5.fit_predict_from_csv())
    cs.append(i)
iv = InternalValidator(dfs, cluster_nums=cs)
iv.make_cvnn_table()
iv.show_cvnn_plot()
iv.save_cvnn_plot()

iv.make_silhouette_table()
iv.show_silhouette_plot()
iv.save_silhouette_plot()

iv.save_csv(cvnn=True, silhouette=True)


#4. external 
km6 = KMeans(n_clusters=2, csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/ZhihengWang_HW1_Skeleton/Datasets/image_segmentation.csv')
km6.fit_from_csv()
km6.show_plot()
data = km6.fit_predict_from_csv()
ev = ExternalValidator(data)
nmi = ev.normalized_mutual_info()
nri = ev.normalized_rand_index()
a = ev.accuracy()
print([nmi, nri, a])

#5 Ten times 
inter_si = []
exter_nmi = []
exter_nri = []
exter_a = []
km7 = KMeans(csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/ZhihengWang_HW1_Skeleton/Datasets/image_segmentation.csv', n_clusters=2)
for _ in range(10):
    data = km7.fit_predict_from_csv()
    ev = ExternalValidator(data)
    nmi = ev.normalized_mutual_info()
    nri = ev.normalized_rand_index()
    a = ev.accuracy()
    exter_a.append(a)
    exter_nmi.append(nmi)
    exter_nri.append(nri)
    print([nmi, nri, a])
print(np.mean(np.array(exter_a)))
print(np.mean(np.array(exter_nmi)))
print(np.mean(np.array(exter_nri)))
print(np.std(np.array(exter_a)))
print(np.std(np.array(exter_nmi)))
print(np.std(np.array(exter_nri)))
dfs = []
cs = []
for _ in range(10):
    km7.n_clusters = 5 # IMPORTANT -- Update the number of clusters to run.
    dfs.append(km7.fit_predict_from_csv())
    cs.append(5)

iv = InternalValidator(dfs, cluster_nums=cs)
iv.make_silhouette_table()
si = iv.silhouette_table


#DBSCAN 
db = DBScan(eps=0.2, min_points=10, csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/Datasets/anthill.csv')
db.fit_from_csv()
db.show_plot()
db.save_plot('DBScan plot')
db.save_csv()
data = db.fit_predict_from_csv()
ev = ExternalValidator(data)
nmi = ev.normalized_mutual_info()
nri = ev.normalized_rand_index()
a = ev.accuracy()
print([nmi, nri, a])




kernel = KernelKM(n_clusters=2, csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/Datasets/eye.csv')
kernel.fit_from_csv()
kernel.show_plot()
kernel.save_plot('kernel_plot')
kernel.save_csv()

spectral = Spectral(n_clusters=2,csv_path='/Users/wangzhiheng/Downloads/Homework_1-2/Datasets/eye.csv' )
spectral.fit_from_csv()
spectral.show_plot()
spectral.save_plot('kernel_plot')
spectral.save_csv()
