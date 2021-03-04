#AGLOMERATIVNI KLASTERING
#IMPORTOVANJE BIBLIOTEKA
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
#IMPORTOVANJE PODATAKA
from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=600, centers=6, random_state=100, cluster_std=0.9)
#PRIKAZ PODATAKA
#plt.scatter(x[:, 0], x[:, 1], c='grey', alpha=0.6) #JEDNOBOJNI
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.6) #VIŠEBOJNI
#FITTOVANJE PODATAKA
HAC = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')  
predikcija=HAC.fit_predict(x)
#print(predikcija)
#DENDOGRAM
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
dn1 = shc.dendrogram(shc.linkage(x, method='ward'), ax=axes[0])
dn2 = shc.dendrogram(shc.linkage(x, method='ward'), ax=axes[1],color_threshold=25)
plt.show()