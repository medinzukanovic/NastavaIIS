#OVO JE ONAJ STO ZABAGA, IZADJU ONI PROZORI I ZAPNE

import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.datasets import make_blobs
# ubacivanje n uzoraka, koliko cnetara, koliko osobina,...
x, y = make_blobs(n_samples=600, centers=6, random_state=100, cluster_std=0.9)
# poltanje skater dijagrama
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1], c='orange', alpha=0.6)
#plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.6)
plt.show()
#implementacija hijerarhijskog aglomerativnog klasteringa u scikit-learnu
from sklearn.cluster import AgglomerativeClustering
HAC = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')  
HAC.fit_predict(x)
#DENDOGRAM 
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
dn1 = shc.dendrogram(shc.linkage(x, method='ward'), ax=axes[0])
dn2 = shc.dendrogram(shc.linkage(x, method='ward'), ax=axes[1],color_threshold=25)
plt.show()