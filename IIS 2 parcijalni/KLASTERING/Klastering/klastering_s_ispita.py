import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
#ubacivanje podataka
data = pd.read_csv(r'C:\Users\HP\Documents\Py Programs\podaci\Clustering.csv')
print(data)
x = np.array(data.iloc[:,0].values).reshape(-1,1)
y = np.array(data.iloc[:,1].values).reshape(-1,1)
#####################
##############plot the dataset
plt.scatter(x,y)
plt.xlabel('X osa')
plt.ylabel('Y osa')
plt.title('Dataset')
plt.show()
from sklearn.cluster import KMeans
SSE = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data)
    SSE.append(kmeans.inertia_)  
print(SSE)
# Plot the elbow
# Metoda "lakta" - bukvalno prevedemo
# Crtanje grafika koji govori koliko ćemo imati klastera(gleda se prelomna tačka)
# U našem slučaju je to 15 
plt.plot(range(1, 20), SSE, color = 'red', marker='o', markersize=10)
plt.xlabel('K')
plt.ylabel('SSE=f(K)')
plt.title('Elbow Method showing optimal K')
plt.show()
#Implementacija K-Means u scikit-learn
kmeans = KMeans(n_clusters=3, max_iter=300, random_state=100)
kmeans.fit(data)
centroids=kmeans.cluster_centers_
print('centrids')
print(centroids)
# crtanje skater dijagrama sa naznačenim centrima klastera 
plt.figure(figsize=(10, 10))
plt.xlabel('X osa')
plt.ylabel('Y osa')
plt.title('Klastering')
plt.scatter(x, y, c=y,alpha=0.6)
plt.scatter(centroids[:,0], centroids[:,1], s=200, c='red')
plt.show()
p2=[[-5,-5], [0,0],[5,5]]
pred = kmeans.predict(p2)
print(pred)