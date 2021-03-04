#CIFRE,HEATMAP; CONFUSIONMATRIX; ACCURACY SCORE
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.cluster import KMeans
#PODACI
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
#FITOVANJE
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)
#CENTRI KLASTERA
#Visualizing cluster centers
fig, ax = plt.subplots(2, 5, figsize=(4, 2))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
#matching each learned cluster label with the true labels found in them
from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
#Accuracy check
from sklearn.metrics import accuracy_score
print(accuracy_score(digits.target, labels))
#Visualizing accuracy - confussion matrix
plt.figure(figsize=(6, 6))
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
