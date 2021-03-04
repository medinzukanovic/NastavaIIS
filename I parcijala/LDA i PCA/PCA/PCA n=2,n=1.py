import numpy as np
import matplotlib.pyplot as plt
######################                      KREIRANJE/IMPORTOVANJE PODATAKA
rng = np.random.RandomState(1)
#X = np.dot(rng.rand(5, 5), rng.rand(5, 200)).T
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
##################                       PLOTOVANJE ORGINALNIH PODATAKA
#plotovanje orginalnih podataka
fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(151)
ax1.scatter(X[:, 0], X[:, 1])
ax1.legend(['X'],loc='best')
ax1.set_title('Orginalni podaci')
print('------------------------')
print('Shape orginalnog niza:')
print(np.shape(X))
print('------------------------')
########################                 PCA
#PCA bez redukovanja dimenzija
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
#######################         PLOTOVANJE TRANSFORMISANIH I ORGINALNIH NA JEDNOM DIJAGRAMU
#plotovanje orginalnih
#originalni podaci
ax2 = fig.add_subplot(152)
ax2.scatter(X[:, 0], X[:, 1], color='black', s=100, alpha=1)

#transformirani podaci - PCA 
ax2.scatter(X_pca[:, 0], X_pca[:, 1], color='red',marker='+', s=100, alpha=0.5)
ax2.legend(['X','XPCA'],loc='best')
ax2.set_title('Org. + PCA')
#############################              PLOTOVANJE ORGINALNI, PCA, i INVERZNIH PODATAKA
#Plotovanje inverznih i orginalnih
#originalni podaci
ax3=fig.add_subplot(153)
ax3.scatter(X[:, 0], X[:, 1], color='black', s=100, alpha=1)

#transformirani podaci - podaci u novom koordinatnom sistemu (koordinatni sistem PCA komponenata)
X_pca = pca.transform(X)
ax3.scatter(X_pca[:, 0], X_pca[:, 1], color='red',marker='v', s=100, alpha=0.5)

#povratak na originalne podatke, ali na osnovu transformiranih podataka
X_new = pca.inverse_transform(X_pca)
ax3.scatter(X_new[:, 0], X_new[:, 1], color='yellow',marker='+', s=100, alpha=0.4)
ax3.legend(['X','XPCA','Xinverzni'],loc='best')
ax3.set_title('Org. + PCA + Inv.Trasform.')
#######################                          MATRICA KOVARIJANSI
#Matrica kovarijansi
print('------------------------')
print('Kovarijansa:')
print(pca.get_covariance())
print('------------------------')
print('Objašnjena varijansa (apsolutne vrijednosti)')
print(pca.explained_variance_)
print('------------------------')
print('\nObjašnjena varijansa (procentualne vrijednosti)')
print(pca.explained_variance_ratio_)
print('------------------------')
###############                                    SVOJSTVENI VEKTORI
#Svojstveni vektor
print('Svojstvei vektori')
print(pca.components_)
print('------------------------')
######################                              CRTANJE VEKTORA i ORGINALNIH PODATAKA
#Crtanje vektora
def draw_vector(start_point, end_point, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', end_point, start_point, arrowprops=arrowprops)

# plot data
ax4=fig.add_subplot(154)
ax4.scatter(X[:, 0], X[:, 1], alpha=0.5)
ax4.legend(['X'],loc='best')
ax4.set_title('Org. sa PCA vektorima')
for length, vector in zip(pca.explained_variance_, pca.components_):
    print('length=',length)
    print('vector=',vector)
    v = vector * 3 * np.sqrt(length)
    print('v=',v)
    print('center point =',pca.mean_)
    end_point=pca.mean_ + v
    print('end_point=', end_point)
    print('-----------------------------------------------------')
    draw_vector(pca.mean_, pca.mean_ + v)
###############################################                        REDUKCIJA JEDNE KOMPONENTE
print('-----------------------------------------------------')
print('-----------------------------------------------------')
print('//////////////////////////////////////////////////////')
print('--------------Redukcija jedne komponente-------------')
print('-----------------------------------------------------')
pca = PCA(n_components=1)
pca.fit(X)
#############################################                         OBJASNJENA VARIJANSA REDUKOVANE
print('------------------------')
print("Objašnjena varijansa (procentualne vrijednosti)")
print(pca.explained_variance_ratio_)
print('------------------------')
#############################                                  GRAFIK ORGINALNIH, PCA, i INVERZNIH PODATAKA
ax5=fig.add_subplot(155)
#originalni podaci
plt.scatter(X[:, 0], X[:, 1], color='black',marker='+', s=100, alpha=1)
print('X shape')
print(X.shape)
print('------------------------')
#transformirani podaci - podaci u novom koordinatnom sistemu (koordinatni sistem PCA komponenata)
X_pca = pca.transform(X)
print('X shape transformisani')
print(X_pca.shape)
print('------------------------')
y_pca=np.zeros(X_pca.shape)
ax5.scatter(X_pca[:, 0],y_pca, color='red', s=100, alpha=0.5)

#povratak na originalne podatke, ali na osnovu transformiranih podataka
X_new = pca.inverse_transform(X_pca)
ax5.scatter(X_new[:, 0], X_new[:, 1], color='yellow',marker='v', s=100, alpha=0.4)
ax5.legend(['X','XPCA','Xinverzni'],loc='best')
ax5.set_title('Org+PCA sa redukc. + Inv.')

plt.show()