import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# database
from sklearn.datasets import load_iris

#ispis koliko redova i kola ima i nazivi kolona
iris = load_iris()
print(iris.data.shape)
print(iris.target.shape)
print(iris.feature_names)
print(iris.target_names)

#ispisivanje tabele sa vrijednostima csv-a
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['label'] = iris.target
# df['class'] = df['class'].map({0:iris.target_names[0], 1:iris.target_names[1], 2:iris.target_names[2]})
print (df)

# split the data table into data X and class labels Y
# ovdje mi je x od 0 do 4 vrijednosti, a cetvrta kolana y
x = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values
print(x.shape, y.shape)

#prikazivanje dijagrama
plt.figure(figsize=(12,6))
colors= ['blue', 'red', 'green']
for f in range(4):
    plt.subplot(2, 2, f+1)
    for label, color in zip(range(len(iris.target_names)), colors):
        plt.hist(x[y==label,f], label=iris.target_names[label], color=color, alpha=0.3)        
        plt.xlabel(iris.feature_names[f])
        plt.legend(loc='upper right', fancybox=True, fontsize=8)
plt.tight_layout()               
plt.show()

# Korak 1: Izračunavanje d-dimenzionalnog vektora srednjih vrijednosti

mean_vectors = []
for c in range(0,3):
    mean_vectors.append(np.mean(x[y==c], axis=0))
    print('Vektor srednjih vrijednosti {0}: {1}'.format(c, mean_vectors[c]))


# Korak 2: Izračunavanje kovarijacionih matrica

S_W = np.zeros((4,4))
for c,mv in zip(range(0,3), mean_vectors):
    class_sc_mat = np.zeros((4,4))                  # scatter matrix for every class
    for row in x[y == c]:
        row, mv = row.reshape(4,1), mv.reshape(4,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices
print('Kovarijaciona matrica 𝐒_𝐖 unutar klasa (within-class):\n', S_W)


overall_mean = np.mean(x, axis=0)

S_B = np.zeros((4,4))
for i, mean_vec in enumerate(mean_vectors):
    n = x[y==i,:].shape[0]
    mean_vec = mean_vec.reshape(4,1) # make column vector
    overall_mean = overall_mean.reshape(4,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    
print('Kovarijaciona matrica između klasa 𝐒_𝐁  (between classes):\n', S_B)

# Korak 3: Izračunavanje svojstvenih vektora i svojstvenih vrijednosti

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)
    print('\nSvojstveni vektor {0}: \n{1}'.format(i+1, eigvec_sc.real))
    print('Svojstvena vrijednost {:}: {:.2e}'.format(i+1, eig_vals[i].real))


# Korak 4: Izbor lineranih diskriminanti za novi podprostor (podskup) svojstava

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print (eig_pairs)

print('Svojstvene vrijednosti (max->min):')
for i in eig_pairs:
    print(i[0])


print('Objašnjena varijansa')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('svojstvena vrijednost {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrica W:', W.real, sep='\n')

# Korak 5: Transformacija uzoraka u novi prostor

Y = x.dot(W)
Y.shape
#print(Y)

def plot_step_lda():
    label_dict = {0: 'Setosa', 1: 'Versicolor', 2:'Virginica'}
    
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(0,3),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=Y[:,0].real[y == label], y=Y[:,1].real[y == label], marker=marker, color=color, alpha=0.5, \
                    label=label_dict[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom=False, top=False,  
            labelbottom=True, left=False, right=False, labelleft=True)

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()


