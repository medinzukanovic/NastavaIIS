#Logistička regresija
#Importovanje biblioteka
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
##################
#importovanje fajla
dataset = pd.read_csv(r'C:\Users\HP\Documents\Py Programs\IIS rok\Log_Reg.csv') 
########################
# input nezavisna varijabla
#prvo red pa kolona
x = dataset.iloc[:, [2, 3]].values 
# output zavisna varijabla
y = dataset.iloc[:, 4].values
###################
#Razdvajanje na train i test
test_velicina=0.01  #procentualno 

from sklearn.model_selection import train_test_split 
x_train, x_test, ytrain, ytest = train_test_split(x, y, test_size = test_velicina) 
#######################
#Skaliranje
from sklearn.preprocessing import minmax_scale
predidanje=np.array([31.2,51600]).reshape(1,-1)
predvidanje_skalirano=minmax_scale(predidanje)
xtrain = minmax_scale(x_train) 
xtest = minmax_scale(x_test) 
print(xtest)
#####################
#Regresija
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression() 
classifier.fit(xtrain, ytrain) 
##############################
#Predikcija
y_pred= classifier.predict(xtest) 
print(y_pred)
###########################
#Confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred) 

print ("Confusion Matrix : \n", cm) 
##############################
#Tačnost modela
from sklearn.metrics import accuracy_score 
print ("Tačnost : ", accuracy_score(ytest, y_pred)) 
#############################
#Malo bolji report
print('intercept_:', classifier.intercept_)
print('coef_:', classifier.coef_)
print('accuracy: {:.2f}%'.format(accuracy_score(ytest,y_pred)*100))
######################
#Predviđanje individualno

vrijednost_predvidanja=classifier.predict(predvidanje_skalirano)
print (vrijednost_predvidanja)
###################
#Crtanje grafika
sns.countplot(x=y, data = dataset)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
##########
#Grafik ravni
#print('xtest',xtest, x_test)
x1 = np.linspace(xtest[:, 0].min(), xtest[:, 0].max())
x2 = np.linspace(xtest[:, 1].min(), xtest[:, 1].max())
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
x1_mesh = x1_mesh.reshape(-1, 1)
x2_mesh = x2_mesh.reshape(-1, 1)

x_mesh = np.hstack((x1_mesh, x2_mesh))
y_predd = classifier.predict(x_mesh)
x1 = np.linspace(x_test[:, 0].min(), x_test[:, 0].max())
x2 = np.linspace(x_test[:, 1].min(), x_test[:, 1].max())
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
x1_mesh = x1_mesh.reshape(-1, 1)
x2_mesh = x2_mesh.reshape(-1, 1)
#Tačkasti
ax.scatter3D(x[:,0], x[:,1], y, c=y.ravel())
#Za povsinu
ax.plot_trisurf(x1_mesh.ravel(), x2_mesh.ravel(), y_predd.ravel(), alpha=0.6, shade=False)
plt.xlabel('X1')
plt.ylabel('X2')
ax.set_zlabel('Y')
plt.show()


