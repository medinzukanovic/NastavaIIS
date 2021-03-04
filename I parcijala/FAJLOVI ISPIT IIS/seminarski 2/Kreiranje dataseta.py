#importing the libraries
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
#setting the path to the directory containing the pics
path = r'C:\Users\HP\Documents\Py Programs\FAJLOVI ISPIT IIS\seminarski 2\Slike'
#appending the pics to the training data list
training_data = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path,img))
    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
    pic = cv2.resize(pic,(80,80))
    training_data.append([pic])
#converting the list to numpy array and saving it to a file using #numpy.save
np.save(os.path.join(path,'features.npy'),np.array(training_data))
#loading the saved file once again
saved = np.load(os.path.join(path,'features.npy'))
print(saved.shape)

x=saved.reshape(27,80*80*3)
target=['Medin','Medin','Medin','Medin','Medin','Medin','Medin','Medin','Medin','Medin','Medin','Medin','Dina','Dina','Dina','Dina','Dina','Dina','Dina','Dina','Dina','Dina','Dina','Dina','Dina','Dina','Dina']
from sklearn import svm
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(x, target)
probaj=(np.load(r'C:\Users\HP\Documents\Py Programs\FAJLOVI ISPIT IIS\seminarski 2\Medin\features.npy')).reshape(1,80*80*3)
probajj=(np.load(r'C:\Users\HP\Documents\Py Programs\FAJLOVI ISPIT IIS\seminarski 2\Medin\features.npy'))
print(clf.predict(probaj))
plt.imshow(probajj[0].reshape(80,80,3))
plt.show()