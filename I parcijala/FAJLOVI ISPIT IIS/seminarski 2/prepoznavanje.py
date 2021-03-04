#importing the libraries
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
root=tk.Tk()
saved = np.load(r'C:\Users\HP\Documents\Py Programs\FAJLOVI ISPIT IIS\seminarski 2\Slike\train.npy')
koje=tk.StringVar()
x=saved.reshape(10,80*80*3)
target=['Dina','Dina','Dina','Dina','Dina','Medin','Medin','Medin','Medin','Medin']
from sklearn import svm
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(x, target)
def gas():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    winName = "image"
    cv2.namedWindow(winName)
    s, im = cam.read() # captures image
    cv2.imshow(winName, im) # displays captured image
    cv2.imwrite("test.jpg",im)
    cv2.waitKey()
    test_data = []
    pic = cv2.imread(r'C:\Users\HP\Documents\Py Programs\test.jpg')
    pic = cv2.resize(pic,(80,80))
    test_data.append([pic])
    #converting the list to numpy array and saving it to a file using #numpy.save
    np.save('test.npy',np.array(test_data))
    proba=np.load(r'C:\Users\HP\Documents\Py Programs\dina\train.npy').reshape(1,80*80*3)
    tekstic=clf.predict(proba)
    koje.set(tekstic)
tk.Button(root,text='Gas',command=gas).pack()
tk.Label(root,textvariable=koje).pack()
root.mainloop()