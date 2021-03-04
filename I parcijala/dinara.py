import matplotlib.pyplot as plt
import numpy as np
x=np.zeros(10)
y=np.zeros(10)
plt.scatter(x,y)
plt.xlim(0,5000)
plt.ylim(0,80000)
plt.ylabel('W/m')
plt.xlabel('m')
plt.title('Raspodjela specifične snage')
plt.show()