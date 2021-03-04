import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#Import podataka
x1 = np.random.randn(1000,1)
x2 = np.random.randn(1000,1)
y = x1 + x2 + np.random.rand(1000,1) # slučajna greška
x = np.hstack((x1, x2))
print(x)
class MultivariateRegression():
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, x, y):
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        b1 = np.linalg.inv(np.dot(x.T, x))
        b2 = np.dot(x.T, y)
        b = np.dot(b1, b2)
        
        self.intercept_ = b[0]
        self.coef_ = b[1:]
    
    def predict(self, x):
        return np.dot(x, self.coef_) + self.intercept_
reg = MultivariateRegression()
reg.fit(x, y)

print(reg.intercept_)
print(reg.coef_)
x1_ = np.linspace(np.min(x1), np.max(x1), 10)
x2_ = np.linspace(np.min(x2), np.max(x2), 10)
x1v, x2v = np.meshgrid(x1_, x2_)
x1v, x2v = x1v.reshape(-1, 1), x2v.reshape(-1,1)

xv = np.hstack((x1v, x2v))
y_pred = reg.predict(xv)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1, x2, y)
ax.plot_trisurf(x1v.ravel(), x2v.ravel(), y_pred.ravel(), alpha=0.3, color='cyan', shade=False)
plt.show()