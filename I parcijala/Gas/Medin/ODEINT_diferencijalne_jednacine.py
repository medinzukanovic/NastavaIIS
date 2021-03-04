import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#y'=u
# Transformisati izraz u dy/dt
#theta''(t) + b*theta'(t) + c*sin(theta(t)) = 0
def pend(y, t,a, b, c):
    dydt = -0.5*y+0.5*t+2
    return dydt
b = 0.3678
a=0.06
c = 31875
y0 = [2]
t = np.linspace(2, 10, 25000)
from scipy.integrate import odeint
sol = odeint(pend, y0, t, args=(a,b, c))
import matplotlib.pyplot as plt
print(sol)
plt.plot(t, sol, 'b', label='theta(t)')
#plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()