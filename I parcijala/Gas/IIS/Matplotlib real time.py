import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd
style.use('fivethirtyeight')
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
def animate(i):
    graph_data=pd.read_excel('primjerrealtime.xlsx')
    xs=[]
    ys=[]
    for index, row in graph_data.iterrows():
        xs.append(row['X'])
        ys.append(row['Y'])
        ax1.plot(xs,ys)
ani=animation.FuncAnimation(fig,animate,interval=1)
plt.show()
