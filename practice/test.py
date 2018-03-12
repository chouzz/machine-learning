import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 1000)
y = -x*np.log2(x)

plt.plot(x,y)
plt.show()