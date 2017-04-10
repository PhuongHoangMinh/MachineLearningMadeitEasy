from numpy import *
from matplotlib.pyplot import *

#drawing a basic function
x = linspace(0, 4, 100)
y =  sin(math.pi*x)#exp(-x)*sin(2*pi*x)
plot(x, y, '-b')
xlabel('x'); ylabel('y')
matplotlib.pyplot.show()

import numpy as np
import matplotlib.pyplot as plt

#generating spiral data
N = 100 #number of points
K = 3   #number of classes
D = 2   #number of dimensionality
X = zeros((N*K, D)).astype("float32")
Y = zeros((N*K,), dtype = 'uint8')

for j in range(K):
    ix = range(N*j, N*(j+1))
    r  = linspace(0.0, 1, N) #radius
    theta = linspace(j*4, (j+1)*4, N) + random.randn(N)*0.2
    X[ix] = np.c_[r*np.sin(theta), r*np.cos(theta)]
    Y[ix] = j

plt.scatter(X[:, 0], X[:, 1], s = 20, c = Y, cmap=plt.cm.Spectral)
plt.show()


"""
Simple demo of a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()


"""
Pyplot animation example.

The method shown here is only for very simple, low-performance
use.  For more demanding applications, look at the animation
module and the examples that use it.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(6)
y = np.arange(5)
z = x * y[:, np.newaxis]

for i in range(5):
    if i == 0:
        p = plt.imshow(z)
        fig = plt.gcf()
        plt.clim()   # clamp the color limits
        plt.title("Boring slide show")
    else:
        z = z + 2
        p.set_data(z)

    print("step", i)
    plt.pause(0.5)
