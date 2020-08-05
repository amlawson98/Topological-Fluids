import numpy as np
import matplotlib.pyplot as plt
def drv_x(arr):
	return (np.roll(arr, 1, axis = 0) - np.roll(arr, -1, axis = 0))/(2*x_step)

x_step = 2*np.pi /200
y = np.arange(0, 2*np.pi, x_step)
x = np.sin(y)
dx  = drv_x(x)
d2x = drv_x(dx)

plt.plot(y, x)
plt.plot(y, d2x)
plt.show()