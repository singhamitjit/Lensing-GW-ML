import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0,10,1000)
tl = t + 3 + np.pi**0.5
h = np.sin(t**2)

hl = 3*np.sin(t**2) - 4*np.sin(tl**2) 

plt.figure()
plt.plot(t, h)
plt.plot(t, hl)
plt.show()

