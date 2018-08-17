import numpy as np
import matplotlib.pyplot as plt

Mp = float(input('Mu_plus:'))
Mm = float(input('Mu_minus:'))
dT = float(input('delta T:'))
wf = input('Waveform:')

t_vals = np.linspace(0,10,1000)
 

def h(t):
    t
    return eval(wf)
c = t_vals + dT + np.pi/2

hl = Mp*h(t_vals)-Mm*h(c)

plt.figure(figsize=(10,8), dpi=100)
plt.plot(t_vals, h(t_vals), 'b', label="Unlensed Waveform")
plt.plot(t_vals, hl,'k', label="Lensed Waveform")
plt.legend()
plt.show()
