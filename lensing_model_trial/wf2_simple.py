import numpy as np
import matplotlib.pyplot as plt

Msun = 4.925e-6
Mpc  = 1.029e14

G = 6.674*(10**(-11))
c = 2.998*(10**(8))

SM = 1.989*(10**30) #solar mass
p = 3.086*(10**16)
M_lz = 1000*SM*G/(c**2)
D_l = 600*(10**6)*p
D_ls = 600*(10**6)*p
D_s = D_l + D_ls
eta = 0.1*p
z = 0

t = np.linspace(0,10,1000)
h = np.sin(t**2)

xi = np.sqrt((4*(M_lz/(1+z))*D_l*D_ls)/D_s)
y = (eta*D_l)/(xi*D_s)

t_delay = 4*M_lz*(0.5*y*np.sqrt(y**2+4) + np.log((np.sqrt(y**2 + 4) + y)/(np.sqrt(y**2 + 4) - y)))

mu_plus = 0.5 + (((y**2) + 2)/(2*y*np.sqrt((y**2) + 4)))
mu_minus = 0.5 - ((y**2) + 2)/(2*y*np.sqrt((y**2) + 4))

q = t + t_delay + np.pi/2
ht = np.sin(q**2)
hl = np.sqrt(np.abs(mu_plus))*h-np.sqrt(np.abs(mu_minus))*ht

plt.figure()
plt.plot(t, h,'k-', label = 'Unlensed Waveform')
plt.plot(t, hl,'b-', label = 'Lensed Waveform (z = ' + str(z) +')')
plt.grid()
plt.legend()
plt.show()
