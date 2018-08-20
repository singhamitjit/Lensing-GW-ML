import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

print('Mass in solar masses and distance in parsec')

G = 6.674*10**-11
c = 2.998*10**8
h_t  = input('Waveform: ')
M_lz = float(input('Redshifted mass: '))*G/(c**3)
D_l = float(input('Distance between observer and lens: '))/c
D_ls = float(input('Distace betweem source and lens: '))/c
D_s = D_l + D_ls
eta = float(input('Distance between line-of-sight and the source: '))/c
t_vals = np.linspace(0, 10, 1000)

z= float(input('Enter redshift: '))

xi_0 = ((4*(M_lz/(1+z))*D_l*D_ls)/D_s)**(1/2)

y = (eta*D_l)/(xi_0*D_s)

t_delay = 4*M_lz*(0.5*y*np.sqrt((y**2) + 4)+np.log((np.sqrt(y**2 +4) + y)/(np.sqrt(y**2 + 4)- y)))

mu_plus = 1/2 + (y**2 + 2)/(2*y*np.sqrt(y**2 + 4))
mu_minus = 1/2 - (y**2 + 2)/(2*y*np.sqrt(y**2 + 4))

def h(t):
    return eval(h_t)

def hl(t):
    return np.sqrt(np.abs(mu_plus))*h(t)-np.sqrt(np.abs(mu_minus))*h(t + t_delay + np.pi/2)


plt.figure(figsize=(10,8), dpi=100)

plt.plot(t_vals, hl(t_vals), label = 'Lensed Waveform (redshift z = ' + str(z) + ')')
plt.grid()
plt.legend()
plt.show()
