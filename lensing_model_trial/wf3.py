import numpy as np
import matplotlib.pyplot as plt

G = 6.674*(10**(-11))
c = 2.998*(10**(8))
p = 3.086*(10**16)

SM = 1.989*(10**30)
M_lz = 10000000*SM*G/(c**3)
D_l = 600*(10**6)*p/c
D_ls = 60*(10**6)*p/c
D_s = D_l + D_ls
eta = 0.1*p/c
z = 0


#mass of the black holes
M1 = 30*SM*G/(c**3)
M2 = M1

M = M1 + M2
m = (M1*M2)/(M1+M2)
n = m/M

def h(t,l=0):
    theta = n*(-t)/(5*M)
    phi = -(theta**(5/8))/n 
    x = 0.25*(theta)**(-1/4)
    return -8*np.sqrt(5/np.pi)*m*x*np.cos(phi+ l)/(D_s)

xi = np.sqrt((4*(M_lz/(1+z))*D_l*D_ls)/D_s)
y = (eta*D_l)/(xi*D_s)
   
t_delay = 4*M_lz*(0.5*y*np.sqrt(y**2+4) + np.log((np.sqrt(y**2 + 4) + y)/(np.sqrt(y**2 + 4) - y)))
    
t_vals = np.linspace(-10,-0,100000)
    
mu_plus = 0.5 + (((y**2) + 2)/(2*y*np.sqrt((y**2) + 4)))
mu_minus = 0.5 - ((y**2) + 2)/(2*y*np.sqrt((y**2) + 4))
    
hl1 = h(t_vals)
hl2 = h(t_vals, l = np.pi/2)

hl = np.sqrt(np.abs(mu_plus))*hl2
hl[t_vals <-t_delay] = np.sqrt(np.abs(mu_plus))*hl1[t_vals>-10 +t_delay]-np.sqrt(np.abs(mu_minus))*hl2[t_vals<-t_delay]


plt.figure()
#plt.plot(t_vals, hl, label = 'Lensed Waveform (z = ' + str(z) +')')
plt.plot(t_vals, h(t_vals),'--', label = 'Unlensed Waveform')


plt.grid()
plt.legend()
plt.show()
