import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 6.674*(10**(-11))
c = 2.998*(10**(8))
p = 3.086*(10**16)

SM = 1.989*(10**30)
M_lzz = 10000
M_lz = M_lzz*SM*G/(c**3)
D_l = 600*(10**6)*p/c
D_ls = 600*(10**6)*p/c
D_s = D_l + D_ls
etan = 0.01
eta = etan*p/c
zn = 0
z = zn


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
    
t_vals = np.linspace(-40,-0,400000)
    
mu_plus = 0.5 + (((y**2) + 2)/(2*y*np.sqrt((y**2) + 4)))
mu_minus = 0.5 - ((y**2) + 2)/(2*y*np.sqrt((y**2) + 4))
    
hl1 = h(t_vals)
hl2 = h(t_vals, l = np.pi/2)

hl = np.sqrt(np.abs(mu_plus))*hl2
hl[t_vals <-t_delay] = np.sqrt(np.abs(mu_plus))*hl1[t_vals> -40+t_delay]-np.sqrt(np.abs(mu_minus))*hl2[t_vals<-t_delay]


#ANIMATION

fig, ax = plt.subplots()

plot, = plt.plot(t_vals, hl, label = 'Lensed Waveform (z = ' + str(z) +')')
plt.plot(t_vals, h(t_vals),'--', label = 'Unlensed Waveform')

def animate(i): 
    eta = i*p/c

    xi = np.sqrt((4*(M_lz/(1+z))*D_l*D_ls)/D_s)
    y = (eta*D_l)/(xi*D_s)
   
    t_delay = 4*M_lz*(0.5*y*np.sqrt(y**2+4) + np.log((np.sqrt(y**2 + 4) + y)/(np.sqrt(y**2 + 4) - y)))

    
    mu_plus = 0.5 + (((y**2) + 2)/(2*y*np.sqrt((y**2) + 4)))
    mu_minus = 0.5 - ((y**2) + 2)/(2*y*np.sqrt((y**2) + 4))
    
    hl1 = h(t_vals)
    hl2 = h(t_vals, l = np.pi/2)

    hl_new = -np.sqrt(np.abs(mu_plus))*hl2
    hl_new[t_vals <-t_delay] = np.sqrt(np.abs(mu_plus))*hl1[t_vals>-40+t_delay]-np.sqrt(np.abs(mu_minus))*hl2[t_vals<-t_delay]
    plot.set_ydata(hl_new)
    return plot,

ani = animation.FuncAnimation(fig, animate,interval=0.1)


plt.grid()
ax.legend()
plt.show()
