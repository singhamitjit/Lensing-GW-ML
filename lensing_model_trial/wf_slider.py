import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

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
    
t_vals = np.linspace(-21.5,-0,215000)
    
mu_plus = 0.5 + (((y**2) + 2)/(2*y*np.sqrt((y**2) + 4)))
mu_minus = 0.5 - ((y**2) + 2)/(2*y*np.sqrt((y**2) + 4))
    
hl1 = h(t_vals)
hl2 = h(t_vals, l = np.pi/2)

hl = np.sqrt(np.abs(mu_plus))*hl2
hl[t_vals <-t_delay] = np.sqrt(np.abs(mu_plus))*hl1[t_vals> -21.5+t_delay]-np.sqrt(np.abs(mu_minus))*hl2[t_vals<-t_delay]


#SLider

fig = plt.figure()
ax = fig.add_subplot(111)

fig.subplots_adjust(left=0.25, bottom = 0.25)

plot, = plt.plot(t_vals, hl, label = 'Lensed Waveform (z = ' + str(z) +')')
plt.plot(t_vals, h(t_vals),'--', label = 'Unlensed Waveform')


mlz_slider_ax = fig.add_axes([0.25, 0.15,0.65, 0.03])
mlz_slider = Slider(mlz_slider_ax, 'Redshifted Lens Mass', 0.1 , 8000000, valinit = M_lzz)

eta_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
eta_slider = Slider(eta_slider_ax, 'Distance of the source from the source and the line of sight', 0, 1, valinit = etan)



def update(val): 
    M_lz = mlz_slider.val*SM*G/(c**3)
    eta = eta_slider.val*p/c

    xi = np.sqrt((4*(M_lz/(1+z))*D_l*D_ls)/D_s)
    y = (eta*D_l)/(xi*D_s)
   
    t_delay = 4*M_lz*(0.5*y*np.sqrt(y**2+4) + np.log((np.sqrt(y**2 + 4) + y)/(np.sqrt(y**2 + 4) - y)))

    
    mu_plus = 0.5 + (((y**2) + 2)/(2*y*np.sqrt((y**2) + 4)))
    mu_minus = 0.5 - ((y**2) + 2)/(2*y*np.sqrt((y**2) + 4))
    
    hl1 = h(t_vals)
    hl2 = h(t_vals, l = np.pi/2)

    hl_new = np.sqrt(np.abs(mu_plus))*hl2
    hl_new[t_vals <-t_delay] = np.sqrt(np.abs(mu_plus))*hl1[t_vals>-21.5+t_delay]-np.sqrt(np.abs(mu_minus))*hl2[t_vals<-t_delay]
    plot.set_ydata(hl_new)
    fig.canvas.draw_idle()

mlz_slider.on_changed(update)
eta_slider.on_changed(update)

#Reset Button
reset_button_ax = fig.add_axes([0.025, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color = 'red', hovercolor = '0.975')

def reset(mouse):
    mlz_slider.reset()
    eta_slider.reset()

reset_button.on_clicked(reset)

plt.grid()
ax.legend()
plt.show()
