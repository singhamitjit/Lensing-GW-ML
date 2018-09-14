import matplotlib.pyplot as plt
from matplotlib.pyplot import mlab
import numpy as np
import random

fs = 1024

# SI units?
G    = 6.674*10**-11
c    = 2.998*10**8
p    = 3.086*10**16

SM   = 1.989*10**30
Mlz  = np.linspace(10, 10000000,10000)*SM*G/(c**3)
M_lz = Mlz[random.randint(0,9999)]
D    = np.linspace(10, 1000, 100)*(10**6)*p/c
D_l  = D[random.randint(0,99)]
D_ls = D[random.randint(0,99)]
D_s  = D_l + D_ls
e    = np.linspace(1.e-6,0.5,100)*p/c
eta  = e[random.randint(0,99)]
zr   = np.linspace(0,2,100)
z    = zr[random.randint(0,99)]

#Mass of black holes
mr = np.linspace(4,35,50)*SM*G/(c**3)
M1 = mr[random.randint(0,49)]
M2 = mr[random.randint(0,49)]

M  = M1 + M2
m  = (M1*M2)/(M1+M2)
n  = m/M

# We may want to replace this with the pycbc waveform
# Gravitational Waveform
def h(t, l=0):
    ''' Gravitational waveform

        :param t: Time series np array
        :param l: Time of coalescence?

        :returns: Returns the waveform h(t)

    '''
    theta = n*(-t)/(5*M)
    phi   = -(theta**(5/8))/n
    x     = 0.25*(theta)**(-1/4)
    return -8*np.sqrt(5/np.pi)*m*x*np.cos(phi+l)/(D_s)


# See Jolien's book
xi  = np.sqrt((4*(M_lz/(1+z))*D_l*D_ls)/D_s)
y   = (eta*D_l)/(xi*D_s)
t_delay = 4*M_lz*(0.5*y*np.sqrt(y**2+4) + np.log((np.sqrt(y**2 + 4) + y)/(np.sqrt(y**2 + 4) - y)))
t_lower = 50
t_vals_lensed  = np.linspace(-t_lower,-0,t_lower*fs)
t_vals_unlensed  = np.linspace(-t_lower,-0,t_lower*fs)

# Takahashi et al. 2001
# Gives the magnifications mu_+, mu_-
mu_plus  = 0.5 + ((y**2 + 2)/(2*y*np.sqrt(y**2 + 4)))
mu_minus = 0.5 - ((y**2 + 2)/(2*y*np.sqrt(y**2 + 4)))


# Create unlensed signal
h_unlensed = h(t_vals_unlensed)

# Delete part of the event so that the lensed waveform won't go outside of the band
for a in [-np.inf, np.inf, np.nan]:
    shift = 0
    for index, value in enumerate(h_unlensed):
        if value == a:
            t_vals_unlensed = np.delete(t_vals_unlensed, index-shift)
            h_unlensed = np.delete(h_unlensed, index-shift)
            shift+=1

# Normalize waveform?
abshmin = abs(h_unlensed.min())
abshmax = abs(h_unlensed.max())
if abshmax > abshmin:
    unnormf = abshmax
elif abshmin > abshmax:
    unnormf = abshmin
h_unlensed = h_unlensed/unnormf

# Create two lensed waveforms
hl1 = h(t_vals_lensed)
hl2 = h(t_vals_lensed, l = np.pi/2)
# Apply shift (time-delay) and magnification
hl = np.sqrt(np.abs(mu_plus))*hl2
hl[t_vals_lensed< -t_delay] = np.sqrt(np.abs(mu_plus))*hl1[t_vals_lensed > -t_lower + t_delay] - np.sqrt(np.abs(mu_minus))*hl2[t_vals_lensed < -t_delay]

# Delete part of the lensed waveform
for a in [-np.inf, np.inf, np.nan]:
    shift = 0
    for index, value in enumerate(hl):
        if value == a:
            t_vals_lensed = np.delete(t_vals_lensed, index-shift)
            hl = np.delete(hl, index-shift)
            shift+=1

# Normalize lensed waveform
abshlmin = abs(hl.min())
abshlmax = abs(hl.max())
if abshlmax > abshlmin:
    normf = abshlmax
elif abshlmin > abshlmax:
    normf = abshlmin
hl = hl/normf

# Unlensed and lensed time arrays
t_vals_lensed = abs(np.flip(t_vals_lensed,0))
t_vals_unlensed = abs(np.flip(t_vals_unlensed,0))

# Plot
plt.plot(t_vals_unlensed, h_unlensed,'--', linewidth=0.5, label='Unlensed')
plt.plot(t_vals_lensed, hl,'--', linewidth=0.5, label='Lensed')
plt.title('Gravitational Waveforms (Normalised) Without Noise')
plt.legend()
plt.xlabel('Time(s)')
plt.axis('tight')
plt.ylabel('Normalized Strain')
# Add a relative path 
plt.savefig('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/2018 Internship/Python/python/Final/Gravitational Waveforms (Normalised) 4',dpi=500) # PDF file -- if possible
plt.close()
