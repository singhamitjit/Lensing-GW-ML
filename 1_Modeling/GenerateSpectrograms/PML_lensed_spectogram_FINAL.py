# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:36:48 2018

@author: Ivan
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import mlab
import numpy as np
import random
import progressbar
import scipy.fftpack
import scipy.integrate
import time

startTime = time.time()

fs = 1024
spec_counter = 0

progress = progressbar.ProgressBar().start()

#Physical constants
G    = 6.674*10**-11
c    = 2.998*10**8
p    = 3.086*10**16
SM   = 1.989*10**30

#Gravitational Waveform
def h(t, l=0):
    theta = n*(-t)/(5*M)
    phi   = -(theta**(5./8.))/n
    x     = 0.25*(theta)**(-1./4.)
    return -8*np.sqrt(5/np.pi)*m*x*np.cos(phi+l)/(D_s)
    
#Functions for STFT and generating spectrogram
def stft(s, fs):
      ''' stft computed using the short time fourier transform
      
      :param s: s(t)
      :param fs: Sampling rate
      :returns: (stft[complex], ff, t, im)
      '''
      NFFT = fs/8.
      window = np.blackman(NFFT)
      noverlap = NFFT*15./16.
      stft, ff, tt = mlab.specgram(s, NFFT=int(NFFT), Fs=fs, window=window, noverlap=int(noverlap), mode='complex')
      return (stft, ff, tt)
      
def plotstft(s, t, f):
    plotted = np.abs(s)
    fig,ax=plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    mesh = ax.pcolormesh(t,f,plotted,cmap='gist_earth',shading='gouraud')
    ax.axis('tight')
    ax.axis('off')
            
###########################
samples = 100              # Change number of samples and directory for saving files
###########################

i = 0
repeats = 0
SNR_sums = []

while i < samples:
    #Binary and lensing parameters
    Mlz  = np.logspace(3, 7,10000)*SM*G/(c**3)
    M_lz = Mlz[random.randint(0,9999)]
    D    = np.linspace(10., 1000., 100)*(10**6)*p/c
    D_l  = D[random.randint(0,99)]
    D_ls = D[random.randint(0,99)]
    D_s  = D_l + D_ls
    e    = np.linspace(1.e-6,0.5,100)*p/c
    eta  = e[random.randint(0,99)]
    zr   = np.linspace(0.,2.,100)
    z    = zr[random.randint(0,99)]
    
    mr = np.linspace(4.,35.,50)*SM*G/(c**3)
    M1 = mr[random.randint(0,49)]
    M2 = mr[random.randint(0,49)]
    M  = M1 + M2
    m  = (M1*M2)/(M1+M2)
    n  = m/M
    
    xi  = np.sqrt((4*(M_lz/(1+z))*D_l*D_ls)/D_s)
    y   = (eta*D_l)/(xi*D_s)

    #Generate spectrogram only if y is small enough
    if y >= 0.05:
        t_lower = 50
        t_vals  = np.linspace(-t_lower,-0,t_lower*fs)
        
        #Calculating magnification factors and time delay
        beta = np.sqrt(y*y + 4)
        k    = (y*y + 2)/(2*y*beta)
        mu_plus  = 0.5 + k
        mu_minus = 0.5 - k
        t_delay = 4*M_lz*((0.5*y*beta) + np.log((beta + y)/(beta - y)))
        
        #Generating waveform and applying magnification factor and time delay
        hl1 = h(t_vals)
        hl2 = h(t_vals, l = np.pi/2)
        
        hl = np.sqrt(np.abs(mu_plus))*hl2
        hl[t_vals < -t_delay] = np.sqrt(np.abs(mu_plus))*hl1[t_vals > -t_lower + t_delay] - np.sqrt(np.abs(mu_minus))*hl2[t_vals < -t_delay]
        
        for a in [-np.inf, np.inf, np.nan]:
            shift = 0
            for index, value in enumerate(hl):
                if value == a:
                    t_vals = np.delete(t_vals, index-shift)
                    hl = np.delete(hl, index-shift)
                    shift+=1
    
        abshlmin = abs(hl.min())
        abshlmax = abs(hl.max())
        
        if abshlmax > abshlmin:
            normf = abshlmax
        elif abshlmin > abshlmax:
            normf = abshlmin
    
        hl = hl/normf #Normalisation
        
        #Adding noise to waveform
        n        = np.random.normal(scale=(5.e-22/normf), size=len(hl))
        signal   = hl + n # For simplicity
        stfti, ff, tt = stft(signal,fs)
        
        #Getting rid of delayed part of wave
        hl_SNR  = hl[t_vals < -t_delay]
        n_SNR   = n[t_vals < -t_delay]
        t_SNR   = t_vals[t_vals < -t_delay]
    
        """
        #Plotting waveform without delayed part
        plt.figure()
        plt.plot(t_SNR, hl_SNR)
        plt.plot(t_SNR, n_SNR, color='orange', alpha=0.7)
        """
        
        #Performing FFT on waveform and noise
        hl_fft     = np.abs(scipy.fftpack.fft(hl_SNR))
        hl_fftfreq = scipy.fftpack.fftfreq(len(hl_fft), d=1./fs)
        hl_f       = hl_fft[:len(hl_fft)/2] #Getting rid of negative frequencies
        hl_freq    = hl_fftfreq[:len(hl_fft)/2] 
        
        n_fft     = np.abs(scipy.fftpack.fft(n_SNR))
        n_fftfreq = scipy.fftpack.fftfreq(len(n_fft), d=1./fs)
        n_f       = n_fft[:len(n_fft)/2]
        n_freq    = n_fftfreq[:len(n_fft)/2]
        
        #Manual cutoff frequency
        hl_f = hl_f[101:]
        hl_freq = hl_freq[101:]
        n_f = n_f[101:]
        n_freq = n_freq[101:]
        
        """
        #Plot GW and noise in frequency domain
        plt.figure()
        plt.plot(n_freq, n_f, color='orange')
        plt.plot(hl_freq, hl_f, color='blue')
        plt.xlim([2, 5000])
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(('Sn(f)^1/2', 'h(f)'))
        plt.xlabel('frequency (Hz)')
        plt.ylabel('strain')
        """
        
        #Calculating SNR
        y_integrand = hl_f*hl_f/(n_f*n_f) #Note n(f) has to be squared to give the PSD as it is calculated manually
        x_integrand = n_freq
        integral    = scipy.integrate.simps(y_integrand, x_integrand)
        delta_f = hl_freq[1] - hl_freq[0]
        SNR_sum  = np.sqrt(4*np.sum(y_integrand)*delta_f)
        
        #Only using result if SNR is within the desirable limit
        if SNR_sum <= 80:
            i += 1
            spec_counter += 1
            #print('SNR = %f' % SNR_sum)
            SNR_sums.append(SNR_sum)
                
            """
            #Plot GW and noise in time domain
            plt.figure()
            plt.plot(t_vals, hl)
            plt.plot(t_vals, n, color='orange', alpha=0.7)
            """
            
            #Plotting and saving spectrogram data of incoming signal
            plt.figure()
            plotstft(stfti,tt,ff)
            plt.ylim([0,400])
            plt.savefig("C:\Users\Ivan\Desktop\Gravitational Lensing CUHK\Amit\PML_Lensed3\PML_Lensed_" + str(spec_counter), bbox_inches=0, pad_inches= 0, transparent= True, dpi=50)
            plt.close()
            
            #Saving binary/lensing parameters in a txt file
            f = open("C:\Users\Ivan\Desktop\Gravitational Lensing CUHK\Amit\PML_Lensed3\PML_Lensed_" + str(spec_counter) + ".txt", "w")
            f.write('Masses of merging black holes         = ' + str(M1*(c**3)/(SM*G))   + ' Solar Masses, ' + str(M2*(c**3)/(SM*G)) + ' Solar Masses\n\n')
            f.write('Mass of lens                          = ' + str(M_lz*(c**3)/(SM*G))    + ' Solar Masses\n')
            f.write('Distance from lens to observer        = ' + str(D_l*c/p)     + ' parsecs\n')
            f.write('Distance from lens to source          = ' + str(D_ls*c/p)    + ' parsecs\n')
            f.write('Distance from observer to source      = ' + str(D_s*c/p)     + ' parsecs\n')
            f.write('Distance of source from line of sight = ' + str(eta*c/p)     + ' parsecs\n')
            f.write('Redshift (z)                          = ' + str(z)       + '\n\n')
            f.write('Time delay                            = ' + str(t_delay) + ' s\n\n\n')
            f.write('SNR                                   = ' + str(SNR_sum))
            f.close()
            
            progress.update(100*i/samples)
    else:
        repeats += 1 #repeats process if y or SNR condition not satisfied
        
print('SNR average = %f' % np.average(SNR_sums))
print('samples = %g, times repeated = %g times' % (samples, repeats))
print('time = %s seconds' % (time.time() - startTime))