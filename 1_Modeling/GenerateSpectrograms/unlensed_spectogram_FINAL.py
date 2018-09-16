# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 17:57:31 2018

@author: Ivan
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import mlab
import numpy as np
import random
import progressbar
import scipy.fftpack
import scipy.integrate
import time
from multiprocessing import Pool
startTime = time.time()

fs = 1024


progress = progressbar.ProgressBar().start()

#Physical constants
G    = 6.674*10**-11
c    = 2.998*10**8
p    = 3.086*10**16
SM   = 1.989*10**30
    
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
samples = 1000              # Change number of samples and directory for saving files
###########################

i = 0
repeats = 0
SNR_sums = []

def multi(spec_count)
    spec_counter = spec_count*100
    
    for i in range(spec counter, spec_counter+100):
        #Binary Parameters
        D    = np.linspace(10., 1000., 100)*(10**6)*p/c
        D_s  = D[random.randint(0, 99)]
        mr = np.linspace(4.,35.,50)*SM*G/(c**3)
        M1 = mr[random.randint(0,49)]
        M2 = mr[random.randint(0,49)]
        M  = M1 + M2
        m  = (M1*M2)/(M1+M2)
        n  = m/M
        
        #Gravitational Waveform
        def h(t, l=0):
            theta = n*(-t)/(5*M)
            phi   = -(theta**(5./8.))/n
            x     = 0.25*(theta)**(-1./4.)
            return -8*np.sqrt(5/np.pi)*m*x*np.cos(phi+l)/(D_s)
        t_lower = 50
        t_vals  = np.linspace(-t_lower,-0,t_lower*fs)
        
        h_unlensed = h(t_vals)
        
        for a in [-np.inf, np.inf, np.nan]:
            shift = 0
            for index, value in enumerate(h_unlensed):
                if value == a:
                    t_vals = np.delete(t_vals, index-shift)
                    h_unlensed = np.delete(h_unlensed, index-shift)
                    shift+=1
    
        abshlmin = abs(h_unlensed.min())
        abshlmax = abs(h_unlensed.max())
        
        if abshlmax > abshlmin:
            normf = abshlmax
        elif abshlmin > abshlmax:
            normf = abshlmin
    
        h_unlensed = h_unlensed/normf
    
        #Adding Gaussian noise to waveform
        n        = np.random.normal(scale=(5.e-22/normf), size=len(h_unlensed))
        signal   = h_unlensed + n # For simplicity
        stfti, ff, tt = stft(signal,fs)
        
        #Used for SNR calculations
        h_SNR   = np.copy(h_unlensed)
        n_SNR   = np.copy(n)
        t_SNR   = np.copy(t_vals)
    
        #Performing FFT on waveform and noise
        h_fft     = np.abs(scipy.fftpack.fft(h_SNR))
        h_fftfreq = scipy.fftpack.fftfreq(len(h_fft), d=1./fs)
        h_f       = h_fft[:len(h_fft)/2] #Getting rid of negative frequencies
        h_freq    = h_fftfreq[:len(h_fft)/2] 
        
        n_fft     = np.abs(scipy.fftpack.fft(n_SNR))
        n_fftfreq = scipy.fftpack.fftfreq(len(n_fft), d=1./fs)
        n_f       = n_fft[:len(n_fft)/2]
        n_freq    = n_fftfreq[:len(n_fft)/2]
        
        #Manual cutoff frequency
        h_f = h_f[101:]
        h_freq = h_freq[101:]
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
        y_integrand = h_f*h_f/(n_f*n_f) #Note n(f) has to be squared to give the PSD as it is calculated manually
        x_integrand = n_freq
        integral    = scipy.integrate.simps(y_integrand, x_integrand)
        delta_f = h_freq[1] - h_freq[0]
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
            plt.plot(t_vals, h_unlensed)
            plt.plot(t_vals, n, color='orange', alpha=0.7)
            """
            
            #Plotting and saving spectrogram data of incoming signal
            plt.figure()
            plotstft(stfti,tt,ff)
            plt.ylim([0,400])
            plt.savefig("/home/amitjit/output/spectrograms/Unlensed/Unlensed_" + str(spec_counter), bbox_inches=0, pad_inches= 0, transparent= True, dpi=50)
            plt.close()
            
            #Saving binary/lensing parameters in a txt file
            f = open("/home/amitjit/output/spectrograms/Unlensed/Unlensed_" + str(spec_counter) + ".txt", "w")
            f.write('Masses of merging black holes         = ' + str(M1*(c**3)/(SM*G))   + ' Solar Masses, ' + str(M2*(c**3)/(SM*G)) + ' Solar Masses\n\n')
            f.write('Distance from observer to source      = ' + str(D_s*c/p)     + ' parsecs\n')
            f.write('SNR                                   = ' + str(SNR_sum))
            f.close()
            
            progress.update(100*i/samples)
        else:
            repeats += 1 #repeats process if SNR condition not satisfied
            
    print('SNR average = %f' % np.average(SNR_sums))
    print('samples = %g, times repeated = %g times' % (samples, repeats))
    print('time = %s seconds' % (time.time() - startTime))


pool = Pool(10)
pool.map(multi, list(range(10)))
