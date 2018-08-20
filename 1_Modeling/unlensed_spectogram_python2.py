import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import mlab
import numpy as np
import random
import progressbar

fs = 1024
spec_counter = 0

progress = progressbar.ProgressBar()

for i in range(spec_counter,4000):
    G    = 6.674*10**-11
    c    = 2.998*10**8
    p    = 3.086*10**16

    SM   = 1.989*10**30
    D    = np.linspace(20, 2000, 200)*(10**6)*p/c
    D_s  = D[random.randint(0, 199)]
    

    #Mass of black holes
    mr = np.linspace(4,35,50)*SM*G/(c**3)
    M1 = mr[random.randint(0,49)]
    M2 = mr[random.randint(0,49)]

    M  = M1 + M2
    m  = (M1*M2)/(M1+M2)
    n  = m/M
        
        
    #Gravitational Waveform
    def h(t, l=0):
        theta = n*(-t)/(5*M)
        phi   = -(theta**(5/8))/n
        x     = 0.25*(theta)**(-1/4)
        return -8*np.sqrt(5/np.pi)*m*x*np.cos(phi+l)/(D_s)
    
    t_lower = 50
    t_vals  = np.linspace(-t_lower,-0,t_lower*fs)

    h_unlensed = h(t_vals)
    
    spec_counter+= 1

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

    def stft(s, fs):
      ''' stft computed using the short time fourier transform
          
          :param s: s(t)
          :param fs: Sampling rate
          :returns: (stft[complex], ff, t, im)
      '''
      # Spectral parameters
      NFFT = fs/8
      window = np.blackman(NFFT)
      noverlap = NFFT*15/16
      # Compute STFT
      stft, ff, tt = mlab.specgram(s, NFFT=int(NFFT), Fs=fs, window=window, noverlap=int(noverlap), mode='complex')
      return (stft, ff, tt)


    n        = np.random.normal(scale= (5.e-22/normf) , size=len(h_unlensed))
    signal   = h_unlensed + n # For simplicity
    stfti, ff, tt = stft(signal,fs)

    def plotstft(s, t, f):
        plotted = np.abs(s)
        fig,ax=plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        mesh = ax.pcolormesh(t,f,plotted,cmap='gist_earth',shading='gouraud')
        ax.axis('tight')
        ax.axis('off')
     
    plotstft(stfti,tt,ff)
    plt.ylim([0,400])
    plt.savefig("/home/amitjit/output/spectrograms/Unlensed/Unlensed_" + str(spec_counter), bbox_inches=0, pad_inches= 0, transparent= True, dpi=50)
    plt.close()
    
    
    f = open("/home/amitjit/output/spectrograms/Unlensed/Unlensed_" + str(spec_counter) + ".txt", "w")
    
    f.write('Masses of merging black holes         = ' + str(M1*(c**3)/(SM*G))   + ' Solar Masses, ' + str(M2*(c**3)/(SM*G)) + ' Solar Masses\n\n')
    f.write('Distance from observer to source      = ' + str(D_s*c/p)     + ' parsecs\n')
   
    f.close()
