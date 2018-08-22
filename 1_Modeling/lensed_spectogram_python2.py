import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import mlab
import numpy as np
import random
from multiprocessing import Pool
import pycbc.noise
import pycbc.psd

def multi(spec_count):
    fs = 1024
    spec_counter = spec_count*1000

    for i in range(spec_counter, spec_counter+1000):
        #Variables
        G    = 6.674*10**-11
        c    = 2.998*10**8
        p    = 3.086*10**16

        SM   = 1.989*10**30
        Mlz  = np.logspace(1, 7,10000)*SM*G/(c**3)
        M_lz = Mlz[random.randint(0,9999)]
        D    = np.linspace(10, 1000, 100)*(10**6)*p/c
        D_l  = D[random.randint(0,99)]
        D_ls = D[random.randint(0,99)]
        D_s  = D_l + D_ls
        e    = np.linspace(1.e-6,0.5,100)*p/c
        eta  = e[random.randint(0,99)]
        zr   = np.linspace(0,2,100)
        z    = zr[random.randint(0,99)]

        #Mass of binaries
        mr = np.linspace(4,35,50)*SM*G/(c**3)
        M1 = mr[random.randint(0,49)]
        M2 = mr[random.randint(0,49)]

        M  = M1 + M2
        m  = (M1*M2)/(M1+M2)
        n  = m/M
        

        #Gravitational Waveform
        def h(t, l=0):
            theta = n*(-t)/(5.0*M)
            phi   = -(theta**(5./8.))/n
            x     = 0.25*(theta)**(-1./4.)
            return -8.0*np.sqrt(5.0/np.pi)*m*x*np.cos(phi+l)/(D_s)
        xi  = np.sqrt((4.0*(M_lz/(1+z))*D_l*D_ls)/D_s)
        y   = (eta*D_l)/(xi*D_s)
        t_delay = 4*M_lz*(0.5*y*np.sqrt(y**2+4) + np.log((np.sqrt(y**2 + 4.0) + y)/(np.sqrt(y**2 + 4) - y)))
        t_lower = 50
        t_vals  = np.linspace(-t_lower,-0,t_lower*fs)

        if t_delay>t_lower:
            print str(i+1) + ': Time delay too large'
            continue


        if 0.05>=y>=0.3:
            print 'y is out of range'
            continue

        spec_counter+= 1

        mu_plus  = 0.5 + ((y**2 + 2)/(2*y*np.sqrt(y**2 + 4)))
        mu_minus = 0.5 - ((y**2 + 2)/(2*y*np.sqrt(y**2 + 4)))
        
        hl1 = h(t_vals)
        hl2 = h(t_vals, l = np.pi/2)
        
       
        hl = np.sqrt(np.abs(mu_plus))*hl2
        hl[t_vals < -t_delay] = np.sqrt(np.abs(mu_plus))*hl1[t_vals > -t_lower + t_delay] - np.sqrt(np.abs(mu_minus))*hl2[t_vals < -t_delay]


        flow = 30.0
        delta_f = 1.0/ 16
        flen = int(2048 / delta_f) + 1
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

        delta_t  = 1.0 / fs
        tsamples = int(t_lower / delta_t)
        n        = pycbc.noise.noise_from_psd(tsamples, delta_t, psd)


        for a in [-np.inf, np.inf, np.nan]:
            shift = 0
            for index, value in enumerate(hl):
                if value == a:
                    t_vals = np.delete(t_vals, index-shift)
                    hl = np.delete(hl, index-shift)
                    n = np.delete(n, index-shift)
                    shift+=1

        abshlmin = abs(hl.min())
        abshlmax = abs(hl.max())

        if abshlmax > abshlmin:
            normf = abshlmax
        elif abshlmin > abshlmax:
            normf = abshlmin

        hl = hl/normf

        def stft(s, fs):
          ''' stft computed using the short time fourier transform
              
              :param s: s(t)
              :param fs: Sampling rate
              :returns: (stft[complex], ff, t, im)
          '''
          # Spectral parameters
          NFFT = fs/8.0
          window = np.blackman(NFFT)
          noverlap = NFFT*15.0/16.0
          # Compute STFT
          stft, ff, tt = mlab.specgram(s, NFFT=int(NFFT), Fs=fs, window=window, noverlap=int(noverlap), mode='complex')
          return (stft, ff, tt)
        
        signal   = hl + n/normf
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
        plt.savefig("/home/amitjit/output/spectrograms/PM/Lensed_" + str(spec_counter), bbox_inches=0, pad_inches= 0, transparent= True, dpi=50)
        plt.close()
        
        
        f = open("/home/amitjit/output/spectrograms/PM/Lensed_" + str(spec_counter) + ".txt", "w")
        
        f.write('Masses of merging black holes         = ' + str(M1*(c**3)/(SM*G))   + ' Solar Masses, ' + str(M2*(c**3)/(SM*G)) + ' Solar Masses\n\n')
        f.write('Mass of lens                          = ' + str(M_lz*(c**3)/(SM*G))    + ' Solar Masses\n')
        f.write('Distance from lens to observer        = ' + str(D_l*c/p)     + ' parsecs\n')
        f.write('Distance from lens to source          = ' + str(D_ls*c/p)    + ' parsecs\n')
        f.write('Distance from observer to source      = ' + str(D_s*c/p)     + ' parsecs\n')
        f.write('Distance of source from line of sight = ' + str(eta*c/p)     + ' parsecs\n')
        f.write('Redshift(z)                           = ' + str(z)           + '\n\n'      )
        f.write('Time delay                            = ' + str(t_delay)     + ' s\n'      )
        f.write('y                                     = ' + str(y)           + '\n'      )
        f.write('mu_plus                               = ' + str(mu_plus)     + '\n'      )
        f.write('mu_minus                              = ' + str(mu_minus)    + '\n'      )
        f.close()
        

pool = Pool(10)
pool.map(multi, list(range(20)))
