from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pyplot import mlab
import numpy as np
from pycbc import waveform
import copy
import sys

print "Usage: python pointlens <samplingFrequency> <outfile>"
print "Example: python pointlens 1024 Data/STFTData.npz"

fs      = int(sys.argv[1]) # Sampling frequency
outfile = sys.argv[2]

solarMass_Time=4.925e-6 # Solar mass in seconds http://www.wolframalpha.com/input/?i=solar+mass+*+G+%2Fc%5E3

def lensedWaveform(approximant="IMRPhenomPv2",mass1=10,mass2=10,distance=150,Mlz=6000*solarMass_Time,y=0.3,delta_t=1./4096.,f_lower=10):
  ''' Waveform lensed by a point mass lens, as in [1, Eq. (18)].

      :param approximant: Approximant name (as in pycbc)
      :param mass1:       1st compact object mass [Msun]
      :param mass2:       2nd compact object mass [Msun]
      :param Mlz:         Redshifted lens mass    [s]
      :param y:           Source position         [1] (see [1])
      :param delta_t:     Sampling rate           [s]
      :param f_lower:     Lower cutoff frequency  [Hz]

      :returns: Straints hp, hc

      [1] https://arxiv.org/pdf/astro-ph/0305055.pdf
  '''
  hp, hc = waveform.get_td_waveform(approximant=approximant,mass1=mass1,mass2=mass2,delta_t=delta_t,f_lower=f_lower,distance=distance)
  # Lens the waveform
  # Calculate time delay (see [1, Eq. (18)]):
  td = 4.*Mlz*(y*np.sqrt(y**2+4.)/2+np.log((np.sqrt(y**2+4.)+y)/(np.sqrt(y**2+4.)-y)))
  # Calculate the amplification:
  muPlus = 0.5 + (y**2+4.)/(2.*y*np.sqrt(y**2+4.))
  muMinus= 0.5 - (y**2+4.)/(2.*y*np.sqrt(y**2+4.))
  # Calculate the time shift in bins:
  time_shift = int(np.round(td/delta_t))
  # Error checking
  if time_shift > len(hp):
    raise ValueError("Time shift too large; these would result in two independent signals")
  # Calculate the delayed waveform:
  hp_delayed, hc_delayed = copy.deepcopy(hp), copy.deepcopy(hc)
  hp_delayed.data[:-time_shift] = hp.data[time_shift:]
  hc_delayed.data[:-time_shift] = hc.data[time_shift:]
  hp_delayed.data[-time_shift:] = np.zeros(time_shift)
  hc_delayed.data[-time_shift:] = np.zeros(time_shift)
  # Return the lensed waveform:
  hp_lensed, hc_lensed = copy.deepcopy(hp), copy.deepcopy(hc)
  hp_lensed.data = np.sqrt(np.abs(muPlus))*hp.data + np.sqrt(np.abs(muMinus))*hp_delayed.data
  hc_lensed.data = np.sqrt(np.abs(muPlus))*hc.data + np.sqrt(np.abs(muMinus))*hc_delayed.data
  return hp_lensed, hc_lensed

def stft(s, fs):
  ''' stft computed using the short time fourier transformr
      
      :param s: s(t)
      :param fs: Sampling rate
      :returns: (stft[complex], ff, t, im)
  '''
  # Spectral parameters
  NFFT = fs/8
  window = np.blackman(NFFT)
  noverlap = NFFT*15/16
  # Compute STFT
  stft, ff, tt = mlab.specgram(s, NFFT=NFFT, Fs=fs, window=window, noverlap=noverlap, mode='complex')
  return (stft, ff, tt)

hp, hc = lensedWaveform(delta_t=1./float(fs))
n        = np.random.normal(scale=5.e-22,size=len(hp))
signal   = hp+n # For simplicity
stfti, ff, tt = stft(signal,fs)

np.savez(outfile, stfti, ff, tt)

