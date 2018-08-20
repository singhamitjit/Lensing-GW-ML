import matplotlib.pyplot as plt
import numpy as np
import sys

print "usage: pointlensplot <infile> <style: abs/re/im/arg/old> <outfile>"
print "usage: pointlensplot Data/STFTData.npz abs absSTFT.png"
print "usage: pointlensplot Data/STFTData.npz re realSTFT.png"
print "usage: pointlensplot Data/STFTData.npz im imagSTFT.png"
print "usage: pointlensplot Data/STFTData.npz arg argSTFT.png"

infile  = sys.argv[1]
style   = sys.argv[2]
outfile = sys.argv[3]

npzfile = np.load(infile)
stfti, ff, tt = npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2']

def plotstft(stft, t, ff, style='abs'):
  if style == 'abs':
    plotted = np.abs(stft)
  elif style == 're':
    plotted = np.real(stft)
  elif style == 'im':
    plotted = np.imag(stft)
  elif style == 'arg':
    plotted = np.angle(stft)
  elif style == 'old':
    plotted = 10*np.log10(np.abs(stft))
  fig,ax=plt.subplots()
  mesh = ax.pcolormesh(t,ff,plotted,cmap='gist_earth',shading='gouraud')
  ax.axis('tight')
  fig.colorbar(mesh)

plotstft(stfti,tt,ff,style=style)
plt.ylim([0,350])
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("%s(stft)"%style)
plt.savefig(outfile,bbox_inches='tight',dpi=200)
plt.close()


