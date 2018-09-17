import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import pylab as p

################################################################################
#
#    PLOTTING OPTIONS
#
################################################################################

ratio = 0.0

# PLOTTING OPTIONS
fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = ratio if ratio != 0.0 else (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]

params = {'axes.labelsize': 10,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Raman',
          'font.size': 10,
          'legend.fontsize': 28,
          'xtick.labelsize': 32,
          'ytick.labelsize': 32,
          'axes.grid' : True,
          'text.usetex': True,
          'savefig.dpi' : 100,
          'lines.markersize' : 14, 
          'axes.formatter.useoffset': False,
          'figure.figsize': fig_size}

mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

mpl.rcParams.update(params)

