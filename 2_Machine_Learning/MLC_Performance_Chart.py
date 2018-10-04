# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 03:47:15 2018

@author: Ivan
"""

import matplotlib.pyplot as plt
import numpy as np

from graphicsSettings import * # Fancy graphics
import sys

ind = [-0.25, 0.25, 1.25, 1.75, 2.75, 3.25]#np.arange(6)
width = 0.4

#PML
pSVC_TP = 250
pSVC_FP = 5
pSVC_TN = 245
pSVC_FN = 0

pRFC_TP = 199
pRFC_FP = 18
pRFC_TN = 232
pRFC_FN = 51

pMLP_TP = 250
pMLP_FP = 13
pMLP_TN = 237
pMLP_FN = 0

#SIS
sSVC_TP = 250
sSVC_FP = 2
sSVC_TN = 248
sSVC_FN = 0

sRFC_TP = 216
sRFC_FP = 13
sRFC_TN = 237
sRFC_FN = 34

sMLP_TP = 248
sMLP_FP = 10
sMLP_TN = 240
sMLP_FN = 2

#TP/FN bar chart
pTP = (pMLP_TP, 0, pRFC_TP, 0, pSVC_TP, 0)
pFN = (pMLP_FN, 0, pRFC_FN, 0, pSVC_FN, 0)
sTP = (0, sMLP_TP, 0, sRFC_TP, 0, sSVC_TP)
sFN = (0, sMLP_FN, 0, sRFC_FN, 0, sMLP_FN)

plt.figure(figsize=(10,7))
pTPbar = plt.barh(ind, pTP, width, color='green')
pFNbar = plt.barh(ind, pFN, width, color='red', left=pTP)
sTPbar = plt.barh(ind, sTP, width, color='#3bcc44')
sFNbar = plt.barh(ind, sFN, width, color='#ff6666', left=sTP)
plt.xlim([0, 250])
plt.ylim([-0.75,5])
plt.grid('off')
plt.xlabel('Number of samples')
plt.yticks([0, 1.5, 3], ('MLP', 'RFC', 'SVC'))
plt.legend(['Point-mass True Positive', 'Point mass False Negative', 'SIS True Positive', 'SIS False Negative'], loc=1, prop={'size': 20}, ncol=2)
plt.savefig("C:\Users\Ivan\Desktop\Gravitational Lensing CUHK\GitHubStuff\TPFN_Chart.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)

#TN/FP bar chart
pTN = (pMLP_TN, 0, pRFC_TN, 0, pSVC_TN, 0)
pFP = (pMLP_FP, 0, pRFC_FP, 0, pSVC_FP, 0)
sTN = (0, sMLP_TN, 0, sRFC_TN, 0, sSVC_TN)
sFP = (0, sMLP_FP, 0, sRFC_FP, 0, sSVC_FP)

plt.figure(figsize=(10,7))
pTPbar = plt.barh(ind, pTN, width, color='green')
pFNbar = plt.barh(ind, pFP, width, color='red', left=pTN)
sTPbar = plt.barh(ind, sTN, width, color='#3bcc44')
sFNbar = plt.barh(ind, sFP, width, color='#ff6666', left=sTN)
plt.xlim([0, 250])
plt.ylim([-0.75, 5])
plt.grid('off')
plt.xlabel('Number of samples')
plt.yticks([0, 1.5, 3], ('MLP', 'RFC', 'SVC'))
plt.legend(['Point-mass True Negative', 'Point mass False Positive', 'SIS True Negative', 'SIS False Positive'], loc=1, prop={'size': 20}, ncol=2)
plt.savefig("C:\Users\Ivan\Desktop\Gravitational Lensing CUHK\GitHubStuff\TNFP_Chart.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)