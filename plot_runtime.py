import pandas as pd
import numpy as np
# import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import scipy.stats as stats
import math
import pickle
import copy
import os
from fileinput import filename
from collections import defaultdict, Counter

axes_dict = {}
figures = {}


#### create empty figures for plots
# mpl.rcParams.update({'font.size': 5, 'lines.linewidth': 4, 'lines.markersize': 40, 'font.family':'CMU Serif'})
mpl.rcParams.update({'font.size': 5, 'lines.linewidth': 7.5, 'font.family':'CMU Serif'})
plt.rcParams["axes.grid"] = False
plt.rcParams['axes.linewidth'] = 1.25
mpl.rcParams['axes.edgecolor'] = 'k'
mpl.rcParams["legend.handlelength"] = 6.0
mpl.rcParams['mathtext.fontset'] = 'stix'

figures['k2'] = plt.figure(figsize=(10,10))
axes_dict['k2'] = plt.axes()

figures['k10'] = plt.figure(figsize=(10,10))
axes_dict['k10'] = plt.axes()

figures['delta2'] = plt.figure(figsize=(10,10))
axes_dict['delta2'] = plt.axes()

figures['delta10'] = plt.figure(figsize=(10,10))
axes_dict['delta10'] = plt.axes()



#### plot variables
EXPERIMENT_NAMES = ['fairepsgreedy','lattice','DP']

colormap = {'prefix': 'limegreen', 'lattice': 'orange', 'fairepsgreedy': 'deepskyblue', 'DP': 'purple'} 
linemap = {'prefix': (0, (3, 1, 1, 1, 1, 1)),'lattice': 'solid', 'fairepsgreedy': 'dashed', 'DP': 'dashdot'} 
markermap = {'prefix': None,'lattice': None, 'fairepsgreedy': None, 'DP': None}
markersizemap = {'prefix': 25,'lattice': 25, 'fairepsgreedy': 25, 'DP': 25}
markerfillstyle = {'prefix': 'none','lattice': 'none', 'fairepsgreedy': 'none', 'DP': 'none'}
labelmap = {'prefix': 'Prefix random walk', 'lattice': 'Random walk', 'fairepsgreedy': 'Fair $\epsilon$-greedy', 'DP': 'DP'} 
capsizemap = {'prefix': 25,'lattice': 20, 'fairepsgreedy': 30, 'DP': 15} 

k = [100,1000,10000,20000]
delta = [0.05, 0.1, 0.15, 0.2]

for filename in os.listdir("./running_times_old/"):
    if 'with' in filename:
        continue
    tokens = filename.split("_")
    experiment = tokens[0]
    n_groups = tokens[4]
    delta = tokens[6]
    y_vals = []
    fp = open("./running_times_old/"+filename, 'r')
    Lines = fp.readlines()   
    print(filename)
    for line in Lines:
        y_vals.append(float(line[:-2]))

    dict_val = "k"+str(n_groups)
    axes_dict[dict_val].set_title("$\eta = 0.1,~~\ell=$"+str(n_groups), fontsize=40)
    axes_dict[dict_val].set_xlabel("$k$", fontsize=40)
    axes_dict[dict_val].set_ylabel("running time in sec.", fontsize=40)
    axes_dict[dict_val].tick_params(axis='both', which='major', labelsize=30)
    axes_dict[dict_val].plot(k, y_vals, \
        color=colormap[experiment], \
        label=labelmap[experiment], \
        marker=markermap[experiment], \
        linestyle=linemap[experiment],\
        markersize=markersizemap[experiment], \
        fillstyle=markerfillstyle[experiment])
axes_dict["k2"].legend(fontsize=20)
figures["k2"].savefig("running_time_k2.pdf")
axes_dict["k10"].legend(fontsize=20)
figures["k10"].savefig("running_time_k10.pdf")