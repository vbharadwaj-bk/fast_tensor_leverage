# [Cell 0]
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os

# [Cell 1]
filename = '../outputs/tt_benchmarks/cross_sin1.json'

exp = None
with open(filename, 'r') as f:
    exp = json.load(f)
    
fig = plt.figure(tight_layout=True, figsize=(9,3))
gs = gridspec.GridSpec(1,3)
axs = []

for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    ax.grid(True)
    axs.append(ax)

axs[0].set_xlabel("LSTSQ Problem Number")
axs[0].set_ylabel("Fit")
axs[0].plot(exp['lstsq_problem_numbers'], exp['fits']) 
    
axs[1].set_xlabel("Unique Evaluations")
axs[1].set_ylabel("Fit")
axs[1].plot(exp['unique_eval_counts'], exp['fits']) 

axs[2].set_xlabel("LSTSQ Problem Number")
axs[2].set_ylabel("Unique Evaluations")
axs[2].plot(exp['lstsq_problem_numbers'], exp['unique_eval_counts']) 