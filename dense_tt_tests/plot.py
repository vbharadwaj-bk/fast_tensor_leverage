import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing

n_trials = 5
N = 3
dim = [100, 200, 300, 400, 500]
# true_rank = rank = range(5,15,5)
true_rank = 20
rank = 5
std_noise = 1e-6
tests = ["tt-svd", "randomized_tt-svd", "tt-als", "randomized_tt-als"]
J = 5000
nsweeps = 15

path = "./outputs/dense_tt/synthetic/jsons"
methods = {"tt-svd": {"time": [], "fit": []},
           "randomized_tt-svd": {"time": [], "fit": []},
           "tt-als": {"time": [], "fit": []},
           "randomized_tt-als": {"time": [], "fit": []},
           # "tns-tt": {"time": [], "fit":[]}
           }
# tensor_sizes = [100, 200, 300, 400, 500]
# J = [125000, 250000, 375000, 500000]

for test in methods.keys():
    for d in dim:
        file_path = f'outputs/dense_tt/synthetic/jsons/oversample_{N}_{J}-{rank}-{test}_{true_rank}_{d}_{n_trials}_{nsweeps}.json'
        with open(file_path, 'r') as file:
            file = json.load(file)
            methods[test]['fit'].append(file['fit'])
            methods[test]['time'].append(file['time'])

colors = ['red', 'blue', 'darkgreen', 'orange']
plt.rcParams['lines.linewidth'] = 2
fig, ax = plt.subplots(figsize=(6, 5))

for (test, color) in zip(methods.keys(), colors):
    ax.plot(dim, methods[test]['fit'], color=color, label=test)
    # ax.plot(dim, methods[test]['time'], color=color, label=test)

ax.set_xlabel('Dimension Size (I)', fontsize=12)
# ax.set_ylabel('Time', fontsize=12)
ax.set_ylabel('Fit', fontsize=12)
ax.set_xticks(dim)
# ax.set_yticks(fit)
ax.legend( ["tt-svd", "rtt-svd", "tt-als", "rtt-als (proposal)"],fontsize=15)
plt.grid(True)

plt.show()
plt.savefig(f'outputs/dense_tt/synthetic/plots/fit_final_rec_15.pdf', format='pdf')
