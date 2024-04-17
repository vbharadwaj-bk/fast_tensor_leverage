import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing

n_trials = 5
N = 10
dim = [3, 4, 5]
# true_rank = rank = range(5,15,5)
true_rank = 10
rank = 2
std_noise = 1e-6
tests = ["tt-svd", "randomized_tt-svd", "tt-als", "randomized_tt-als"]
J = 2000
nsweeps = 50

path = "./outputs/dense_tt/synthetic/"
methods = {"tt-svd": {"time": [], "fit": []},
           "randomized_tt-svd": {"time": [], "fit": []},
           "tt-als": {"time": [], "fit": []},
           "randomized_tt-als": {"time": [], "fit": []},
           }
# tensor_sizes = [100, 200, 300, 400, 500]
# J = [125000, 250000, 375000, 500000]

for test in methods.keys():
    for d in dim:
        file_path = f'outputs/dense_tt/synthetic/svd_{J}-{rank}-{test}_{true_rank}_{d}_{n_trials}_{nsweeps}.json'
        with open(file_path, 'r') as file:
            file = json.load(file)
            methods[test]['fit'].append(file['fit'])
            methods[test]['time'].append(file['time'])

colors = ['red', 'blue', 'green', 'orange']

fig, ax = plt.subplots()
for (test, color) in zip(methods.keys(), colors):
    # ax.plot(dim, methods[test]['fit'], color=color, label=test)
    ax.plot(dim, methods[test]['time'], color=color, label=test)

ax.set_xlabel('dim')
# ax.set_ylabel('fit')
ax.set_ylabel('time')
ax.set_title(f'synthetic-svd Initialization: Synthetic Tensor')
ax.grid(True)
ax.set_xticks(dim)
ax.set_yscale('log')
# ax.set_ylim(0, 1)
#
ax.legend()

plt.show()
outfile = f'outputs/dense_tt/synthetic/plots/newsvd{0,1}-compare-time-no_noise_r{[10,2]}_dim{[3,4,5]}_order{10}sweep_{50,5}_2k.png'
plt.savefig(outfile)
