import numpy as np
import json
import os
import matplotlib.pyplot as plt

r_true = 5
std_noise = 10
path = "./outputs/dense_tt/synthetic/"
methods = {"tt-svd": {"time": [], "fit": []},
           "randomized_tt-svd": {"time": [], "fit": []},
           "tt-als": {"time": [], "fit": []},
           "randomized_tt-als": {"time": [], "fit": []},
           }
for i in range(1, 3):
    for method in methods.keys():
        file_path = f'{path}{std_noise}_{method}_data{i}_rank{r_true}.json'
        with open(file_path, 'r') as file:
            file = json.load(file)
            methods[method]['fit'].append(file['fit'])
            methods[method]['time'].append(file['time'])

# tensor_sizes = [100, 200, 300, 400, 500]
# colors = ['red', 'blue', 'green', 'orange']
tensor_sizes = [100, 200]
colors = ['red', 'blue', 'green', 'orange']

fig, ax = plt.subplots()
for (method, color) in zip(methods.keys(), colors):
    print(methods[method]['fit'])

    ax.plot(tensor_sizes, methods[method]['fit'], color=color, label=method)

ax.set_xlabel('Tensor Size')
ax.set_ylabel('fit')
ax.set_title('synthetic')
ax.grid(True)
ax.set_xticks(tensor_sizes)

# ax.set_ylim(0, 1.1)
ax.legend()

plt.show()
outfile = f'outputs/dense_tt/synthetic/plots/{std_noise}_fit_rank{r_true}.png'
plt.savefig(outfile)



# plots_folder = os.path.join(path, 'plots')
# if not os.path.exists(plots_folder):
#     os.makedirs(plots_folder)

# create_plot(tensor_sizes, fit, 'Tensor Size', 'fit',
#             'Synthetic Data Experiment - fit', colors, methods.keys(),
#             os.path.join(plots_folder, f"fit_rank{r_true}.png"), tensor_sizes)
#
# # Time Plot (assuming methods[method]['time'] exists)
# time_data = [methods[method]['time'] for method in methods]
# create_plot(tensor_sizes, time_data, 'Tensor Size', 'Time(s)',
#             'Synthetic Data Experiment - Time', colors, methods.keys(),
#             os.path.join(plots_folder, f"time_rank{r_true}.png"), tensor_sizes)
