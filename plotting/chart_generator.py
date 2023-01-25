#!/usr/bin/env python
# coding: utf-8

# In[132]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import json
import sys
import matplotlib.gridspec as gridspec

sys.path.append("..")
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 11})


# In[133]:


data = None
with open("../paper_data_archive/runtime_bench.json") as f:
    data = json.load(f)

I_values = [el["I"] for el in data["I_trace"]]
I_con_times = [np.mean(el["construction_times"]) for el in data["I_trace"]]
I_sampling_times = [np.mean(el["sampling_times"]) for el in data["I_trace"]]
I_trace_R = data["I_trace"][0]["R"]
I_trace_N = data["I_trace"][0]["N"]

R_values = [el["R"] for el in data["R_trace"]]
R_con_times = [np.mean(el["construction_times"]) for el in data["R_trace"]]
R_sampling_times = [np.mean(el["sampling_times"]) for el in data["R_trace"]]
R_trace_I = data["R_trace"][0]["I"]
R_trace_N = data["R_trace"][0]["N"]

N_values = [el["N"] for el in data["N_trace"]]
N_con_times = [np.mean(el["construction_times"]) for el in data["N_trace"]]
N_sampling_times = [np.mean(el["sampling_times"]) for el in data["N_trace"]]
N_trace_I = data["N_trace"][0]["I"]
N_trace_R = data["N_trace"][0]["R"]

fig = plt.figure(tight_layout=True, figsize=(5.5,5))
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, :])
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True)
ax.plot(I_values, I_con_times, '-o',label="Construction")
ax.plot(I_values, I_sampling_times, '-o', label="Sampling")
props = dict(boxstyle='square', facecolor='white', alpha=1.0)
ax.text(0.05, 0.95, '$\mathregular{R=32, N=3}$', transform=ax.transAxes, fontsize=11,
        verticalalignment='top',bbox=props)

ax.set_ylabel('Time (s)')
ax.set_xlabel('I')

for i in range(2):
    ax = fig.add_subplot(gs[1, i])
    ax.grid(True)
    if i == 0:
        ax.set_xscale('log')
        ax.set_ylabel('Time (s)')
        ax.plot(R_values, R_con_times, '-o')
        ax.plot(R_values, R_sampling_times, '-o')
        ax.set_xlabel('R')
        ax.set_xticks(R_values)
        ax.minorticks_off()
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.text(0.05, 0.95, '$\mathregular{I=2^{22}}$, N=3', transform=ax.transAxes, fontsize=11,
        verticalalignment='top',bbox=props)
    if i == 1:
        ax.text(0.05, 0.95, '$\mathregular{I=2^{22}}$, R=32', transform=ax.transAxes, fontsize=11,
        verticalalignment='top',bbox=props)
        ax.plot(N_values, N_con_times, '-o')
        ax.plot(N_values, N_sampling_times, '-o')
        ax.set_xticks([2, 4, 6, 8])
        ax.set_xlabel('N')
        
fig.align_labels()
fig.legend(bbox_to_anchor=(0.8085, 0.04), ncol=5)
fig.savefig("paper_images/runtime_benchmark.pdf", bbox_inches='tight')
plt.show()


# In[13]:


def get_data(files):
    data = []
    for filepath in files:
        with open(filepath, 'r') as infile:
            data_file = json.load(infile)
            data.extend(data_file)
        
    return data

uber_files = ['../paper_data_archive/uber_sparse_traces1.json']
uber_data = get_data(uber_files)

amazon_files = ['../paper_data_archive/amazon-reviews_sparse_traces1.json',
               '../paper_data_archive/amazon-reviews_sparse_traces2.json',
               '../paper_data_archive/amazon-reviews_sparse_traces3.json'
               ]
amazon_data = get_data(amazon_files)
    
reddit_files = ['../paper_data_archive/reddit-2015_sparse_traces1.json',
               '../paper_data_archive/reddit-2015_sparse_traces2.json',
               '../paper_data_archive/reddit-2015_sparse_traces3.json']
reddit_data = get_data(reddit_files)

nell2_files = ['../paper_data_archive/nell-2_sparse_traces1.json',
              '../paper_data_archive/nell-2_sparse_traces2.json']
nell2_data = get_data(nell2_files)

enron_files = ['../paper_data_archive/enron_sparse_traces.json']
enron_data = get_data(enron_files)

sample_counts = [2 ** 16] 
R_values = [25, 50, 75, 100, 125]        

def process_data(data):     
    samplers = ["larsen_kolda", "larsen_kolda_hybrid", "efficient"]

    trial_dict = {}
    for trial in data:
        key = (trial['R'], trial['sampler'])
        if key not in trial_dict:
            trial_dict[key] = []

        trial_dict[key].append(trial)

    data_points = {"larsen_kolda": ([], [], []), "larsen_kolda_hybrid": ([], [], []), "efficient": ([], [], [])}
    
    for key in trial_dict.keys():
        trials = trial_dict[key]
        max_fits = []
        for trial in trials:
            max_fits.append(np.max(trial['trace']['fits']))

        mean_fit = np.mean(max_fits)
        std_fit = np.std(max_fits)

        y_position = key[0]

        data_points[key[1]][0].append(mean_fit)
        data_points[key[1]][1].append(y_position)
        data_points[key[1]][2].append(3 * std_fit)
        
    return data_points


# In[118]:


fig = plt.figure()
fig.set_size_inches(15 * (1.1), 2.5 * (1.1))
spec = fig.add_gridspec(1, 5, wspace=0, hspace=0.0, left=0.0, right=0.8)
axs = [fig.add_subplot(spec[0, j]) for j in range(5)]
axs[2].set_xlabel("Fit")

for i in range(1, 5):
    axs[i].sharey(axs[0])
    axs[i].label_outer()
    
for i in range(5):
    axs[i].grid(True)
    
axs[0].set_yticks(R_values)
axs[0].set_ylim([15,135])
axs[0].set_ylabel("Target Rank $\mathregular{R}$")

colormap = {"larsen_kolda": "blue", "larsen_kolda_hybrid": "orange", "efficient": "green"}

points = process_data(uber_data)
for key in points:
    entry = points[key]
    if key == "efficient":
        label="STS-CP (ours)"
    elif key == "larsen_kolda":
        label="CP-ARLS-LEV"
    elif key == "larsen_kolda_hybrid":
        label="CP-ARLS-LEV (hybrid)"
        
    axs[0].errorbar(entry[0], entry[1], fmt='o', c=colormap[key], xerr=entry[2], label=label)
    axs[0].set_title("Uber (~3.3e6 nz)")

def plot_data(data, ax):
    points = process_data(data)
    for key in points:
        entry = points[key]
        ax.errorbar(entry[0], entry[1], fmt='o', c=colormap[key], xerr=entry[2])
   
plot_data(enron_data, axs[1])
axs[1].set_title("Enron* (~5.4e7 nz)")

plot_data(nell2_data, axs[2])
axs[2].set_title("NELL-2* (~7.7e7 nz)")        

plot_data(amazon_data, axs[3]) 
axs[3].set_title("Amazon (~1.8e9 nz)")
    
plot_data(reddit_data, axs[4])
axs[4].set_title("Reddit* (~4.7e9 nz)")

fig.legend(bbox_to_anchor=(0.805, 0.04), ncol=5)
fig.savefig("paper_images/spten_accuracies.pdf", bbox_inches='tight')
plt.show()


# In[57]:


def generate_plot(ax, filename, label_fig=False):
    traces = {}

    samplers = ["larsen_kolda_hybrid", "efficient"]
    plot_map = {"larsen_kolda_hybrid": ("CP-ARLS-LEV hybrid", "orange"), "efficient": ("STS-CP (ours)", "green")}

    with open(filename, 'r') as f:
        data = json.load(f)

        for sampler in samplers:
            traces[sampler] =  [[measurement["ratio"] for measurement in trace] for trace in data[sampler]]

    x_axis = [el + 1 for el in list(range(len(traces[samplers[0]][0])))]
    
    for sampler in samplers:
        trace_array = np.array(traces[sampler])
        mean_trace = np.mean(trace_array, axis=0)
        traces[sampler] = [mean_trace]
    
    for sampler in samplers:
        first=True
        for el in traces[sampler]:
            label="_" + plot_map[sampler][0]
            if first:
                first = False
                label = label[1:]
            color = plot_map[sampler][1]
            if not label_fig:
                label = '_' + label
            ax.plot(x_axis, np.array(el)-1, '-o', c=color,label=label, markersize=4.0)

filename = '../paper_data_archive/amazon-reviews_exact_solve_comp_1_extended.json'

fig = plt.figure(tight_layout=True, figsize=(6,3))
gs = gridspec.GridSpec(1,2)
ax1 = fig.add_subplot(gs[0, 1])
generate_plot(ax1, filename)

#ax1.legend()
ax1.set_title("Amazon")
ax1.grid(True)
#ax.set_title(title)
ax1.set_ylim([-0.0005, 0.009])
ax1.set_xlabel("LSTSQ Problem Number")

filename = '../paper_data_archive/uber_exact_solve_comp_1.json'
ax2 = fig.add_subplot(gs[0, 0])
generate_plot(ax2, filename, label_fig=True)

ax2.grid(True)
ax2.set_ylim([-0.0005, 0.029])
ax2.set_xlabel("LSTSQ Problem Number")
ax2.set_ylabel(r"$\mathregular{\varepsilon}$", fontsize=15)
ax2.set_title("Uber")
fig.legend(bbox_to_anchor=(0.915, 0.04), ncol=5)
fig.savefig("paper_images/epsilon_progression.pdf",bbox_inches='tight')
plt.show()


# In[151]:


reddit_time_trace_files = ['../paper_data_archive/reddit-2015_time_comparison1.json',
                           '../paper_data_archive/reddit-2015_time_comparison2.json']
reddit_time_traces = get_data(reddit_time_trace_files)

reddit_efficient_files = ['../paper_data_archive/reddit-2015_time_comparison_efficient.json']
reddit_efficient_traces = get_data(reddit_efficient_files)

def get_time_update_pairs(result):
    trace = result["trace"]
    order = result["tensor_order"]
    prefix_sum = np.cumsum(trace["update_times"])
    iterations = trace["iterations"]
    fits = np.maximum(trace["fits"], 0.0)
    max_fits = []
    fit_computation_epoch = iterations[1] - iterations[0]
    times = [0.0]
    for i in range(order * fit_computation_epoch, len(prefix_sum)+1, order * fit_computation_epoch):
        times.append(prefix_sum[i-1])
    
    for fit in fits:
        if len(max_fits) == 0 or max_fits[-1] < fit:
            max_fits.append(fit)
        else:
            max_fits.append(max_fits[-1])
    
    return times, max_fits

def filter_data(results, J):
    return [el for el in results if el["J"] == J]

dataset = reddit_time_traces
efficient_dataset = reddit_efficient_traces
J_values = sorted(list(dict.fromkeys([el['J'] for el in dataset])))
J_values_efficient = sorted(list(dict.fromkeys([el['J'] for el in efficient_dataset])))
colors={J_values[0]: 'brown', J_values[1]: 'violet', J_values[2]:'indigo', J_values[3]:'orange', J_values[4]:'skyblue'}
colors_efficient={J_values[0]: 'green'}

sf=0.9
fig, ax = plt.subplots(figsize=(8*sf,4*sf))
ax.grid(True)
ax.set_xlabel("Cumulative ALS Update Time (s)")
ax.set_ylabel("Fit")
ax.set_ylim(0.08, 0.102)
#ax.set_ylim(0.20, 0.23)

for J in J_values:
    results = filter_data(dataset, J)
    max_time = max([get_time_update_pairs(result)[0][-1] for result in results])
    x = np.linspace(0.0, max_time, num=10000)
    
    interp_y = []
    for result in results:
        times, fits = get_time_update_pairs(result)
        ax.plot(times, fits, '--o', linewidth=0.4, markersize=1.9, c=colors[J])
        interp_y.append(np.interp(x, times, fits))
    mean_y = np.mean(interp_y, axis=0)
    ax.plot(x, mean_y, c=colors[J],label=f"CP-ARLS-LEV, J={J:,}")
    
    
for J in J_values_efficient:
    results = filter_data(efficient_dataset, J)
    max_time = max([get_time_update_pairs(result)[0][-1] for result in results])
    x = np.linspace(0.0, max_time, num=10000)
    
    interp_y = []
    for result in results:
        times, fits = get_time_update_pairs(result)
        ax.plot(times, fits, '--*', linewidth=0.4, markersize=1.9, c=colors_efficient[J])
        interp_y.append(np.interp(x, times, fits))
    mean_y = np.mean(interp_y, axis=0)
    ax.plot(x, mean_y, c=colors_efficient[J],label=f"STS-CP (ours), J={J:,}")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
fig.savefig("paper_images/reddit_time_plot.pdf",bbox_inches='tight')
    


# In[74]:


fig = plt.figure(tight_layout=True, figsize=(4.5,2.5))
gs = gridspec.GridSpec(1,2, wspace=0)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

with open("../outputs/accuracy_bench.json", "r") as infile:
    data = json.load(infile)
    N_values = [int(key) for key in data['N_trace'].keys()]
    N_eps_lk = [np.mean(data['N_trace'][key]["larsen_kolda"])-1 for key in data['N_trace'].keys()]
    N_eps_efficient = [np.mean(data['N_trace'][key]["efficient"])-1 for key in data['N_trace'].keys()]
    ax1.plot(N_values, N_eps_lk, '-o', c='blue')
    ax1.plot(N_values, N_eps_efficient, '-o', c='green')
    ax1.set_xticks([4,6,8])
    ax1.set_yscale('log')
    ax1.grid(True)
    
    R_values = [int(key) for key in data['R_trace'].keys()]
    R_eps_lk = [np.mean(data['R_trace'][key]["larsen_kolda"])-1 for key in data['R_trace'].keys()]
    R_eps_efficient = [np.mean(data['R_trace'][key]["efficient"])-1 for key in data['R_trace'].keys()]
    ax2.plot(R_values, R_eps_lk, '-o', c='blue', label="CP-ARLS-LEV")
    ax2.plot(R_values, R_eps_efficient, '-o', c='green', label="Our Sampler")
    ax2.set_yscale('log')
    ax2.grid(True)
    ax1.sharey(ax2)
    ax2.label_outer()
    ax2.set_xscale("log")
    ax2.set_xticks([16, 32, 64, 128])
    ax2.minorticks_off()
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    ax1.set_ylabel(r"$\mathregular{\varepsilon}$", fontsize=14)
    ax1.set_xlabel("N")
    ax2.set_xlabel("R")
    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    ax1.text(0.05, 0.95, '$\mathregular{R=64}$', transform=ax1.transAxes, fontsize=11,verticalalignment='top',bbox=props)
    ax2.text(0.05, 0.95, '$\mathregular{N=6}$', transform=ax2.transAxes, fontsize=11,verticalalignment='top',bbox=props)
    ax1.set_ylim(1e-3,1)
    
fig.show()
fig.legend(bbox_to_anchor=(0.965, 0.07), ncol=5)
fig.savefig("paper_images/accuracy_bench.pdf",bbox_inches='tight')


# In[131]:


with open("../outputs/distribution_comparison.json", "r") as infile:
    result = json.load(infile)
    N = result["N"]
    I = result["I"]
    J = result["J"]
    krp_height = I ** N
    
    true_distribution = result['true_distribution']
    sts_sampler_draws = np.array(result['sts_sampler_draws']).astype(np.int32)
    bins = np.array(np.bincount(sts_sampler_draws, minlength=krp_height)) / len(sts_sampler_draws)
    
    scale=5
    fig, ax = plt.subplots(figsize=(2 * scale,1 * scale))
    ax.plot(true_distribution / np.sum(true_distribution), label="True Leverage Score Distribution")
    ax.plot(bins, label=f"Histogram of Draws from Our Sampler")
    ax.grid(True)
    ax.set_xlabel("Row Index from KRP")
    ax.set_ylabel("Density")
    ax.legend()
    fig.show()
    fig.savefig("paper_images/distribution_comparison.pdf",bbox_inches='tight')


# In[154]:


# Generates a table with all of the data from the accuracy comparison
data = {"Uber": uber_data, 
        "Enron": enron_data, 
        "NELL-2": nell2_data, 
        "Amazon": amazon_data, 
        "Reddit": reddit_data}

samplers = ["larsen_kolda", "larsen_kolda_hybrid", "efficient"]
sampler_real_names = {"larsen_kolda": "CP-ARLS-LEV (rand)", 
                      "larsen_kolda_hybrid": "CP-ARLS-LEV (hybrid)",
                      "efficient": "STS-CP (ours)"}

for key in data:
    data[key] = process_data(data[key])
    
    for sampler in data[key]:
        means, ranks, stds = data[key][sampler]
        sampler_dict = {}
        for i in range(len(means)):
            sampler_dict[ranks[i]] = (means[i], stds[i])
            
        data[key][sampler] = sampler_dict
    
datasets = list(data.keys())

ranks = sorted(list(data[keys[0]]['larsen_kolda'].keys()))
num_ranks = len(ranks)

row_count = len(keys) * num_ranks + 1
col_count = len(samplers) + 2

table = [["" for _ in range(col_count)] for _ in range(row_count)]

# Header:
table[0][0], table[0][1] = "Tensor", "Rank"
for i in range(2, col_count):
    table[0][i] = sampler_real_names[samplers[i-2]]
    
for i in range(1, row_count):
    dataset = datasets[(i - 1) // num_ranks]
    rank_idx = (i - 1) % num_ranks
    if rank_idx == 0:
        table[i][0] = "\multirow{" + f"{num_ranks}" + "}{*}{" + dataset + "}"
        
    table[i][1] = f"{ranks[rank_idx]}"
    
    for j in range(len(samplers)):
        sampler = samplers[j]
        mean, std = data[dataset][sampler][ranks[rank_idx]]
        if sampler == "efficient":
            table[i][2 + j] = f"\\textbf{{{mean:#.3g}}} $\pm$ {std:.2e}"
        else:
            table[i][2 + j] = f"{mean:#.3g} $\pm$ {std:.2e}"
    
lines = ['\t& '.join(lst) for lst in table]
lines = [line + "\t\\\\" for line in lines]
lines.insert(0, "\\\\\n\\toprule")
lines.insert(2, "\\midrule")
for i in range(len(samplers) + 1):
    lines.insert((num_ranks) * (i+1) + 3 + i, '\\midrule')

lines.insert(len(lines), "\\bottomrule")
print("\n".join(lines))


# In[153]:


amazon_time_trace_files = ['../paper_data_archive/amazon-reviews_time_comparison1.json',
                           '../paper_data_archive/amazon-reviews_time_comparison2.json']
amazon_time_traces = get_data(amazon_time_trace_files)

amazon_efficient_files = ['../paper_data_archive/amazon-reviews_time_comparison_efficient.json']
amazon_efficient_traces = get_data(amazon_efficient_files)

def get_time_update_pairs(result):
    trace = result["trace"]
    order = result["tensor_order"]
    prefix_sum = np.cumsum(trace["update_times"])
    iterations = trace["iterations"]
    fits = np.maximum(trace["fits"], 0.0)
    max_fits = []
    fit_computation_epoch = iterations[1] - iterations[0]
    times = [0.0]
    for i in range(order * fit_computation_epoch, len(prefix_sum)+1, order * fit_computation_epoch):
        times.append(prefix_sum[i-1])
    
    for fit in fits:
        if len(max_fits) == 0 or max_fits[-1] < fit:
            max_fits.append(fit)
        else:
            max_fits.append(max_fits[-1])
    
    return times, max_fits

def filter_data(results, J):
    return [el for el in results if el["J"] == J]

dataset = amazon_time_traces
efficient_dataset = amazon_efficient_traces
J_values = sorted(list(dict.fromkeys([el['J'] for el in dataset])))
J_values_efficient = sorted(list(dict.fromkeys([el['J'] for el in efficient_dataset])))
colors={J_values[0]: 'brown', J_values[1]: 'violet', J_values[2]:'indigo', J_values[3]:'orange', J_values[4]:'skyblue'}
colors_efficient={J_values[0]: 'green'}

sf=1.0
fig, ax = plt.subplots(figsize=(8*sf,4*sf))
ax.grid(True)
ax.set_xlabel("Cumulative ALS Update Time (s)")
ax.set_ylabel("Fit")
ax.set_ylim(0.365, 0.394)

for J in J_values:
    results = filter_data(dataset, J)
    max_time = max([get_time_update_pairs(result)[0][-1] for result in results])
    x = np.linspace(0.0, max_time, num=10000)
    
    interp_y = []
    for result in results:
        times, fits = get_time_update_pairs(result)
        ax.plot(times, fits, '--o', linewidth=0.4, markersize=1.9, c=colors[J])
        interp_y.append(np.interp(x, times, fits))
    mean_y = np.mean(interp_y, axis=0)
    ax.plot(x, mean_y, c=colors[J],label=f"CP-ARLS-LEV, J={J:,}")
    
    
for J in J_values_efficient:
    results = filter_data(efficient_dataset, J)
    max_time = max([get_time_update_pairs(result)[0][-1] for result in results])
    x = np.linspace(0.0, max_time, num=10000)
    
    interp_y = []
    for result in results:
        times, fits = get_time_update_pairs(result)
        ax.plot(times, fits, '--*', linewidth=0.4, markersize=1.9, c=colors_efficient[J])
        interp_y.append(np.interp(x, times, fits))
    mean_y = np.mean(interp_y, axis=0)
    ax.plot(x, mean_y, c=colors_efficient[J],label=f"STS-CP (ours), J={J:,}")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
fig.savefig("paper_images/amazon_time_plot.pdf",bbox_inches='tight')


# In[161]:


nell2_time_trace_files = ['../paper_data_archive/nell-2_time_comparison.json']
nell2_time_traces = get_data(nell2_time_trace_files)

nell2_efficient_files = ['../paper_data_archive/nell-2_time_comparison_efficient.json']
nell2_efficient_traces = get_data(nell2_efficient_files)

def get_time_update_pairs(result):
    trace = result["trace"]
    order = result["tensor_order"]
    prefix_sum = np.cumsum(trace["update_times"])
    iterations = trace["iterations"]
    fits = np.maximum(trace["fits"], 0.0)
    max_fits = []
    fit_computation_epoch = iterations[1] - iterations[0]
    times = [0.0]
    for i in range(order * fit_computation_epoch, len(prefix_sum)+1, order * fit_computation_epoch):
        times.append(prefix_sum[i-1])
    
    for fit in fits:
        if len(max_fits) == 0 or max_fits[-1] < fit:
            max_fits.append(fit)
        else:
            max_fits.append(max_fits[-1])
    
    return times, max_fits

def filter_data(results, J):
    return [el for el in results if el["J"] == J]

dataset = nell2_time_traces
efficient_dataset = nell2_efficient_traces
J_values = sorted(list(dict.fromkeys([el['J'] for el in dataset])))
J_values_efficient = sorted(list(dict.fromkeys([el['J'] for el in efficient_dataset])))
colors={J_values[0]: 'brown', J_values[1]: 'violet', J_values[2]:'indigo', J_values[3]:'orange', J_values[4]:'skyblue'}
colors_efficient={J_values[0]: 'green'}

sf=1.0
fig, ax = plt.subplots(figsize=(8*sf,4*sf))
ax.grid(True)
ax.set_xlabel("Cumulative ALS Update Time (s)")
ax.set_ylabel("Fit")
ax.set_ylim(0.06, 0.08)

for J in J_values:
    results = filter_data(dataset, J)
    max_time = max([get_time_update_pairs(result)[0][-1] for result in results])
    x = np.linspace(0.0, max_time, num=10000)
    
    interp_y = []
    for result in results:
        times, fits = get_time_update_pairs(result)
        ax.plot(times, fits, '--o', linewidth=0.4, markersize=1.9, c=colors[J])
        interp_y.append(np.interp(x, times, fits))
    mean_y = np.mean(interp_y, axis=0)
    ax.plot(x, mean_y, c=colors[J],label=f"CP-ARLS-LEV, J={J:,}")
    
    
for J in J_values_efficient:
    results = filter_data(efficient_dataset, J)
    max_time = max([get_time_update_pairs(result)[0][-1] for result in results])
    x = np.linspace(0.0, max_time, num=10000)
    
    interp_y = []
    for result in results:
        times, fits = get_time_update_pairs(result)
        ax.plot(times, fits, '--*', linewidth=0.4, markersize=1.9, c=colors_efficient[J])
        interp_y.append(np.interp(x, times, fits))
    mean_y = np.mean(interp_y, axis=0)
    ax.plot(x, mean_y, c=colors_efficient[J],label=f"STS-CP (ours), J={J:,}")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
fig.savefig("paper_images/nell2_plot.pdf",bbox_inches='tight')


# In[ ]:




