import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import json
import sys
import matplotlib.gridspec as gridspec

sys.path.append("..")
plt.rcParams['font.family'] = 'Liberation Serif'
plt.rcParams.update({'font.size': 11})

# ==========================================================================
# Runtime Benchmark
# ==========================================================================
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
# ==========================================================================
# Sparse Tensor Decomposition
# ==========================================================================
def get_data(files):
    data = []
    for filepath in files:
        with open(filepath) as infile:
            data_file = json.load(infile)
            data.extend(data_file)
        
    return data

uber_files = ['../paper_data_archive/uber_sparse_traces1.json']
uber_data = get_data(uber_files)

amazon_files = ['../paper_data_archive/amazon-reviews_sparse_traces1.json',
               '../paper_data_archive/amazon-reviews_sparse_traces2.json',
               '../paper_data_archive/amazon-reviews_sparse_traces3.json'
               ]
uber_data = get_data(amazon_files)
    
reddit_files = ['../paper_data_archive/reddit-2015_sparse_traces1.json',
               '../paper_data_archive/reddit-2015_sparse_traces2.json',
               '../paper_data_archive/reddit-2015_sparse_traces3.json']
reddit_data = get_data(reddit_files)

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
# ==========================================================================
fig = plt.figure()
fig.set_size_inches(15 * (1.1), 2.5 * (1.1))
spec = fig.add_gridspec(1, 5, wspace=0, hspace=0.0, left=0.0, right=0.8)
axs = [fig.add_subplot(spec[0, j]) for j in range(5)]

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
    axs[0].set_title("Uber Pickups (~3.3E6 nz)")

points = process_data(amazon_data)
for key in points:
    entry = points[key]
    axs[1].errorbar(entry[0], entry[1], fmt='o', c=colormap[key], xerr=entry[2])
axs[1].set_title("Amazon Reviews (~1.8E9 nz)")
    
points = process_data(reddit_data)
for key in points:
    entry = points[key]
    axs[2].errorbar(entry[0], entry[1], fmt='o', c=colormap[key], xerr=entry[2])
axs[2].set_title("Reddit-2015 (~4.7E9 nz)")
axs[2].set_xlabel("Fit")

axs[3].set_title("Enron")

fig.legend(bbox_to_anchor=(0.805, 0.04), ncol=5)
fig.savefig("paper_images/spten_accuracies.pdf", bbox_inches='tight')
plt.show()
# =============================================================

def generate_plot(ax, filename):
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
            ax.plot(x_axis, np.array(el)-1, '-o', c=color,label=label)

filename = '../paper_data_archive/amazon-reviews_exact_solve_comp_1_extended.json'
fig, ax = plt.subplots(figsize=(4,4))
generate_plot(ax, filename)

#ax.set_yscale('log')
ax.legend()
ax.grid(True)
#ax.set_title(title)
ax.set_ylim([-0.0005, 0.009])
ax.set_xlabel("LSTSQ Problem Number")
ax.set_ylabel(r"$\mathregular{\varepsilon}$", fontsize=15)
fig.savefig("paper_images/amazon_epsilon_progression.pdf",bbox_inches='tight')
plt.show()

# ===============================================================
# Time trace plots
uber_time_trace_files = ['../outputs/uber_time_comparison.json']
uber_time_traces = get_data(uber_time_trace_files)

reddit_time_trace_files = ['../paper_data_archive/reddit-2015_time_comparison1.json',
                           '../paper_data_archive/reddit-2015_time_comparison2.json']
reddit_time_traces = get_data(reddit_time_trace_files)

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
J_values = sorted(list(dict.fromkeys([el['J'] for el in dataset])))
colors={J_values[0]: 'brown', J_values[1]: 'violet', J_values[2]:'indigo', J_values[3]:'orange'}

fig, ax = plt.subplots(figsize=(8,4))
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
    ax.plot(x, mean_y, c=colors[J],label=f"L&K, J={J}")
    
ax.legend() 
# ===============================================================