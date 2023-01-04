import numpy as np
import pandas as pd
import sys
import pickle
import time
from matplotlib.ticker import StrMethodFormatter
import argparse
import matplotlib.pyplot as plt

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
# 

def parse_args():
    parser = argparse.ArgumentParser("Online Coalitional Skill Vector Games - Greedy")
    parser.add_argument('-mul', "--multiplier", type=bool, default=True)
    parser.add_argument('-bm', "--budget_size", type=int, default=20)
    parser.add_argument('-nb', "--n_budget_sizes", type=int, default=6)
    parser.add_argument('-bd', "--baseline_budgets", nargs="+", default=[50, 100, 150, 250, 500, 1000], help="Baseline budgets")
    return parser.parse_args()

def get_dists_per_task_and_sample_complexity(filename):
    f = open(filename, 'rb')
    j = 0
    while 1:
        try:
            j += 1
            [NUMBER_SIMULATIONS, sample_complexity, dists_per_task, coalition_structure] = pickle.load(f)
            # if j == 4:
            #     break
        except EOFError:
            break
    f.close()
    
    return sample_complexity, dists_per_task, coalition_structure
    
def get_dists(dists_per_task):
    return [max(dists_per_task[sim]) for sim in range(NUMBER_SIMULATIONS)], [min(dists_per_task[sim]) for sim in range(NUMBER_SIMULATIONS)]

arglist = parse_args()
budget_size = arglist.budget_size
n_budget_sizes = arglist.n_budget_sizes
baseline_budgets = arglist.baseline_budgets


# NUMBER_SIMULATIONS = 5
MaxDistRequirements_mean_greedy_01 = [0 for h in range(n_budget_sizes)]
MaxDistRequirements_std_greedy_01 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_mean_greedy_01 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_std_greedy_01 = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean_greedy_01 = [0 for h in range(n_budget_sizes)]
SampleComplexity_std_greedy_01 = [0 for h in range(n_budget_sizes)]

MaxDistRequirements_mean_greedy_05 = [0 for h in range(n_budget_sizes)]
MaxDistRequirements_std_greedy_05 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_mean_greedy_05 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_std_greedy_05 = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean_greedy_05 = [0 for h in range(n_budget_sizes)]
SampleComplexity_std_greedy_05 = [0 for h in range(n_budget_sizes)]

MaxDistRequirements_mean_greedy_1 = [0 for h in range(n_budget_sizes)]
MaxDistRequirements_std_greedy_1 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_mean_greedy_1 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_std_greedy_1 = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean_greedy_1 = [0 for h in range(n_budget_sizes)]
SampleComplexity_std_greedy_1 = [0 for h in range(n_budget_sizes)]

MaxDistRequirements_mean_greedy_5 = [0 for h in range(n_budget_sizes)]
MaxDistRequirements_std_greedy_5 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_mean_greedy_5 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_std_greedy_5 = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean_greedy_5 = [0 for h in range(n_budget_sizes)]
SampleComplexity_std_greedy_5 = [0 for h in range(n_budget_sizes)]

MaxDistRequirements_mean_cmdp = [0 for h in range(n_budget_sizes)]
MaxDistRequirements_std_cmdp = [0 for h in range(n_budget_sizes)]
MinDistRequirements_mean_cmdp = [0 for h in range(n_budget_sizes)]
MinDistRequirements_std_cmdp = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean_cmdp = [0 for h in range(n_budget_sizes)]
SampleComplexity_std_cmdp = [0 for h in range(n_budget_sizes)]

MaxDistRequirements_mean_cmdp_4 = [0 for h in range(n_budget_sizes)]
MaxDistRequirements_std_cmdp_4 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_mean_cmdp_4 = [0 for h in range(n_budget_sizes)]
MinDistRequirements_std_cmdp_4 = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean_cmdp_4 = [0 for h in range(n_budget_sizes)]
SampleComplexity_std_cmdp_4 = [0 for h in range(n_budget_sizes)]
        
for l in range(n_budget_sizes):
    if n_budget_sizes == 1:
        filename_greedy_01 = 'GREEDY-' + str(0.01) + '.pckl'
        filename_greedy_05 = 'GREEDY-' + str(0.05) + '.pckl'
        filename_greedy_1 = 'GREEDY-' + str(0.1) + '.pckl'
        filename_greedy_5 = 'GREEDY-' + str(0.5) + '.pckl'
        filename_cmdp = 'OCSF-CMDP.pckl'
        filename_cmdp_4 = 'OCSF-CMDP-4.pckl'
        f = open('model.pckl', 'rb')
    elif arglist.multiplier:
        filename_greedy_01 = 'GREEDY-' + str(0.01) + '-' + str(baseline_budgets[l]) + '-mul-6.pckl'
        filename_greedy_05 = 'GREEDY-' + str(0.05) + '-' + str(baseline_budgets[l]) + '-mul-6.pckl'
        filename_greedy_1 = 'GREEDY-' + str(0.1) + '-' + str(baseline_budgets[l]) + '-mul-6.pckl'
        filename_greedy_5 = 'GREEDY-' + str(0.5) + '-' + str(baseline_budgets[l]) + '-mul-6.pckl'
        filename_cmdp = 'OCSF-CMDP-' + str(baseline_budgets[l]) + '-mul-6.pckl'
        filename_cmdp_4 = 'OCSF-CMDP-' + str(baseline_budgets[l]) + '-mul-4.pckl'
        f = open('model-mul' + str(baseline_budgets[l]) + '.pckl', 'rb')
    else:
        filename_greedy_01 = 'GREEDY-' + str(0.01) + '-' + str(baseline_budgets[l]) + '-6.pckl'
        filename_greedy_05 = 'GREEDY-' + str(0.05) + '-' + str(baseline_budgets[l]) + '-6.pckl'
        filename_greedy_1 = 'GREEDY-' + str(0.1) + '-' + str(baseline_budgets[l]) + '-6.pckl'
        filename_greedy_5 = 'GREEDY-' + str(0.5) + '-' + str(baseline_budgets[l]) + '-6.pckl'
        filename_cmdp = 'OCSF-CMDP-' + str(baseline_budgets[l]) + '.pckl'
        f = open('model' + str(baseline_budgets[l]) + '.pckl', 'rb')
    
    [NUMBER_SIMULATIONS, P, R, C, CONSTRAINT, C_b, alphas, m, k, budgets, states, actions, delta, goals] = pickle.load(f)
    f.close()
    
    NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)
    
    alphas = [2, 2, 3, 8, 2, 2]
    m = 6

    # dists_per_task = [[ [ [ 0 for j in range(alphas[i])] for i in range(m) ] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    sample_complexity_greedy_01, dists_per_task_greedy_01, coalition_structure_greedy_01 = get_dists_per_task_and_sample_complexity(filename_greedy_01)
    sample_complexity_greedy_05, dists_per_task_greedy_05, coalition_structure_greedy_05 = get_dists_per_task_and_sample_complexity(filename_greedy_05)
    sample_complexity_greedy_1, dists_per_task_greedy_1, coalition_structure_greedy_1 = get_dists_per_task_and_sample_complexity(filename_greedy_1)
    sample_complexity_greedy_5, dists_per_task_greedy_5, coalition_structure_greedy_5 = get_dists_per_task_and_sample_complexity(filename_greedy_5)
    # sample_complexity_cmdp, dists_per_task_cmdp, coalition_structure_cmdp = get_dists_per_task_and_sample_complexity(filename_cmdp)
    # sample_complexity_cmdp_4, dists_per_task_cmdp_4, coalition_structure_cmdp_4 = get_dists_per_task_and_sample_complexity(filename_cmdp_4)

    max_dists_greedy_01, min_dists_greedy_01 = get_dists(dists_per_task_greedy_01)
    max_dists_greedy_05, min_dists_greedy_05 = get_dists(dists_per_task_greedy_05)
    max_dists_greedy_1, min_dists_greedy_1 = get_dists(dists_per_task_greedy_1)
    max_dists_greedy_5, min_dists_greedy_5 = get_dists(dists_per_task_greedy_5)
    # max_dists_cmdp, min_dists_cmdp  = get_dists(dists_per_task_cmdp, coalition_structure_cmdp)
    # max_dists_cmdp_4, min_dists_cmdp_4  = get_dists(dists_per_task_cmdp_4, coalition_structure_cmdp_4)
    
    MaxDistRequirements_mean_greedy_01[l] = np.mean(max_dists_greedy_01, axis = 0)
    MaxDistRequirements_std_greedy_01[l] = np.std(max_dists_greedy_01, axis = 0)
    MinDistRequirements_mean_greedy_01[l] = np.mean(min_dists_greedy_01, axis = 0)
    MinDistRequirements_std_greedy_01[l] = np.std(min_dists_greedy_01, axis = 0)
    SampleComplexity_mean_greedy_01[l] = np.mean(sample_complexity_greedy_01, axis = 0)
    SampleComplexity_std_greedy_01[l] = np.std(sample_complexity_greedy_01, axis = 0)
    
    MaxDistRequirements_mean_greedy_05[l] = np.mean(max_dists_greedy_05, axis = 0)
    MaxDistRequirements_std_greedy_05[l] = np.std(max_dists_greedy_05, axis = 0)
    MinDistRequirements_mean_greedy_05[l] = np.mean(min_dists_greedy_05, axis = 0)
    MinDistRequirements_std_greedy_05[l] = np.std(min_dists_greedy_05, axis = 0)
    SampleComplexity_mean_greedy_05[l] = np.mean(sample_complexity_greedy_05, axis = 0)
    SampleComplexity_std_greedy_05[l] = np.std(sample_complexity_greedy_05, axis = 0)
    
    MaxDistRequirements_mean_greedy_1[l] = np.mean(max_dists_greedy_1, axis = 0)
    MaxDistRequirements_std_greedy_1[l] = np.std(max_dists_greedy_1, axis = 0)
    MinDistRequirements_mean_greedy_1[l] = np.mean(min_dists_greedy_1, axis = 0)
    MinDistRequirements_std_greedy_1[l] = np.std(min_dists_greedy_1, axis = 0)
    SampleComplexity_mean_greedy_1[l] = np.mean(sample_complexity_greedy_1, axis = 0)
    SampleComplexity_std_greedy_1[l] = np.std(sample_complexity_greedy_1, axis = 0)
    
    MaxDistRequirements_mean_greedy_5[l] = np.mean(max_dists_greedy_5, axis = 0)
    MaxDistRequirements_std_greedy_5[l] = np.std(max_dists_greedy_5, axis = 0)
    MinDistRequirements_mean_greedy_5[l] = np.mean(min_dists_greedy_5, axis = 0)
    MinDistRequirements_std_greedy_5[l] = np.std(min_dists_greedy_5, axis = 0)
    SampleComplexity_mean_greedy_5[l] = np.mean(sample_complexity_greedy_5, axis = 0)
    SampleComplexity_std_greedy_5[l] = np.std(sample_complexity_greedy_5, axis = 0)

    # MaxDistRequirements_mean_cmdp[l] = np.mean(max_dists_cmdp, axis = 0)
    # MaxDistRequirements_std_cmdp[l] = np.std(max_dists_cmdp, axis = 0)
    # MinDistRequirements_mean_cmdp[l] = np.mean(min_dists_cmdp, axis = 0)
    # MinDistRequirements_std_cmdp[l] = np.std(min_dists_cmdp, axis = 0)
    # SampleComplexity_mean_cmdp[l] = np.mean(sample_complexity_cmdp, axis = 0)
    # SampleComplexity_std_cmdp[l] = np.std(sample_complexity_cmdp, axis = 0)
    
    # MaxDistRequirements_mean_cmdp_4[l] = np.mean(max_dists_cmdp_4, axis = 0)
    # MaxDistRequirements_std_cmdp_4[l] = np.std(max_dists_cmdp_4, axis = 0)
    # MinDistRequirements_mean_cmdp_4[l] = np.mean(min_dists_cmdp_4, axis = 0)
    # MinDistRequirements_std_cmdp_4[l] = np.std(min_dists_cmdp_4, axis = 0)
    # SampleComplexity_mean_cmdp_4[l] = np.mean(sample_complexity_cmdp_4, axis = 0)
    # SampleComplexity_std_cmdp_4[l] = np.std(sample_complexity_cmdp_4, axis = 0)
    
    # print("The mean maximum distance between a coalition's skill coverage and its corresponding task's goal is: " + str(DistRequirements_mean[l]) + "+-" + str(DistRequirements_std[l]))
    # print("The mean sample complexity is: " + str(SampleComplexity_mean[l]) + "+-" + str(SampleComplexity_std[l]))
    # print("-----------------------------------------------------------------------")


MaxDistRequirements_mean_greedy_01 = np.array(MaxDistRequirements_mean_greedy_01)
MaxDistRequirements_std_greedy_01 = np.array(MaxDistRequirements_std_greedy_01)
MinDistRequirements_mean_greedy_01 = np.array(MinDistRequirements_mean_greedy_01)
MinDistRequirements_std_greedy_01 = np.array(MinDistRequirements_std_greedy_01)
SampleComplexity_mean_greedy_01 = np.array(SampleComplexity_mean_greedy_01)
SampleComplexity_std_greedy_01 = np.array(SampleComplexity_std_greedy_01)

MaxDistRequirements_mean_greedy_05 = np.array(MaxDistRequirements_mean_greedy_05)
MaxDistRequirements_std_greedy_05 = np.array(MaxDistRequirements_std_greedy_05)
MinDistRequirements_mean_greedy_05 = np.array(MinDistRequirements_mean_greedy_05)
MinDistRequirements_std_greedy_05 = np.array(MinDistRequirements_std_greedy_05)
SampleComplexity_mean_greedy_05 = np.array(SampleComplexity_mean_greedy_05)
SampleComplexity_std_greedy_05 = np.array(SampleComplexity_std_greedy_05)

MaxDistRequirements_mean_greedy_1 = np.array(MaxDistRequirements_mean_greedy_1)
MaxDistRequirements_std_greedy_1 = np.array(MaxDistRequirements_std_greedy_1)
MinDistRequirements_mean_greedy_1 = np.array(MinDistRequirements_mean_greedy_1)
MinDistRequirements_std_greedy_1 = np.array(MinDistRequirements_std_greedy_1)
SampleComplexity_mean_greedy_1 = np.array(SampleComplexity_mean_greedy_1)
SampleComplexity_std_greedy_1 = np.array(SampleComplexity_std_greedy_1)

MaxDistRequirements_mean_greedy_5 = np.array(MaxDistRequirements_mean_greedy_5)
MaxDistRequirements_std_greedy_5 = np.array(MaxDistRequirements_std_greedy_5)
MinDistRequirements_mean_greedy_5 = np.array(MinDistRequirements_mean_greedy_5)
MinDistRequirements_std_greedy_5 = np.array(MinDistRequirements_std_greedy_5)
SampleComplexity_mean_greedy_5 = np.array(SampleComplexity_mean_greedy_5)
SampleComplexity_std_greedy_5 = np.array(SampleComplexity_std_greedy_5)

# MaxDistRequirements_mean_cmdp = np.array(MaxDistRequirements_mean_cmdp)
# MaxDistRequirements_mean_cmdp = np.array(MaxDistRequirements_mean_cmdp)
# MinDistRequirements_mean_cmdp = np.array(MinDistRequirements_mean_cmdp)
# MinDistRequirements_mean_cmdp = np.array(MinDistRequirements_mean_cmdp)
# SampleComplexity_mean_cmdp = np.array(SampleComplexity_mean_cmdp)
# SampleComplexity_std_cmdp = np.array(SampleComplexity_std_cmdp)

# MaxDistRequirements_mean_cmdp_4 = np.array(MaxDistRequirements_mean_cmdp_4)
# MaxDistRequirements_mean_cmdp_4 = np.array(MaxDistRequirements_mean_cmdp_4)
# MinDistRequirements_mean_cmdp_4 = np.array(MinDistRequirements_mean_cmdp_4)
# MinDistRequirements_mean_cmdp_4 = np.array(MinDistRequirements_mean_cmdp_4)
# SampleComplexity_mean_cmdp_4 = np.array(SampleComplexity_mean_cmdp_4)
# SampleComplexity_std_cmdp_4 = np.array(SampleComplexity_std_cmdp_4)

title = 'OCSF-CMDP'
x = baseline_budgets


final_dists_plot = plt.rcParams.update({'font.size': 12})
ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)


plt.plot(x, MaxDistRequirements_mean_greedy_01, label='0.01-GREEDY', color='saddlebrown', alpha=0.6, linewidth=2.5, marker="D",markersize='5', markeredgewidth='3')
plt.fill_between(x, MaxDistRequirements_mean_greedy_01 - MaxDistRequirements_std_greedy_01 ,MaxDistRequirements_mean_greedy_01 + MaxDistRequirements_std_greedy_01, alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')

plt.plot(x, MaxDistRequirements_mean_greedy_05, color='royalblue',alpha=0.6, label = '0.05-GREEDY',linewidth=2.5, marker="o", markersize='5', markeredgewidth='4')
plt.fill_between(x, MaxDistRequirements_mean_greedy_05 - MaxDistRequirements_std_greedy_05 ,MaxDistRequirements_mean_greedy_05 + MaxDistRequirements_std_greedy_05, alpha=0.2, linewidth=2.5, edgecolor='royalblue', facecolor='royalblue')#

plt.plot(x, MaxDistRequirements_mean_greedy_1, label = '0.1-GREEDY', color='red',alpha=0.6, linewidth=2.5, marker="x",markersize='8', markeredgewidth='2')
plt.fill_between(x, MaxDistRequirements_mean_greedy_1 - MaxDistRequirements_std_greedy_1 ,MaxDistRequirements_mean_greedy_1 + MaxDistRequirements_std_greedy_1, alpha=0.2, linewidth=2.5, edgecolor='red', facecolor='red')

plt.plot(x, MaxDistRequirements_mean_greedy_5, color='forestgreen', alpha = 0.6, label = '0.5-GREEDY', linewidth=2.5, marker="*", markersize='10', markeredgewidth='2')
plt.fill_between(x, MaxDistRequirements_mean_greedy_5 - MaxDistRequirements_std_greedy_5 , MaxDistRequirements_mean_greedy_5 + MaxDistRequirements_std_greedy_5, alpha=0.2, linewidth=2.5, edgecolor='forestgreen', facecolor='forestgreen')

# plt.plot(x, MaxDistRequirements_mean_cmdp, color='black', alpha = 0.6, label = 'OCSF-CMDP (m=6)', linewidth=2.5, marker="+", markersize='10', markeredgewidth='2')
# plt.fill_between(x, MaxDistRequirements_mean_cmdp - MaxDistRequirements_std_cmdp , MaxDistRequirements_mean_cmdp + MaxDistRequirements_std_cmdp, alpha=0.2, linewidth=2.5, edgecolor='black', facecolor='black')

# plt.plot(x, MaxDistRequirements_mean_cmdp_4, color='darkviolet', alpha = 0.6, label = 'OCSF-CMDP (m=4)', linewidth=2.5, marker="s", markersize='10', markeredgewidth='2')
# plt.fill_between(x, MaxDistRequirements_mean_cmdp_4 - MaxDistRequirements_std_cmdp_4 , MaxDistRequirements_mean_cmdp_4 + MaxDistRequirements_std_cmdp_4, alpha=0.2, linewidth=2.5, edgecolor='darkviolet', facecolor='darkviolet')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3)
plt.xlabel('Baseline Budget')
plt.ylabel('Maximum Distance')
plt.tight_layout()
plt.savefig("known_distribution_max_distances.pdf", bbox_inches = 'tight')
plt.show()


final_dists_plot = plt.rcParams.update({'font.size': 12})
ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)


plt.plot(x, MinDistRequirements_mean_greedy_01, label='0.01-GREEDY', color='saddlebrown', alpha=0.6, linewidth=2.5, marker="D",markersize='5', markeredgewidth='3')
plt.fill_between(x, MinDistRequirements_mean_greedy_01 - MinDistRequirements_std_greedy_01 ,MinDistRequirements_mean_greedy_01 + MinDistRequirements_std_greedy_01, alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')

plt.plot(x, MinDistRequirements_mean_greedy_05, color='royalblue',alpha=0.6, label = '0.05-GREEDY',linewidth=2.5, marker="o", markersize='5', markeredgewidth='4')
plt.fill_between(x, MinDistRequirements_mean_greedy_05 - MinDistRequirements_std_greedy_05 , MinDistRequirements_mean_greedy_05 + MinDistRequirements_std_greedy_05, alpha=0.2, linewidth=2.5, edgecolor='royalblue', facecolor='royalblue')#

plt.plot(x, MinDistRequirements_mean_greedy_1, label = '0.1-GREEDY', color='red',alpha=0.6, linewidth=2.5, marker="x",markersize='8', markeredgewidth='2')
plt.fill_between(x, MinDistRequirements_mean_greedy_1 - MinDistRequirements_std_greedy_1 , MinDistRequirements_mean_greedy_1 + MinDistRequirements_std_greedy_1, alpha=0.2, linewidth=2.5, edgecolor='red', facecolor='red')

plt.plot(x, MinDistRequirements_mean_greedy_5, color='forestgreen', alpha = 0.6, label = '0.5-GREEDY', linewidth=2.5, marker="*", markersize='10', markeredgewidth='2')
plt.fill_between(x, MinDistRequirements_mean_greedy_5 - MinDistRequirements_std_greedy_5 , MinDistRequirements_mean_greedy_5 + MinDistRequirements_std_greedy_5, alpha=0.2, linewidth=2.5, edgecolor='forestgreen', facecolor='forestgreen')

# plt.plot(x, MinDistRequirements_mean_cmdp, color='black', alpha = 0.6, label = 'OCSF-CMDP (m=6)', linewidth=2.5, marker="+", markersize='10', markeredgewidth='2')
# plt.fill_between(x, MinDistRequirements_mean_cmdp - MinDistRequirements_std_cmdp , MinDistRequirements_mean_cmdp + MinDistRequirements_std_cmdp, alpha=0.2, linewidth=2.5, edgecolor='black', facecolor='black')

# plt.plot(x, MinDistRequirements_mean_cmdp_4, color='darkviolet', alpha = 0.6, label = 'OCSF-CMDP (m=4)', linewidth=2.5, marker="s", markersize='10', markeredgewidth='2')
# plt.fill_between(x, MinDistRequirements_mean_cmdp_4 - MinDistRequirements_std_cmdp_4 , MinDistRequirements_mean_cmdp_4 + MinDistRequirements_std_cmdp_4, alpha=0.2, linewidth=2.5, edgecolor='darkviolet', facecolor='darkviolet')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05),
          ncol=2)
plt.xlabel('Baseline Budget')
plt.ylabel('Minimum Distance')
plt.tight_layout()
plt.savefig("known_distribution_min_distances.pdf", bbox_inches = 'tight')
plt.show()

ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)

plt.plot(x, SampleComplexity_mean_greedy_01, label='0.01-GREEDY', color='saddlebrown', alpha=0.6, linewidth=2.5, marker="D",markersize='5', markeredgewidth='3')
plt.fill_between(x, SampleComplexity_mean_greedy_01 - SampleComplexity_std_greedy_01 ,SampleComplexity_mean_greedy_01 + SampleComplexity_std_greedy_01, alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')

plt.plot(x, SampleComplexity_mean_greedy_05, color='royalblue',alpha=0.6, label = '0.05-GREEDY',linewidth=2.5, marker="o", markersize='5', markeredgewidth='4')
plt.fill_between(x, SampleComplexity_mean_greedy_05 - SampleComplexity_std_greedy_05 ,SampleComplexity_mean_greedy_05 + SampleComplexity_std_greedy_05, alpha=0.2, linewidth=2.5, edgecolor='royalblue', facecolor='royalblue')#

plt.plot(x, SampleComplexity_mean_greedy_1, label = '0.1-GREEDY', color='red',alpha=0.6, linewidth=2.5, marker="x",markersize='8', markeredgewidth='2')
plt.fill_between(x, SampleComplexity_mean_greedy_1 - SampleComplexity_std_greedy_1 ,SampleComplexity_mean_greedy_1 + SampleComplexity_std_greedy_1, alpha=0.2, linewidth=2.5, edgecolor='red', facecolor='red')

plt.plot(x, SampleComplexity_mean_greedy_5, color='forestgreen', alpha = 0.6, label = '0.5-GREEDY', linewidth=2.5, marker="*", markersize='10', markeredgewidth='2')
plt.fill_between(x, SampleComplexity_mean_greedy_5 - SampleComplexity_std_greedy_5 ,SampleComplexity_mean_greedy_5 + SampleComplexity_std_greedy_5, alpha=0.2, linewidth=2.5, edgecolor='forestgreen', facecolor='forestgreen')

# plt.plot(x, SampleComplexity_mean_cmdp, color='black', alpha = 0.6, label = 'OCSF-CMDP (m=6)', linewidth=2.5, marker="+", markersize='10', markeredgewidth='2')
# plt.fill_between(x, SampleComplexity_mean_cmdp - SampleComplexity_std_cmdp ,SampleComplexity_mean_cmdp + SampleComplexity_std_cmdp, alpha=0.2, linewidth=2.5, edgecolor='black', facecolor='black')

# plt.plot(x, SampleComplexity_mean_cmdp_4, color='darkviolet', alpha = 0.6, label = 'OCSF-CMDP (m=4)', linewidth=2.5, marker="+", markersize='10', markeredgewidth='2')
# plt.fill_between(x, SampleComplexity_mean_cmdp_4 - SampleComplexity_std_cmdp_4 ,SampleComplexity_mean_cmdp_4 + SampleComplexity_std_cmdp_4, alpha=0.2, linewidth=2.5, edgecolor='darkviolet', facecolor='darkviolet')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc = 'upper left',prop={'size': 13})
plt.xlabel('Baseline Budget')
plt.ylabel('Sample Complexity')
plt.tight_layout()
plt.savefig("known_distribution_sample_complexity.pdf", bbox_inches = 'tight')
plt.show()

