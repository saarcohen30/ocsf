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
    return parser.parse_args()

def get_data_greedy(filename):
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
    
def get_data_opt(filename):
    f = open(filename, 'rb')
    j = 0
    while 1:
        try:
            j += 1
            [NUMBER_SIMULATIONS, t, RewardRegret , ConsViolationsRegret, dists_per_task, coalition_structure, sample_complexity, pi_e, varphi_e] = pickle.load(f)
            # if j == 4:
            #     break
        except EOFError:
            break
    f.close()
    
    return RewardRegret, ConsViolationsRegret, sample_complexity, dists_per_task, coalition_structure
    
def get_dists(dists_per_task, ):
    return [max(dists_per_task[sim]) for sim in range(NUMBER_SIMULATIONS)], [min(dists_per_task[sim]) for sim in range(NUMBER_SIMULATIONS)]

arglist = parse_args()
budget_size = arglist.budget_size
n_budget_sizes = arglist.n_budget_sizes


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

MaxDistRequirements_mean_opt = [0 for h in range(n_budget_sizes)]
MaxDistRequirements_std_opt = [0 for h in range(n_budget_sizes)]
MinDistRequirements_mean_opt = [0 for h in range(n_budget_sizes)]
MinDistRequirements_std_opt = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean_opt = [0 for h in range(n_budget_sizes)]
SampleComplexity_std_opt = [0 for h in range(n_budget_sizes)]

MaxDistRequirements_mean_tune = [0 for h in range(n_budget_sizes)]
MaxDistRequirements_std_tune = [0 for h in range(n_budget_sizes)]
MinDistRequirements_mean_tune = [0 for h in range(n_budget_sizes)]
MinDistRequirements_std_tune = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean_tune = [0 for h in range(n_budget_sizes)]
SampleComplexity_std_tune = [0 for h in range(n_budget_sizes)]

# MaxDistRequirements_mean_pess = [0 for h in range(n_budget_sizes)]
# MaxDistRequirements_std_pess = [0 for h in range(n_budget_sizes)]
# MinDistRequirements_mean_pess = [0 for h in range(n_budget_sizes)]
# MinDistRequirements_std_pess = [0 for h in range(n_budget_sizes)]
# SampleComplexity_mean_pess = [0 for h in range(n_budget_sizes)]
# SampleComplexity_std_pess = [0 for h in range(n_budget_sizes)]


RewardRegret_mean_fin_opt = [0 for h in range(n_budget_sizes)]
RewardRegret_std_fin_opt = [0 for h in range(n_budget_sizes)]
ConsViolationsRegret_mean_fin_opt = [0 for h in range(n_budget_sizes)]
ConsViolationsRegret_std_fin_opt = [0 for h in range(n_budget_sizes)]
ConsViolationsRegret_flattened_opt = [0 for h in range(n_budget_sizes)]

RewardRegret_mean_fin_tune = [0 for h in range(n_budget_sizes)]
RewardRegret_std_fin_tune = [0 for h in range(n_budget_sizes)]
ConsViolationsRegret_mean_fin_tune = [0 for h in range(n_budget_sizes)]
ConsViolationsRegret_std_fin_tune = [0 for h in range(n_budget_sizes)]
ConsViolationsRegret_flattened_tune = [0 for h in range(n_budget_sizes)]
        
# RewardRegret_mean_fin_pess = [0 for h in range(n_budget_sizes)]
# RewardRegret_std_fin_pess = [0 for h in range(n_budget_sizes)]
# ConsViolationsRegret_mean_fin_pess = [0 for h in range(n_budget_sizes)]
# ConsViolationsRegret_std_fin_pess = [0 for h in range(n_budget_sizes)]
# ConsViolationsRegret_flattened_pess = [0 for h in range(n_budget_sizes)]
        
for l in range(n_budget_sizes):
    if n_budget_sizes == 1:
        filename_greedy_01 = 'GREEDY-' + str(0.01) + '.pckl'
        filename_greedy_05 = 'GREEDY-' + str(0.05) + '.pckl'
        filename_opt = 'OptOCSF-short.pckl'
        filename_tune = 'TuneOptOCSF-short.pckl'
        # filename_pess = 'PessOptOCSF-short.pckl'
        f = open('model.pckl', 'rb')
    elif arglist.multiplier:
        filename_greedy_01 = 'GREEDY-' + str(0.01) + '-' + str(baseline_budgets[l]) + '.pckl'
        filename_greedy_05 = 'GREEDY-' + str(0.05) + '-' + str(baseline_budgets[l]) + '.pckl'
        filename_opt = 'OptOCSF-short-' + str(baseline_budgets[l]) + '.pckl'
        filename_tune = 'TuneOptOCSF-short-' + str(baseline_budgets[l]) + '.pckl'
        # filename_pess = 'PessOptOCSF-short-' + str(baseline_budgets[l]) + '.pckl'
        f = open('model-mul' + str(baseline_budgets[l]) + '.pckl', 'rb')
    else:
        filename_greedy_01 = 'GREEDY-' + str(0.01) + '-' + str(baseline_budgets[l]) + '.pckl'
        filename_greedy_05 = 'GREEDY-' + str(0.05) + '-' + str(baseline_budgets[l]) + '.pckl'
        filename_opt = 'OptOCSF-short-' + str(baseline_budgets[l]) + '.pckl'
        filename_tune = 'TuneOptOCSF-short-' + str(baseline_budgets[l]) + '.pckl'
        # filename_pess = 'PessOptOCSF-short-' + str(baseline_budgets[l]) + '.pckl'
        f = open('model' + str(baseline_budgets[l]) + '.pckl', 'rb')
    
    [NUMBER_SIMULATIONS, P, R, C, CONSTRAINT, C_b, alphas, m, k, budgets, states, actions, delta, goals] = pickle.load(f)
    f.close()
    
    NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)
    

    # dists_per_task = [[ [ [ 0 for j in range(alphas[i])] for i in range(m) ] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    sample_complexity_greedy_01, dists_per_task_greedy_01, coalition_structure_greedy_01 = get_data_greedy(filename_greedy_01)
    sample_complexity_greedy_05, dists_per_task_greedy_05, coalition_structure_greedy_05 = get_data_greedy(filename_greedy_05)
    RewardRegret_opt, ConsViolationsRegret_opt, sample_complexity_opt, dists_per_task_opt, coalition_structure_opt = get_data_opt(filename_opt)
    RewardRegret_tune, ConsViolationsRegret_tune, sample_complexity_tune, dists_per_task_tune, coalition_structure_tune = get_data_opt(filename_tune)
    # RewardRegret_pess, ConsViolationsRegret_pess, sample_complexity_pess, dists_per_task_pess, coalition_structure_pess = get_data_opt(filename_pess)

    max_dists_greedy_01, min_dists_greedy_01 = get_dists(dists_per_task_greedy_01)
    max_dists_greedy_05, min_dists_greedy_05 = get_dists(dists_per_task_greedy_05)
    max_dists_opt, min_dists_opt = get_dists(dists_per_task_opt)
    max_dists_tune, min_dists_tune = get_dists(dists_per_task_tune)
    # max_dists_pess, min_dists_pess  = get_dists(dists_per_task_pess, coalition_structure_pess)
    
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
    
    MaxDistRequirements_mean_opt[l] = np.mean(max_dists_opt, axis = 0)
    MaxDistRequirements_std_opt[l] = np.std(max_dists_opt, axis = 0)
    MinDistRequirements_mean_opt[l] = np.mean(min_dists_opt, axis = 0)
    MinDistRequirements_std_opt[l] = np.std(min_dists_opt, axis = 0)
    SampleComplexity_mean_opt[l] = np.mean(sample_complexity_opt, axis = 0)
    SampleComplexity_std_opt[l] = np.std(sample_complexity_opt, axis = 0)
    
    MaxDistRequirements_mean_tune[l] = np.mean(max_dists_tune, axis = 0)
    MaxDistRequirements_std_tune[l] = np.std(max_dists_tune, axis = 0)
    MinDistRequirements_mean_tune[l] = np.mean(min_dists_tune, axis = 0)
    MinDistRequirements_std_tune[l] = np.std(min_dists_tune, axis = 0)
    SampleComplexity_mean_tune[l] = np.mean(sample_complexity_tune, axis = 0)
    SampleComplexity_std_tune[l] = np.std(sample_complexity_tune, axis = 0)

    # MaxDistRequirements_mean_pess[l] = np.mean(max_dists_pess, axis = 0)
    # MaxDistRequirements_std_pess[l] = np.std(max_dists_pess, axis = 0)
    # MinDistRequirements_mean_pess[l] = np.mean(min_dists_pess, axis = 0)
    # MinDistRequirements_std_pess[l] = np.std(min_dists_pess, axis = 0)
    # SampleComplexity_mean_pess[l] = np.mean(sample_complexity_pess, axis = 0)
    # SampleComplexity_std_pess[l] = np.std(sample_complexity_pess, axis = 0)
    
    
    ConsViolationsRegret_flattened_opt[l] = [np.max([ConsViolationsRegret_opt[l][sim][i][j][q] for i in range(m) for j in range(alphas[i]) for q in range(k)], axis=0) for sim in range(NUMBER_SIMULATIONS)]
      
    regret_fin = [RewardRegret_opt[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)]
    cons_fin = [ConsViolationsRegret_flattened_opt[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)]
    
    RewardRegret_mean_fin_opt[l] = np.mean(regret_fin)
    RewardRegret_std_fin_opt[l] = np.std(regret_fin)
    ConsViolationsRegret_mean_fin_opt[l] = np.mean(cons_fin)
    ConsViolationsRegret_std_fin_opt[l] = np.std(cons_fin)
    
    ConsViolationsRegret_flattened_tune[l] = [np.max([ConsViolationsRegret_tune[l][sim][i][j][q] for i in range(m) for j in range(alphas[i]) for q in range(k)], axis=0) for sim in range(NUMBER_SIMULATIONS)]
      
    regret_fin = [RewardRegret_tune[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)] 
    cons_fin = [ConsViolationsRegret_flattened_tune[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)] #- CONSTRAINT[0][0][0] * len(ConsViolationsRegret_flattened_tune[l][sim])
    
    RewardRegret_mean_fin_tune[l] = np.mean(regret_fin)
    RewardRegret_std_fin_tune[l] = np.std(regret_fin)
    ConsViolationsRegret_mean_fin_tune[l] = np.mean(cons_fin)
    ConsViolationsRegret_std_fin_tune[l] = np.std(cons_fin)
    
    # ConsViolationsRegret_flattened_pess[l] = [np.max([ConsViolationsRegret_pess[l][sim][i][j][q] for i in range(m) for j in range(alphas[i]) for q in range(k)], axis=0) for sim in range(NUMBER_SIMULATIONS)]
      
    # regret_fin = [RewardRegret_pess[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)]
    # cons_fin = [ConsViolationsRegret_flattened_pess[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)]
    
    # RewardRegret_mean_fin_pess[l] = np.mean(regret_fin)
    # RewardRegret_std_fin_pess[l] = np.std(regret_fin)
    # ConsViolationsRegret_mean_fin_pess[l] = np.mean(cons_fin)
    # ConsViolationsRegret_std_fin_pess[l] = np.std(cons_fin)
    
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

MaxDistRequirements_mean_opt = np.array(MaxDistRequirements_mean_opt)
MaxDistRequirements_std_opt = np.array(MaxDistRequirements_std_opt)
MinDistRequirements_mean_opt = np.array(MinDistRequirements_mean_opt)
MinDistRequirements_std_opt = np.array(MinDistRequirements_std_opt)
SampleComplexity_mean_opt = np.array(SampleComplexity_mean_opt)
SampleComplexity_std_opt = np.array(SampleComplexity_std_opt)

MaxDistRequirements_mean_tune = np.array(MaxDistRequirements_mean_tune)
MaxDistRequirements_std_tune = np.array(MaxDistRequirements_std_tune)
MinDistRequirements_mean_tune = np.array(MinDistRequirements_mean_tune)
MinDistRequirements_std_tune = np.array(MinDistRequirements_std_tune)
SampleComplexity_mean_tune = np.array(SampleComplexity_mean_tune)
SampleComplexity_std_tune = np.array(SampleComplexity_std_tune)

# MaxDistRequirements_mean_pess = np.array(MaxDistRequirements_mean_pess)
# MaxDistRequirements_mean_pess = np.array(MaxDistRequirements_mean_pess)
# MinDistRequirements_mean_pess = np.array(MinDistRequirements_mean_pess)
# MinDistRequirements_mean_pess = np.array(MinDistRequirements_mean_pess)
# SampleComplexity_mean_pess = np.array(SampleComplexity_mean_pess)
# SampleComplexity_std_pess = np.array(SampleComplexity_std_pess)

RewardRegret_mean_fin_opt = np.array(RewardRegret_mean_fin_opt)
RewardRegret_std_fin_opt = np.array(RewardRegret_std_fin_opt)
ConsViolationsRegret_mean_fin_opt = np.array(ConsViolationsRegret_mean_fin_opt)
ConsViolationsRegret_std_fin_opt = np.array(ConsViolationsRegret_std_fin_opt)

RewardRegret_mean_fin_tune= np.array(RewardRegret_mean_fin_tune)
RewardRegret_std_fin_tune = np.array(RewardRegret_std_fin_tune)
ConsViolationsRegret_mean_fin_tune = np.array(ConsViolationsRegret_mean_fin_tune)
ConsViolationsRegret_std_fin_tune = np.array(ConsViolationsRegret_std_fin_tune)

# RewardRegret_mean_fin_pess= np.array(RewardRegret_mean_fin_pess)
# RewardRegret_std_fin_pess = np.array(RewardRegret_std_fin_pess)
# ConsViolationsRegret_mean_fin_pess = np.array(ConsViolationsRegret_mean_fin_pess)
# ConsViolationsRegret_std_fin_pess = np.array(ConsViolationsRegret_std_fin_pess)


if arglist.multiplier:
    x = range(budget_size, budget_size + 20 * n_budget_sizes, 20)
else:
    x = range(budget_size, budget_size + 50 * n_budget_sizes, 50)


final_dists_plot = plt.rcParams.update({'font.size': 14})
ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)


plt.plot(x, MaxDistRequirements_mean_greedy_01, label='0.01-GREEDY', color='saddlebrown', alpha=0.6, linewidth=2.5, marker="D",markersize='5', markeredgewidth='3')
plt.fill_between(x, MaxDistRequirements_mean_greedy_01 - MaxDistRequirements_std_greedy_01 ,MaxDistRequirements_mean_greedy_01 + MaxDistRequirements_std_greedy_01, alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')

plt.plot(x, MaxDistRequirements_mean_greedy_05, color='royalblue',alpha=0.6, label = '0.05-GREEDY',linewidth=2.5, marker="o", markersize='5', markeredgewidth='4')
plt.fill_between(x, MaxDistRequirements_mean_greedy_05 - MaxDistRequirements_std_greedy_05 ,MaxDistRequirements_mean_greedy_05 + MaxDistRequirements_std_greedy_05, alpha=0.2, linewidth=2.5, edgecolor='royalblue', facecolor='royalblue')#

plt.plot(x, MaxDistRequirements_mean_opt, label = 'OptOCSF', color='red',alpha=0.6, linewidth=2.5, marker="x",markersize='8', markeredgewidth='2')
plt.fill_between(x, MaxDistRequirements_mean_opt - MaxDistRequirements_std_opt ,MaxDistRequirements_mean_opt + MaxDistRequirements_std_opt, alpha=0.2, linewidth=2.5, edgecolor='red', facecolor='red')

plt.plot(x, MaxDistRequirements_mean_tune, color='forestgreen', alpha = 0.6, label = 'TuneOptOCSF', linewidth=2.5, marker="*", markersize='10', markeredgewidth='2')
plt.fill_between(x, MaxDistRequirements_mean_tune - MaxDistRequirements_std_tune , MaxDistRequirements_mean_tune + MaxDistRequirements_std_tune, alpha=0.2, linewidth=2.5, edgecolor='forestgreen', facecolor='forestgreen')

# plt.plot(x, MaxDistRequirements_mean_pess, color='black', alpha = 0.6, label = 'PessOptOCSF', linewidth=2.5, marker="+", markersize='10', markeredgewidth='2')
# plt.fill_between(x, MaxDistRequirements_mean_pess - MaxDistRequirements_std_pess , MaxDistRequirements_mean_pess + MaxDistRequirements_std_pess, alpha=0.2, linewidth=2.5, edgecolor='black', facecolor='black')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc = 'center right', bbox_to_anchor=(0.35, 0.35, 0.55, 0.55), prop={'size': 13})
plt.xlabel('Baseline Budget')
plt.ylabel('Maximum Distance')
plt.tight_layout()
plt.savefig("unknown_distribution_max_distances.pdf", bbox_inches = 'tight')
plt.show()

############################################################

final_dists_plot = plt.rcParams.update({'font.size': 12})
ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)


plt.plot(x, MinDistRequirements_mean_greedy_01, label='0.01-GREEDY', color='saddlebrown', alpha=0.6, linewidth=2.5, marker="D",markersize='5', markeredgewidth='3')
plt.fill_between(x, MinDistRequirements_mean_greedy_01 - MinDistRequirements_std_greedy_01 ,MinDistRequirements_mean_greedy_01 + MinDistRequirements_std_greedy_01, alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')

plt.plot(x, MinDistRequirements_mean_greedy_05, color='royalblue',alpha=0.6, label = '0.05-GREEDY',linewidth=2.5, marker="o", markersize='5', markeredgewidth='4')
plt.fill_between(x, MinDistRequirements_mean_greedy_05 - MinDistRequirements_std_greedy_05 , MinDistRequirements_mean_greedy_05 + MinDistRequirements_std_greedy_05, alpha=0.2, linewidth=2.5, edgecolor='royalblue', facecolor='royalblue')#

plt.plot(x, MinDistRequirements_mean_opt, label = 'OptOCSF', color='red',alpha=0.6, linewidth=2.5, marker="x",markersize='8', markeredgewidth='2')
plt.fill_between(x, MinDistRequirements_mean_opt - MinDistRequirements_std_opt , MinDistRequirements_mean_opt + MinDistRequirements_std_opt, alpha=0.2, linewidth=2.5, edgecolor='red', facecolor='red')

plt.plot(x, MinDistRequirements_mean_tune, color='forestgreen', alpha = 0.6, label = 'TuneOptOCSF', linewidth=2.5, marker="*", markersize='10', markeredgewidth='2')
plt.fill_between(x, MinDistRequirements_mean_tune - MinDistRequirements_std_tune , MinDistRequirements_mean_tune + MinDistRequirements_std_tune, alpha=0.2, linewidth=2.5, edgecolor='forestgreen', facecolor='forestgreen')

# plt.plot(x, MinDistRequirements_mean_pess, color='black', alpha = 0.6, label = 'PessOptOCSF', linewidth=2.5, marker="+", markersize='10', markeredgewidth='2')
# plt.fill_between(x, MinDistRequirements_mean_pess - MinDistRequirements_std_pess , MinDistRequirements_mean_pess + MinDistRequirements_std_pess, alpha=0.2, linewidth=2.5, edgecolor='black', facecolor='black')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2)
plt.xlabel('Baseline Budget')
plt.ylabel('Minimum Distance')
plt.tight_layout()
plt.savefig("unknown_distribution_min_distances.pdf", bbox_inches = 'tight')
plt.show()

############################################################

ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)

plt.plot(x, SampleComplexity_mean_greedy_01, label='0.01-GREEDY', color='saddlebrown', alpha=0.6, linewidth=2.5, marker="D",markersize='5', markeredgewidth='3')
plt.fill_between(x, SampleComplexity_mean_greedy_01 - SampleComplexity_std_greedy_01 ,SampleComplexity_mean_greedy_01 + SampleComplexity_std_greedy_01, alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')

plt.plot(x, SampleComplexity_mean_greedy_05, color='royalblue',alpha=0.6, label = '0.05-GREEDY',linewidth=2.5, marker="o", markersize='5', markeredgewidth='4')
plt.fill_between(x, SampleComplexity_mean_greedy_05 - SampleComplexity_std_greedy_05 ,SampleComplexity_mean_greedy_05 + SampleComplexity_std_greedy_05, alpha=0.2, linewidth=2.5, edgecolor='royalblue', facecolor='royalblue')#

plt.plot(x, SampleComplexity_mean_opt, label = 'OptOCSF', color='red',alpha=0.6, linewidth=2.5, marker="x",markersize='8', markeredgewidth='2')
plt.fill_between(x, SampleComplexity_mean_opt - SampleComplexity_std_opt ,SampleComplexity_mean_opt + SampleComplexity_std_opt, alpha=0.2, linewidth=2.5, edgecolor='red', facecolor='red')

plt.plot(x, SampleComplexity_mean_tune, color='forestgreen', alpha = 0.6, label = 'TuneOptOCSF', linewidth=2.5, marker="*", markersize='10', markeredgewidth='2')
plt.fill_between(x, SampleComplexity_mean_tune - SampleComplexity_std_tune ,SampleComplexity_mean_tune + SampleComplexity_std_tune, alpha=0.2, linewidth=2.5, edgecolor='forestgreen', facecolor='forestgreen')

# plt.plot(x, SampleComplexity_mean_pess, color='black', alpha = 0.6, label = 'PessOptOCSF', linewidth=2.5, marker="+", markersize='10', markeredgewidth='2')
# plt.fill_between(x, SampleComplexity_mean_pess - SampleComplexity_std_pess ,SampleComplexity_mean_pess + SampleComplexity_std_pess, alpha=0.2, linewidth=2.5, edgecolor='black', facecolor='black')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc = 'upper left',prop={'size': 13})
plt.xlabel('Baseline Budget')
plt.ylabel('Sample Complexity')
plt.tight_layout()
plt.savefig("unknown_distribution_sample_complexity.pdf", bbox_inches = 'tight')
plt.show()

############################################################

ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)


plt.plot(x, RewardRegret_mean_fin_opt, label = 'OptOCSF', color='red',alpha=0.6, linewidth=2.5, marker="x",markersize='8', markeredgewidth='2')
plt.fill_between(x, RewardRegret_mean_fin_opt - RewardRegret_std_fin_opt ,RewardRegret_mean_fin_opt + RewardRegret_std_fin_opt, alpha=0.2, linewidth=2.5, edgecolor='red', facecolor='red')

plt.plot(x, RewardRegret_mean_fin_tune, color='forestgreen', alpha = 0.6, label = 'TuneOptOCSF', linewidth=2.5, marker="*", markersize='10', markeredgewidth='2')
plt.fill_between(x, RewardRegret_mean_fin_tune - RewardRegret_std_fin_tune ,RewardRegret_mean_fin_tune + RewardRegret_std_fin_tune, alpha=0.2, linewidth=2.5, edgecolor='forestgreen', facecolor='forestgreen')

# plt.plot(x, RewardRegret_mean_fin_pess, color='black', alpha = 0.6, label = 'PessOptOCSF', linewidth=2.5, marker="+", markersize='10', markeredgewidth='2')
# plt.fill_between(x, RewardRegret_mean_fin_pess - RewardRegret_std_fin_pess ,RewardRegret_mean_fin_pess + RewardRegret_std_fin_pess, alpha=0.2, linewidth=2.5, edgecolor='black', facecolor='black')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc = 'upper left',prop={'size': 13})
plt.xlabel('Baseline Budget')
plt.ylabel('Reward Regret')
plt.tight_layout()
plt.savefig("unknown_distribution_regret.pdf", bbox_inches = 'tight')
plt.show()

############################################################

ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)


plt.plot(x, ConsViolationsRegret_mean_fin_opt, label = 'OptOCSF', color='red',alpha=0.6, linewidth=2.5, marker="x",markersize='8', markeredgewidth='2')
plt.fill_between(x, ConsViolationsRegret_mean_fin_opt - ConsViolationsRegret_std_fin_opt ,ConsViolationsRegret_mean_fin_opt + ConsViolationsRegret_std_fin_opt, alpha=0.2, linewidth=2.5, edgecolor='red', facecolor='red')

plt.plot(x, ConsViolationsRegret_mean_fin_tune, color='forestgreen', alpha = 0.6, label = 'TuneOptOCSF', linewidth=2.5, marker="*", markersize='10', markeredgewidth='2')
plt.fill_between(x, ConsViolationsRegret_mean_fin_tune - ConsViolationsRegret_std_fin_tune ,ConsViolationsRegret_mean_fin_tune + ConsViolationsRegret_std_fin_tune, alpha=0.2, linewidth=2.5, edgecolor='forestgreen', facecolor='forestgreen')

# plt.plot(x, ConsViolationsRegret_mean_fin_pess, color='black', alpha = 0.6, label = 'PessOptOCSF', linewidth=2.5, marker="+", markersize='10', markeredgewidth='2')
# plt.fill_between(x, ConsViolationsRegret_mean_fin_pess - ConsViolationsRegret_std_fin_pess ,ConsViolationsRegret_mean_fin_pess + ConsViolationsRegret_std_fin_pess, alpha=0.2, linewidth=2.5, edgecolor='black', facecolor='black')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc = 'upper left',prop={'size': 13})
plt.xlabel('Baseline Budget')
plt.ylabel('Regret of Constraint Violations')
plt.tight_layout()
plt.savefig("unknown_distribution_regret_cons.pdf", bbox_inches = 'tight')
plt.show()