#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 22:12:41 2021

@author: Anonymous
"""

#Imports
import numpy as np
import pandas as pd
from CMDPLinearProgramming import CMDPLPSolver
import matplotlib.pyplot as plt
import random
import time
import os
import math
import pickle
import sys
import argparse


start_time = time.time()

def reward_per_task(a, q):
    if a == q + 1:
        return 1
    return 0

def parse_args():
    parser = argparse.ArgumentParser("Online Coalitional Skill Vector Games - OptOCSF")
    parser.add_argument('-mul', "--multiplier", type=bool, default=True)
    parser.add_argument('-bm', "--budget_size", type=int, default=20)
    parser.add_argument('-m', "--num_skills", type=int, default=6, help="The number of skills")
    parser.add_argument('-nb', "--n_budget_sizes", type=int, default=6)
    parser.add_argument('-bd', "--baseline_budgets", nargs="+", default=[50, 100, 150, 250, 500, 1000], help="Baseline budgets")
    return parser.parse_args()


arglist = parse_args()
m = arglist.num_skills
budget_size = arglist.budget_size
n_budget_sizes = arglist.n_budget_sizes
baseline_budgets = arglist.baseline_budgets

# temp = sys.argv[1:]
# RUN_NUMBER = int(temp[0])

RUN_NUMBER = 11

random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

f = open('solution.pckl', 'rb')
[opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_gain, opt_con] = pickle.load(f)
f.close()

DistRequirements_mean = [0 for h in range(n_budget_sizes)]
DistRequirements_std = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean = [0 for h in range(n_budget_sizes)]
SampleComplexity_std = [0 for h in range(n_budget_sizes)]


RewardRegret_mean_fin = [0 for h in range(n_budget_sizes)]
RewardRegret_std_fin = [0 for h in range(n_budget_sizes)]
ConsViolationsRegret_mean_fin = [0 for h in range(n_budget_sizes)]
ConsViolationsRegret_std_fin = [0 for h in range(n_budget_sizes)]
ConsViolationsRegret_flattened = [0 for h in range(n_budget_sizes)]

for l in range(n_budget_sizes):  
    #Initialize:
    if n_budget_sizes == 1:
        # filename = 'OptOCSF.pckl'
        filename_short = 'OptOCSF-short.pckl'
        f = open('model.pckl', 'rb')
    elif arglist.multiplier:
        # filename = 'OptOCSF-' + str(baseline_budgets[l]) + '.pckl'
        filename_short = 'OptOCSF-short-' + str(baseline_budgets[l]) + '-mul-' + str(m) + '.pckl'
        f = open('model-mul' + str(baseline_budgets[l]) + '.pckl', 'rb')
        print("Initiating the test for a budget of size " + str(baseline_budgets[l]) + "q for task tau_q")
    else:
        # filename = 'OptOCSF-' + str(baseline_budgets[l]) + '.pckl'
        filename_short = 'OptOCSF-short-' + str(baseline_budgets[l]) + '-' + str(m) + '.pckl'
        f = open('model' + str(baseline_budgets[l]) + '.pckl', 'rb')
        print("Initiating the test for a budget of size " + str(baseline_budgets[l]) + " for each task")
    [NUMBER_SIMULATIONS, P, R, C, CONSTRAINT, C_b, alphas, m, k, budgets, states, actions, delta, goals] = pickle.load(f)
    f.close()
    
    NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)
    
    if l == 0:
        RewardRegret = [[[] for sim in range(NUMBER_SIMULATIONS)] for h in range(n_budget_sizes)]
        ConsViolationsRegret = [[ [ [ [ [] for q in range(k) ] for j in range(alphas[i])] for i in range(m)] for sim in range(NUMBER_SIMULATIONS)] for h in range(n_budget_sizes)]
        
    
#np.random.seed(RUN_NUMBER)

    # RewardRegret = [[] for sim in range(NUMBER_SIMULATIONS)]
    # ConsViolationsRegret = [ [ [ [ [] for q in range(k) ] for j in range(alphas[i])] for i in range(m)] for sim in range(NUMBER_SIMULATIONS)]
    
    coalition_structure =[[[] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    skill_coverage = [[ [ [ 0 for j in range(alphas[i])] for i in range(m) ] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    sample_complexity = [0 for sim in range(NUMBER_SIMULATIONS)]
    dists_per_task = [[0 for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    
    
    for sim in range(NUMBER_SIMULATIONS): 
        t_total = 0
        for q in range(k):
            t = 1
            actions = [0, q + 1] 
            R = {}
            for s in states:
                R[s] = {a: reward_per_task(a, q) for a in actions}           
            
            lp_solver = CMDPLPSolver(delta, P, R, C, alphas, m, k, states, actions, CONSTRAINT)
            state_count = {s: 0 for s in states}   
                
            s = lp_solver.step() 
            state_count[s] = 1
            
            # regs = []
            # cons = [ [ [ [] for q in range(k) ] for j in range(alphas[i])] for i in range(m)]
            while len(coalition_structure[sim][q]) < budgets[q]:
                # print("init epoch", t)
                T_e = t + 1
                lp_solver.setCounts(state_count)
                lp_solver.update_empirical_model(0)
                # if T_e - 1 > 0:
                lp_solver.compute_confidence_intervals(T_e)
                
                # if t == 0:
                #     pi_e = {s_1: {a: 0 for a in actions} for s_1 in states}
                #     for s_1 in states:
                #         for a in actions:
                #             pi_e[s_1][a] = 1 / len(actions)
                # else:
                
                # if feasible:
                #     pi_e, val_e, cost_e, status, varphi_e = lp_solver.compute_extended_LP() 
                #     if status != 'Optimal':
                #         feasible = False
                # else:
                #     pi_e = {s_1: {a: 0 for a in actions} for s_1 in states}
                #     for s_1 in states:
                #         for a in actions:
                #             pi_e[s_1][a] = 1 / len(actions)
                
                pi_e, val_e, cost_e, status, varphi_e = lp_solver.compute_extended_LP_task(q) 
                      
                state_count = {s: 0 for s in states}
        
                while lp_solver.NUMBER_OF_OCCURANCES[s] + state_count[s] < 2 * lp_solver.NUMBER_OF_OCCURANCES[s]:
                    # print(t)
                    # if t == 0:
                    #     state_count[s] = 1
                    # else:

                    if t != 1:
                        s = lp_solver.step()
                        state_count[s] += 1               
                          
                    prob = [pi_e[s][a] for a in actions]
                                        
                    a = int(np.random.choice(actions, 1, replace=True, p=prob))
                    
                    # if len(plausible_actions) == 1:
                    #     a = 0
                    # else:
                    #     a = int(np.random.choice(actions, 1, replace=True, p=prob))
                    #     while a not in plausible_actions:
                    #         a = int(np.random.choice(actions, 1, replace=True, p=prob))
                    
                    if a != 0:
                        coalition_structure[sim][a-1].append(s)
                        for i in range(m):
                            skill_coverage[sim][a-1][i][s[i]] += 1
                    
                    if t == 1:
                        RewardRegret[l][sim].append(opt_gain - R[s][a])
                        for i in range(m): 
                            for j in range(alphas[i]): 
                                # for q in range(k):
                                ConsViolationsRegret[l][sim][i][j][q].append(C[s][a][i][j][q])
                    else:
                        RewardRegret[l][sim].append(RewardRegret[l][sim][-1] + opt_gain - R[s][a])
                        for i in range(m): 
                            for j in range(alphas[i]): 
                                # for q in range(k):
                                ConsViolationsRegret[l][sim][i][j][q].append(ConsViolationsRegret[l][sim][i][j][q][-1] + C[s][a][i][j][q])

                    if len(coalition_structure[sim][q]) == budgets[q]:
                        break

                    # regs.append(RewardRegret[l][sim][-1])
                    # for i in range(m): 
                    #     for j in range(alphas[i]): 
                    #         for q in range(k):    
                    #             cons[i][j][q].append(ConsViolationsRegret[l][sim][i][j][q][-1])
                                    
                    # if t % 50 == 0:
                    #     f = open(filename, 'ab')
                    #     pickle.dump([NUMBER_SIMULATIONS, t, RewardRegret , ConsViolationsRegret, skill_coverage, coalition_structure, sample_complexity, pi_e, varphi_e], f)
                    #     f.close()
                        # regs = []
                        # cons = [ [ [ [] for q in range(k) ] for j in range(alphas[i])] for i in range(m)]
                    
                    t += 1
                    
                    # if len(plausible_actions) == 1:
                    #     break  
            
            t_total += t    
        
        print("Simulation nubmer " + str(sim) + " has terminated after " + str(t) + " time steps")
        sample_complexity[sim] = t_total
        
        for q in range(k):
            for i in range(m): 
                for j in range(alphas[i]):
                    skill_coverage[sim][q][i][j] /= len(coalition_structure[sim][q])
                    if np.abs(skill_coverage[sim][q][i][j] - goals[q][i][j]) > dists_per_task[sim][q]:
                        dists_per_task[sim][q] = np.abs(skill_coverage[sim][q][i][j] - goals[q][i][j])
        
        f = open(filename_short, 'ab')
        pickle.dump([NUMBER_SIMULATIONS, t, RewardRegret , ConsViolationsRegret, dists_per_task, coalition_structure, sample_complexity, pi_e, varphi_e], f)
        f.close()        
    
    
    dists = [min(dists_per_task[sim]) for sim in range(NUMBER_SIMULATIONS)]
    
    
    DistRequirements_mean[l] = np.mean(dists, axis = 0)
    DistRequirements_std[l] = np.std(dists, axis = 0)
    SampleComplexity_mean[l] = np.mean(sample_complexity, axis = 0)
    SampleComplexity_std[l] = np.std(sample_complexity, axis = 0)
    
    # RewardRegret_mean[l] = np.mean(RewardRegret, axis = 0)
    # RewardRegret_std[l] = np.std(RewardRegret, axis = 0)

    ConsViolationsRegret_flattened[l] = [np.max([ConsViolationsRegret[l][sim][i][j][q] for i in range(m) for j in range(alphas[i]) for q in range(k)], axis=0) for sim in range(NUMBER_SIMULATIONS)]
    # ConsViolationsRegret_mean[l] = np.mean(ConsViolationsRegret_flattened, axis = 0)
    # ConsViolationsRegret_std[l] = np.std(ConsViolationsRegret_flattened, axis = 0)
      
    regret_fin = [RewardRegret[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)]
    cons_fin = [ConsViolationsRegret_flattened[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)]
    
    RewardRegret_mean_fin[l] = np.mean(regret_fin)
    RewardRegret_std_fin[l] = np.std(regret_fin)
    ConsViolationsRegret_mean_fin[l] = np.mean(cons_fin)
    ConsViolationsRegret_std_fin[l] = np.std(cons_fin)
    
    #print(NUMBER_INFEASIBILITIES)
    # print(ConsViolationsRegret_mean)
    print("The mean maximum distance between a coalition's skill coverage and its corresponding task's goal is: " + str(DistRequirements_mean[l]) + "+-" + str(DistRequirements_std[l]))
    print("The mean sample complexity is: " + str(SampleComplexity_mean[l]) + "+-" + str(SampleComplexity_std[l]))
    print("The mean regret is: " + str(RewardRegret_mean_fin[l]) + "+-" + str(RewardRegret_std_fin[l]))
    print("The mean regret of constraint violations is: " + str(ConsViolationsRegret_mean_fin[l]) + "+-" + str(ConsViolationsRegret_std_fin[l]))
    print("-----------------------------------------------------------------------")

title = 'OptOCSF'
if arglist.multiplier:
    x = range(budget_size, budget_size + 20 * n_budget_sizes, 20)
else:
    x = range(budget_size, budget_size + 50 * n_budget_sizes, 50)

DistRequirements_mean = np.array(DistRequirements_mean)
DistRequirements_std = np.array(DistRequirements_std)
SampleComplexity_mean = np.array(SampleComplexity_mean)
SampleComplexity_std = np.array(SampleComplexity_std)

distPlot = plt.figure()
plt.plot(x, DistRequirements_mean)
plt.fill_between(x, DistRequirements_mean - DistRequirements_std, DistRequirements_mean + DistRequirements_std, alpha = 0.5)
plt.grid()
plt.xlabel('Baseline Budget')
plt.ylabel('Maximum Distance')
plt.title(title)
plt.show()
distPlot.savefig('OptOCSF - Distance.pdf', bbox_inches = 'tight')

samplePlot = plt.figure()
plt.plot(x, SampleComplexity_mean)
plt.fill_between(x, SampleComplexity_mean - SampleComplexity_std, SampleComplexity_mean + SampleComplexity_std, alpha = 0.5)
plt.grid()
plt.xlabel('Baseline Budget')
plt.ylabel('Sample Complexity')
plt.title(title)
plt.show()
samplePlot.savefig('OptOCSF - Sample Complexity.pdf', bbox_inches = 'tight')


# for l in range(n_budget_sizes):  
#     regret_fin = [RewardRegret[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)]
#     cons_fin = [ConsViolationsRegret_flattened[l][sim][-1] for sim in range(NUMBER_SIMULATIONS)]
    
#     RewardRegret_mean_fin[l] = np.mean(regret_fin)
#     RewardRegret_std_fin[l] = np.std(regret_fin)
#     ConsViolationsRegret_mean_fin[l] = np.mean(cons_fin)
#     ConsViolationsRegret_std_fin[l] = np.std(cons_fin)
    
    # if arglist.multiplier:
    #     title = 'OptOCSF-' + str(baseline_budgets[l])
    # else:
    #     title = 'OptOCSF-' + str(baseline_budgets[l])
    # regretPlot = plt.figure()
    # plt.plot(range(len(RewardRegret_mean[l])), RewardRegret_mean[l])
    # plt.fill_between(range(len(RewardRegret_mean[l])), RewardRegret_mean[l] - RewardRegret_std[l], RewardRegret_mean[l] + RewardRegret_std[l], alpha = 0.5)
    # plt.grid()
    # plt.xlabel('Time Step')
    # plt.ylabel('Reward Regret')
    # plt.title(title)
    # plt.show()
    # regretPlot.savefig(title + ' - Regret.pdf', bbox_inches = 'tight')
    
    
    # consPlot = plt.figure()
    # plt.plot(range(len(ConsViolationsRegret_mean[l])), ConsViolationsRegret_mean[l])
    # plt.fill_between(range(len(ConsViolationsRegret_mean[l])), ConsViolationsRegret_mean[l] - ConsViolationsRegret_std[l], ConsViolationsRegret_mean[l] + ConsViolationsRegret_std[l], alpha = 0.5)
    # plt.grid()
    # plt.xlabel('Time Step')
    # plt.ylabel('Regret of Constraint Violations')
    # plt.title(title)
    # plt.show()
    # consPlot.savefig(title + ' - Regret of Constraint Violations.pdf', bbox_inches = 'tight')

RewardRegret_mean_fin = np.array(RewardRegret_mean_fin)
RewardRegret_std_fin = np.array(RewardRegret_std_fin)
ConsViolationsRegret_mean_fin = np.array(ConsViolationsRegret_mean_fin)
ConsViolationsRegret_std_fin = np.array(ConsViolationsRegret_std_fin)

title = 'OptOCSF'    
regretPlot = plt.figure()
plt.plot(x, RewardRegret_mean_fin)
plt.fill_between(x, RewardRegret_mean_fin - RewardRegret_std_fin, RewardRegret_mean_fin + RewardRegret_std_fin, alpha = 0.5)
plt.grid()
plt.xlabel('Baseline Budget')
plt.ylabel('Reward Regret')
plt.title(title)
plt.show()
regretPlot.savefig('OptOCSF - Regret.pdf', bbox_inches = 'tight')

consPlot = plt.figure()
plt.plot(x, ConsViolationsRegret_mean_fin)
plt.fill_between(x, ConsViolationsRegret_mean_fin - ConsViolationsRegret_std_fin, ConsViolationsRegret_mean_fin + ConsViolationsRegret_std_fin, alpha = 0.5)
plt.grid()
plt.xlabel('Baseline Budget')
plt.ylabel('Regret of Constraint Violations')
plt.title(title)
plt.show()
consPlot.savefig('OptOCSF - Regret of Constraint Violations.pdf', bbox_inches = 'tight')