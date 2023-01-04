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
    parser = argparse.ArgumentParser("Online Coalitional Skill Vector Games - OCSF-CMDP")
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

# f = open('solution.pckl', 'rb')
# [opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_gain, opt_con] = pickle.load(f)
# f.close()


DistRequirements_mean = [0 for h in range(n_budget_sizes)]
DistRequirements_std = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean = [0 for h in range(n_budget_sizes)]
SampleComplexity_std = [0 for h in range(n_budget_sizes)]


for l in range(n_budget_sizes):  
    #Initialize:
    if n_budget_sizes == 1:
        filename = 'OCSF-CMDP.pckl'
        f = open('model.pckl', 'rb')
    elif arglist.multiplier:
        filename = 'OCSF-CMDP-' + str(baseline_budgets[l]) + '-mul-' + str(m) + '.pckl'
        f = open('model-mul' + str(baseline_budgets[l]) + '.pckl', 'rb')
        print("Initiating the test for a budget of size " + str(baseline_budgets[l]) + "q for task tau_q")
    else:
        filename = 'OCSF-CMDP-' + str(baseline_budgets[l]) + '-' + str(m) + '.pckl'
        f = open('model' + str(baseline_budgets[l]) + '.pckl', 'rb')
        print("Initiating the test for a budget of size " + str(baseline_budgets[l]) + " for each task")
    [NUMBER_SIMULATIONS, P, R, C, CONSTRAINT, C_b, alphas, m, k, budgets, states, actions, delta, goals] = pickle.load(f)
    f.close()
    
    NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)
    
    dists_per_task = [[0 for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    coalition_structure =[[[] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    skill_coverage = [[ [ [ 0 for j in range(alphas[i])] for i in range(m) ] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    sample_complexity = [0 for sim in range(NUMBER_SIMULATIONS)]
    
    opt_policy_con = {}
    for q in range(k):
        actions = [0, q + 1] 
        R = {}
        for s in states:
            R[s] = {a: reward_per_task(a, q) for a in actions}           
        lp_solver = CMDPLPSolver(delta, P, R, C, alphas, m, k, states, actions, CONSTRAINT)
        opt_policy_con[q], opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_gain, opt_con = lp_solver.compute_opt_LP_Constrained_task(0, q)
    
    
    # if l == 0:
    
    for sim in range(NUMBER_SIMULATIONS):  
        t = 0
        while len([q for q in range(1, k + 1) if len(coalition_structure[sim][q-1]) < budgets[q-1]]) > 0:
            s = lp_solver.step()          
            q_max = 0
            pi_max = 0.0
            plausible_actions = [q for q in range(k) if len(coalition_structure[sim][q]) < budgets[q]]

            if len(plausible_actions) == 1:
                q_max = plausible_actions[0] + 1
            else:
                for q in plausible_actions:
                    if opt_policy_con[q][s][q+1] > pi_max and len(coalition_structure[sim][q]) < budgets[q]:
                        pi_max = opt_policy_con[q][s][q+1]
                        q_max = q + 1
            
            if q_max == 0:
                continue
            
            actions = [0, q_max] 
            prob = [opt_policy_con[q_max-1][s][a] for a in actions]
            # print(prob)
            # plausible_actions = [0] + [a for a in range(1, k + 1) if len(coalition_structure[sim][a-1]) < budgets[a-1] and prob[a] > 0.0]

            # if len(plausible_actions) == 1:
            #     a = 0
            # else:
            a = int(np.random.choice(actions, 1, replace=True, p=prob))
            # while a not in plausible_actions:
            #     a = int(np.random.choice(actions, 1, replace=True, p=prob))
            
            if a != 0:                   
                coalition_structure[sim][a-1].append(s)
            
                for i in range(m):
                    skill_coverage[sim][a-1][i][s[i]] += 1
                            
            # if t % 100 == 0:                
            #     f = open(filename, 'ab')
            #     pickle.dump([NUMBER_SIMULATIONS, sample_complexity, skill_coverage, coalition_structure], f)
            #     f.close()
            
            t += 1
        
        print("Simulation nubmer " + str(sim) + " has terminated after " + str(t) + " time steps")
        sample_complexity[sim] = t
        
        for q in range(k):
            for i in range(m): 
                for j in range(alphas[i]):
                    skill_coverage[sim][q][i][j] /= len(coalition_structure[sim][q])
                    if np.abs(skill_coverage[sim][q][i][j] - goals[q][i][j]) > dists_per_task[sim][q]:
                        dists_per_task[sim][q] = np.abs(skill_coverage[sim][q][i][j] - goals[q][i][j])        
        
        f = open(filename, 'ab')
        pickle.dump([NUMBER_SIMULATIONS, sample_complexity, dists_per_task, coalition_structure], f)
        f.close()
        
    
    dists = [min(dists_per_task[sim]) for sim in range(NUMBER_SIMULATIONS)]
    
    DistRequirements_mean[l] = np.mean(dists, axis = 0)
    DistRequirements_std[l] = np.std(dists, axis = 0)
    SampleComplexity_mean[l] = np.mean(sample_complexity, axis = 0)
    SampleComplexity_std[l] = np.std(sample_complexity, axis = 0)
    
    print("The mean maximum distance between a coalition's skill coverage and its corresponding task's goal is: " + str(DistRequirements_mean[l]) + "+-" + str(DistRequirements_std[l]))
    print("The mean sample complexity is: " + str(SampleComplexity_mean[l]) + "+-" + str(SampleComplexity_std[l]))
    print("-----------------------------------------------------------------------")

#print(NUMBER_INFEASIBILITIES)
# print(ConsViolationsRegret_mean)

title = 'OCSF-CMDP'
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
plt.ylabel('Minimum Distance')
plt.title(title)
plt.show()
distPlot.savefig('OCSF-CMDP - Distance.pdf', bbox_inches = 'tight')


samplePlot = plt.figure()
plt.plot(x, SampleComplexity_mean)
plt.fill_between(x, SampleComplexity_mean - SampleComplexity_std, SampleComplexity_mean + SampleComplexity_std, alpha = 0.5)
plt.grid()
plt.xlabel('Baseline Budget')
plt.ylabel('Sample Complexity')
plt.title(title)
plt.show()
samplePlot.savefig('OCSF-CMDP - Sample Complexity.pdf', bbox_inches = 'tight')