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

def parse_args():
    parser = argparse.ArgumentParser("Online Coalitional Skill Vector Games - Greedy")
    parser.add_argument('-mul', "--multiplier", type=bool, default=True)
    parser.add_argument('-bm', "--budget_size", type=int, default=20)
    parser.add_argument('-nb', "--n_budget_sizes", type=int, default=6)
    parser.add_argument('-e', "--epsilon", type=int, default=0.01)
    return parser.parse_args()


arglist = parse_args()
budget_size = arglist.budget_size
n_budget_sizes = arglist.n_budget_sizes
EPS = arglist.epsilon

# temp = sys.argv[1:]
# RUN_NUMBER = int(temp[0])

RUN_NUMBER = 11

random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

#np.random.seed(RUN_NUMBER)


#Initialize:
for l in range(n_budget_sizes):  
    if n_budget_sizes == 1:
        filename = 'GREEDY' + str(EPS) + '.pckl'
        f = open('model.pckl', 'rb')
    elif arglist.multiplier:
        filename = 'GREEDY-' + str(EPS) + '-' + str(budget_size + 20 * l) + '.pckl'
        f = open('model-mul' + str(budget_size + 20 * l) + '.pckl', 'rb')
        print("Initiating the test for a budget of size " + str(budget_size + 20 * l) + "q for task tau_q")
    else:
        filename = 'GREEDY-' + str(EPS) + '-' + str(budget_size + 50 * l) + '.pckl'
        f = open('model' + str(budget_size + 50 * l) + '.pckl', 'rb')
        print("Initiating the test for a budget of size " + str(budget_size + 50 * l) + " for each task")
    [NUMBER_SIMULATIONS, P, R, C, CONSTRAINT, C_b, alphas, m, k, budgets, states, actions, delta, goals] = pickle.load(f)
    f.close()
    
    NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)
    
    if l == 0:
        DistRequirements_mean = [0 for h in range(n_budget_sizes)]
        DistRequirements_std = [0 for h in range(n_budget_sizes)]
        SampleComplexity_mean = [0 for h in range(n_budget_sizes)]
        SampleComplexity_std = [0 for h in range(n_budget_sizes)]

    coalition_structure = [[[] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    skill_coverage = [[ [ [ 0 for j in range(alphas[i])] for i in range(m) ] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    sample_complexity = [0 for sim in range(NUMBER_SIMULATIONS)]
    
    lp_solver = CMDPLPSolver(delta, P, R, C, alphas, m, k, states, actions, CONSTRAINT)
    
    for sim in range(NUMBER_SIMULATIONS):  
        t = 0
        while len([q for q in range(1, k + 1) if len(coalition_structure[sim][q-1]) == budgets[q-1]]) < k:            
            s = lp_solver.step()
    
            plausible_actions = [q for q in range(k) if len(coalition_structure[sim][q]) < budgets[q]]
            n_quota = 0
            greedy_coalition = -1
            for q in plausible_actions:
                coalition_quota = 0
                for i in range(m):
                    for j in range(alphas[i]):
                        if skill_coverage[sim][q][i][j] + (s[i] == j) <= math.ceil(goals[q][i][j] * budgets[q]) + EPS * budgets[q] / (alphas[i] - 1):
                            coalition_quota += 1
                
                if coalition_quota > n_quota:
                    n_quota = coalition_quota
                    greedy_coalition = q
                    
            if greedy_coalition != -1 and n_quota >= 1:
                coalition_structure[sim][greedy_coalition].append(s)
                
                for i in range(m):
                    skill_coverage[sim][greedy_coalition][i][s[i]] += 1
                            
            if t % 100 == 0:
                # filename = 'GREEDY' + str(EPS) + '.pckl'
                f = open(filename, 'ab')
                pickle.dump([NUMBER_SIMULATIONS, sample_complexity, skill_coverage], f)
                f.close()
            
            t += 1

        
        f = open(filename, 'ab')
        pickle.dump([NUMBER_SIMULATIONS, sample_complexity, skill_coverage], f)
        f.close()

        print("Simulation nubmer " + str(sim) + " has terminated after " + str(t) + " time steps")
        sample_complexity[sim] = t
            
        
    dists_per_task = [[0 for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    for sim in range(NUMBER_SIMULATIONS):
        for q in range(k):
            for i in range(m): 
                for j in range(alphas[i]):
                    skill_coverage[sim][q][i][j] /= len(coalition_structure[sim][q])
                    if np.abs(skill_coverage[sim][q][i][j] - goals[q][i][j]) > dists_per_task[sim][q]:
                        dists_per_task[sim][q] = np.abs(skill_coverage[sim][q][i][j] - goals[q][i][j])
    
    dists = [max(dists_per_task[sim]) for sim in range(NUMBER_SIMULATIONS)]
    
    DistRequirements_mean[l] = np.mean(dists, axis = 0)
    DistRequirements_std[l] = np.std(dists, axis = 0)
    SampleComplexity_mean[l] = np.mean(sample_complexity, axis = 0)
    SampleComplexity_std[l] = np.std(sample_complexity, axis = 0)
    
    print("The mean maximum distance between a coalition's skill coverage and its corresponding task's goal is: " + str(DistRequirements_mean[l]) + "+-" + str(DistRequirements_std[l]))
    print("The mean sample complexity is: " + str(SampleComplexity_mean[l]) + "+-" + str(SampleComplexity_std[l]))
    print("-----------------------------------------------------------------------")
#print(NUMBER_INFEASIBILITIES)
# print(ConsViolationsRegret_mean)

title = 'GREEDY ' + str(EPS)
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
distPlot.savefig('GREEDY - ' + str(EPS) + ' - Distance.pdf', bbox_inches = 'tight')


samplePlot = plt.figure()
plt.plot(x, SampleComplexity_mean)
plt.fill_between(x, SampleComplexity_mean - SampleComplexity_std, SampleComplexity_mean + SampleComplexity_std, alpha = 0.5)
plt.grid()
plt.xlabel('Baseline Budget')
plt.ylabel('Sample Complexity')
plt.title(title)
plt.show()
samplePlot.savefig('GREEDY - ' + str(EPS) + ' - Sample Complexity.pdf', bbox_inches = 'tight')