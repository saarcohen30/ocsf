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


def get_dists(k, m, alphas, sim, skill_coverage, coalition_structure, goals):
    dists_per_task = [0 for q in range(k)]
    for q in range(k):
        for i in range(m): 
            for j in range(alphas[i]):
                skill_coverage[sim][q][i][j] /= len(coalition_structure[sim][q])
                if np.abs(skill_coverage[sim][q][i][j] - goals[q][i][j]) > dists_per_task[q]:
                    dists_per_task[q] = np.abs(skill_coverage[sim][q][i][j] - goals[q][i][j])
    
    return dists_per_task

def parse_args():
    parser = argparse.ArgumentParser("Online Coalitional Skill Vector Games - MinGREEDY")
    parser.add_argument('-mul', "--multiplier", type=bool, default=True)
    parser.add_argument('-bm', "--budget_size", type=int, default=20)
    parser.add_argument('-m', "--num_skills", type=int, default=6, help="The number of skills")
    parser.add_argument('-nb', "--n_budget_sizes", type=int, default=6)
    parser.add_argument('-e', "--epsilon", type=float, default=0.01)
    parser.add_argument('-bd', "--baseline_budgets", nargs="+", default=[50, 100, 150, 250, 500, 1000], help="Baseline budgets")
    return parser.parse_args()


arglist = parse_args()
m = arglist.num_skills
budget_size = arglist.budget_size
n_budget_sizes = arglist.n_budget_sizes
EPS = arglist.epsilon
baseline_budgets = arglist.baseline_budgets

# temp = sys.argv[1:]
# RUN_NUMBER = int(temp[0])

RUN_NUMBER = 11

random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

#np.random.seed(RUN_NUMBER)

DistRequirements_mean = [0 for h in range(n_budget_sizes)]
DistRequirements_std = [0 for h in range(n_budget_sizes)]
SampleComplexity_mean = [0 for h in range(n_budget_sizes)]
SampleComplexity_std = [0 for h in range(n_budget_sizes)]
        
#Initialize:
for l in range(n_budget_sizes):  
    if n_budget_sizes == 1:
        filename = 'MinGREEDY' + str(EPS) + '.pckl'
        f = open('model.pckl', 'rb')
    elif arglist.multiplier:
        filename = 'MinGREEDY-' + str(EPS) + '-' + str(baseline_budgets[l]) + '-mul-' + str(m) + '.pckl'
        f = open('model-mul' + str(baseline_budgets[l]) + '.pckl', 'rb')
        print("Initiating the test for a budget of size " + str(baseline_budgets[l]) + "q for task tau_q")
    else:
        filename = 'MinGREEDY-' + str(EPS) + '-' + str(baseline_budgets[l]) + '-' + str(m) + '.pckl'
        f = open('model' + str(baseline_budgets[l]) + '.pckl', 'rb')
        print("Initiating the test for a budget of size " + str(baseline_budgets[l]) + " for each task")
    [NUMBER_SIMULATIONS, P, R, C, CONSTRAINT, C_b, alphas, m, k, budgets, states, actions, delta, goals] = pickle.load(f)
    f.close()
    
    NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)
    
    # if l == 0:
    dists_per_task = [[0 for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    coalition_structure = [[[] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    skill_coverage = [[ [ [ 0 for j in range(alphas[i])] for i in range(m) ] for q in range(k)] for sim in range(NUMBER_SIMULATIONS)]
    sample_complexity = [0 for sim in range(NUMBER_SIMULATIONS)]
    
    lp_solver = CMDPLPSolver(delta, P, R, C, alphas, m, k, states, actions, CONSTRAINT)
    
    for sim in range(NUMBER_SIMULATIONS):  
        t = 0
        while len([q for q in range(k) if len(coalition_structure[sim][q]) == budgets[q]]) < k:            
            s = lp_solver.step()
            
            plausible_actions = [q for q in range(k) if len(coalition_structure[sim][q]) < budgets[q]]
            n_quota = 0
            greedy_plausible_actions = []
            for q in plausible_actions:
                assign = True
                for i in range(m):
                    for j in range(alphas[i]):
                        if alphas[i] == 1:
                            if skill_coverage[sim][q][i][j] + (s[i] == j) > math.ceil(goals[q][i][j] * budgets[q]):
                                assign = False
                        else:
                            if skill_coverage[sim][q][i][j] + (s[i] == j) > math.ceil(goals[q][i][j] * budgets[q]) + EPS * budgets[q] / (alphas[i] - 1):
                                assign = False
                                
                if assign:
                    greedy_plausible_actions.append(q)
    
            # plausible_actions = [q for q in range(k) if len(coalition_structure[sim][q]) < budgets[q]]
            # # n_quota = 0
            # # greedy_coalition = -1
            # for q in plausible_actions:
            #     for i in range(m):
            #         for j in range(alphas[i]):
            #             if alphas[i] == 1:
            #                 if skill_coverage[sim][q][i][j] + (s[i] == j) > math.ceil(goals[q][i][j] * budgets[q]):
            #                     plausible_actions.remove(q)
            #                     break
            #             elif skill_coverage[sim][q][i][j] + (s[i] == j) > math.ceil(goals[q][i][j] * budgets[q]) + EPS * budgets[q] / (alphas[i] - 1):
            #                     plausible_actions.remove(q)
            #                     break
            #         if q not in plausible_actions:
            #             break
                    
            #     if q not in plausible_actions:
            #         break
                # if assign:
                #     greedy_coalition = q
                #     break
            # print(plausible_actions)  
                   
            if len(greedy_plausible_actions) != 0:
                greedy_coalition = -1
                greedy_dist = float('inf')
                dists_per_task_sim = [0 for q in range(k)] 
                skill_coverage_sim = [ [ [ 0 for j in range(alphas[i])] for i in range(m) ] for q in range(k)] 
                for q in range(k):
                    for i in range(m): 
                        for j in range(alphas[i]):
                            skill_coverage_sim[q][i][j] = skill_coverage[sim][q][i][j]
                            skill_coverage_sim[q][i][s[i]] += 1
                                
                            skill_coverage_sim[q][i][j] /= len(coalition_structure[sim][q])+1
                            if np.abs(skill_coverage_sim[q][i][j] - goals[q][i][j]) > dists_per_task_sim[q]:
                                dists_per_task_sim[q] = np.abs(skill_coverage_sim[q][i][j] - goals[q][i][j])

                for q in greedy_plausible_actions:
                    if dists_per_task_sim[q] < greedy_dist:
                        greedy_dist = dists_per_task_sim[q]
                        greedy_coalition = q

                if greedy_coalition != -1:
                    coalition_structure[sim][greedy_coalition].append(s)
                    
                    for i in range(m):
                        skill_coverage[sim][greedy_coalition][i][s[i]] += 1
                            
            # if t % 100 == 0:
            #     # filename = 'MinGREEDY' + str(EPS) + '.pckl'
            #     f = open(filename, 'ab')
            #     pickle.dump([NUMBER_SIMULATIONS, sample_complexity, skill_coverage, coalition_structure], f)
            #     f.close()
            
            t += 1

        
        sample_complexity[sim] = t
        print("Simulation number " + str(sim) + " has terminated after " + str(t) + " time steps")
        
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

title = 'MinGREEDY ' + str(EPS)
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
distPlot.savefig('MinGREEDY - ' + str(EPS) + ' - Distance.pdf', bbox_inches = 'tight')


samplePlot = plt.figure()
plt.plot(x, SampleComplexity_mean)
plt.fill_between(x, SampleComplexity_mean - SampleComplexity_std, SampleComplexity_mean + SampleComplexity_std, alpha = 0.5)
plt.grid()
plt.xlabel('Baseline Budget')
plt.ylabel('Sample Complexity')
plt.title(title)
plt.show()
samplePlot.savefig('MinGREEDY - ' + str(EPS) + ' - Sample Complexity.pdf', bbox_inches = 'tight')