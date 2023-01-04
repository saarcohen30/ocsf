#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:40:31 2021

@author: Anonymous
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CMDPLinearProgramming import CMDPLPSolver
import sys
import pickle
import time
import pulp as p
import math
from copy import copy
from operator import mul
import argparse
import itertools
from functools import reduce


def parse_args():
    parser = argparse.ArgumentParser("Online Coalitional Skill Vector Games")
    parser.add_argument('-k', "--num_tasks", type=int, default=10, help="The number of tasks")
    parser.add_argument('-m', "--num_skills", type=int, default=3, help="The number of skills")
    parser.add_argument('-a', "--mastering_levels", nargs="+", default=[2, 2, 2], help="Mastering levels of each skill") #[2] + [i for i in range(2, 7)]
    parser.add_argument('-bd', "--baseline_budgets", nargs="+", default=[50, 100, 150, 250, 500, 1000], help="Baseline budgets")
    parser.add_argument('-b', "--budgets", nargs="+", default=[20 * i for i in range(1, 11)], help="A list of the budgets for each task")
    parser.add_argument('-mul', "--multiplier", type=bool, default=True)
    parser.add_argument('-bm', "--budget_size", type=int, default=20)
    parser.add_argument('-nb', "--n_budget_sizes", type=int, default=6)
    return parser.parse_args()


arglist = parse_args()

k = arglist.num_tasks
m = arglist.num_skills
alphas = arglist.mastering_levels
budget_size = arglist.budget_size
n_budget_sizes = arglist.n_budget_sizes
baseline_budgets = arglist.baseline_budgets
if n_budget_sizes == 1:
    budgets = arglist.budgets
elif budget_size > 0:
    if arglist.multiplier:
        budgets = [[baseline_budgets[l] * i for i in range(1, 11)] for l in range(n_budget_sizes)]
    else:
        budgets = [[baseline_budgets[l] for i in range(1, 11)] for l in range(n_budget_sizes)]

# N_STATES = reduce(mul, alphas)
# print("Number of states: " + str(N_STATES))

states = list(itertools.product(*[range(alphas[i]) for i in range(m)]))
N_STATES = len(states)

# Creating a random stationary distribution P over skill vectors 
P = np.random.random(N_STATES).tolist()
P /= np.sum(P)

# P = [np.random.random(alphas[i]).tolist() for i in range(m)]
# for i in range(m):
#   for j in range(alphas[i]):
#     P[i][j] /= sum(sum(P,[]))

# Creating random goals for each task 
goals = []
for q in range(k):
    task_goal = []
    for i in range(m):
        probs = np.random.random(alphas[i])
        task_goal += [(probs / np.sum(probs)).tolist()]
    goals += [task_goal]

def reward(a):
    if a != 0:
        return 1
    return 0

def cost_constraint(a, s, i, j, q):
    if a != 0:
        if a == q+1 and s[i] == j:
            return 1 - goals[q][i][j]
        elif a == q+1 and s[i] != j:
            return - goals[q][i][j]
        else:
            return 0
    return 0

# P = {}
R = {}
C = {}
actions = range(k + 1)

for s in states:
    R[s] = {a: reward(a) for a in actions}
    C[s] = {}
    for a in actions:
        C[s][a] = {}
        for i in range(m):
            C[s][a][i] = {}
            for j in range(alphas[i]):
                C[s][a][i][j] = {}
                for q in range(k):
                    C[s][a][i][j][q] = cost_constraint(a, s, i, j, q)
    
#print(P[0][1][2])
# print(R)
# print(C)
# print(P[0][1])


CONSTRAINT = [[[1e-10 for q in range(k)] for j in range(alphas[i])] for i in range(m)] #0.0005

C_b = [[[CONSTRAINT[i][j][q] / 5 for q in range(k)] for j in range(alphas[i])] for i in range(m)] # Change this if you want a different baseline policy.

# NUMBER_EPISODES = 1e5
NUMBER_SIMULATIONS = 5

delta = 0.01


lp_solver = CMDPLPSolver(delta, P, R, C, alphas, m, k, states, actions, CONSTRAINT)
opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_gain, opt_con = lp_solver.compute_opt_LP_Constrained(0)
# opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon = lp_solver.compute_opt_LP_Unconstrained(0)
f = open('solution.pckl', 'wb')
pickle.dump([opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_gain, opt_con], f)
f.close()

print('*******')

lp_solver = CMDPLPSolver(delta, P, R, C, alphas, m, k, states, actions, C_b)
policy_b, value_b, cost_b, q_b, gain_b, con_b = lp_solver.compute_opt_LP_Constrained(0, is_tightened=True)
f = open('baseline.pckl', 'wb')
pickle.dump([policy_b, value_b, cost_b, q_b, gain_b, con_b], f)
f.close()


if n_budget_sizes == 1:
    f = open('model.pckl', 'wb')
    pickle.dump([NUMBER_SIMULATIONS, P, R, C, CONSTRAINT, C_b, alphas, m, k, budgets, states, actions, delta, goals], f)
    f.close()
else:
    for l in range(n_budget_sizes):
        if arglist.multiplier:
            f = open('model-mul' + str(baseline_budgets[l]) + '.pckl', 'wb')
        else:
            f = open('model' + str(baseline_budgets[l]) + '.pckl', 'wb')
        pickle.dump([NUMBER_SIMULATIONS, P, R, C, CONSTRAINT, C_b, alphas, m, k, budgets[l], states, actions, delta, goals], f)
        f.close()

# print('*******')
# print(opt_value_LP_con,opt_cost_LP_con)
# print(value_b,cost_b)
