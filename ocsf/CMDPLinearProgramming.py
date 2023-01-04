#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pulp as p
import time
import math
import sys
import itertools
from functools import reduce
from operator import mul


class CMDPLPSolver:
    def __init__(self, delta, P, R, C, alphas, m, k, states, ACTIONS, CONSTRAINT):
        self.P = P.copy()
        self.R = R.copy()
        self.C = C.copy()
        self.delta = delta
        
        self.m = m
        self.k = k
        self.alphas = alphas
        self.states = states
        self.N_STATES = len(self.states)        
        self.ACTIONS = ACTIONS
        self.N_ACTIONS = len(self.ACTIONS)
        self.beta_e = 0
        
        self.P_hat = {}
        self.NUMBER_OF_OCCURANCES = {s: 0 for s in self.states} 
        self.P_hat = {s: 0 for s in self.states} 
        self.CONSTRAINT = CONSTRAINT
        
        self.Pstates = {self.states[l]: self.P[l] for l in range(self.N_STATES)}
        # for s in self.states:
        #     if self.Pstates[s] == 0.0:
        #         print(self.Pstates[s])
    

    def step(self):
        '''
        Draws a skill vector i.i.d. from the stationary distribution P.
        '''
        vec_idx = np.random.choice(range(self.N_STATES), 1, replace=True, p=self.P)[0]
        return self.states[vec_idx]
        # return list(itertools.product(*[np.random.choice(np.arange(self.alphas[i]), 1, replace=True, p=self.P[i]) for i in range(self.m)]))[0]

    def setCounts(self, state_count):
        '''
        Updates the nubmer of occurances of each state.
        '''
        for s in self.states:
            self.NUMBER_OF_OCCURANCES[s] += state_count[s]

    def compute_confidence_intervals(self, T_e):
        self.beta_e = math.sqrt(2 * self.N_STATES * math.log(3 * self.N_STATES * self.N_ACTIONS * T_e * (T_e - 1) / self.delta) / (T_e - 1))


    def update_empirical_model(self, T_e):
        for s in self.states:
            # if self.NUMBER_OF_OCCURANCES[s] == 0:
            #     self.P_hat[s] = 1 / self.N_STATES # Uniform Disritbution
            # else:
            self.P_hat[s] = self.NUMBER_OF_OCCURANCES[s] / (T_e - 1)
            # if abs(sum(self.P_hat[s][a]) - 1)  >  0.001:
            #     print("empirical is wrong")
                    #print(self.P_hat)
                

    def compute_opt_LP_Constrained(self, ep, is_tightened=False):
        opt_policy = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]
        opt_prob = p.LpProblem("OPT_LP_problem", p.LpMaximize)
        opt_varphi = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]
                                                                                  
        # Creating the problem's variables
        varphi_keys = [(s, a) for s in self.states for a in self.ACTIONS] # Occupancy Measure
        varphi = p.LpVariable.dicts("varphi", varphi_keys, lowBound=0, cat='Continuous')

        opt_prob += p.lpSum([varphi[(s,a)] * self.R[s][a] for s in self.states for a in self.ACTIONS])
            
        if is_tightened:
            for i in range(self.m):
                for j in range(self.alphas[i]): 
                    for q in range(self.k):
                        opt_prob += p.lpSum([varphi[(s,a)] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) - self.CONSTRAINT[i][j][q] <= 0
        else:
            for i in range(self.m):
                for j in range(self.alphas[i]): 
                    for q in range(self.k):
                        opt_prob += p.lpSum([varphi[(s,a)] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) == 0
            
        for s in self.states:
            opt_prob += p.lpSum([varphi[(s,a)] for a in self.ACTIONS]) - self.Pstates[s] == 0
        
        opt_prob += p.lpSum([varphi[(s,a)] for s in self.states for a in self.ACTIONS]) == 1

        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.001, msg = 0))

        #print(p.LpStatus[status])   # The solution status
        #print(opt_prob)
        #print("printing best value constrained")
        #print(p.value(opt_prob.objective))
                                                                                                                  
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                                                                                                                  
        for s in self.states:
            for a in self.ACTIONS:                
                opt_varphi[s][a] = varphi[(s,a)].varValue        
            opt_varphi_sum = np.sum([opt_varphi[s][act] for act in self.ACTIONS])             
            if opt_varphi_sum == 0:
                for a in self.ACTIONS:
                    opt_policy[s][a] = 1 / len(self.ACTIONS)
            else:
                for a in self.ACTIONS:
                    opt_policy[s][a] = opt_varphi[s][a] / opt_varphi_sum 
                    if math.isnan(opt_policy[s][a]):
                        opt_policy[s][a] = 1 / len(self.ACTIONS)
                    elif opt_policy[s][a] > 1.0:
                        print("The optimal policy contains an invalid value of: " + str(opt_policy[s][a]) + ", for the skill vector: " + str(s) + ", and the action: " + str(a))
                        #print(opt_policy[s,h,a])
            #probs = opt_policy[s,h,:]
            #optimal_policy[s,h] = int(np.argmax(probs))
                                                                                                                                                                  
        if ep != 0:
            return opt_policy, 0, 0, 0, 0, 0
        
        opt_gain = 0
        opt_con = [[[0 for q in range(self.k)] for j in range(self.alphas[i])] for i in range(self.m)]
        for s in self.states:
            for a in self.ACTIONS:
                if opt_policy[s][a] < 0.0:
                    opt_policy[s][a] = 0
                elif opt_policy[s][a] > 1.0:
                    opt_policy[s][a] = 1.0
                
                for i in range(self.m): 
                    for j in range(self.alphas[i]): 
                        for q in range(self.k):
                            opt_con[i][j][q] += opt_policy[s][a] * self.Pstates[s] * self.C[s][a][i][j][q]
                opt_gain += opt_policy[s][a] * self.Pstates[s] * self.R[s][a]
        # print("value from the conLPsolver")
        # print(opt_gain)
        # print(opt_con)

        varphi_policy, val_policy_per_vec, con_policy_per_vec = self.PolicyEvaluation(self.Pstates, opt_policy, self.R, self.C)
                                                                                                                                                                          
        return opt_policy, val_policy_per_vec, con_policy_per_vec, varphi_policy, opt_gain, opt_con
        
        # opt_policy = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]
        # opt_prob = p.LpProblem("OPT_LP_problem", p.LpMaximize)

        # # Creating the problem's variables
        # pi_keys = [(s, a) for s in self.states for a in self.ACTIONS] # Occupancy Measure
        # pi = p.LpVariable.dicts("pi", pi_keys, lowBound=0, upBound=1, cat='Continuous')

        # opt_prob += p.lpSum([pi[(s,a)] * self.Pstates[s] * self.R[s][a] for s in self.states for a in self.ACTIONS])
            
        # if is_tightened:
        #     for i in range(self.m):
        #         for j in range(self.alphas[i]): 
        #             for q in range(self.k):
        #                 opt_prob += p.lpSum([pi[(s,a)] * self.Pstates[s] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) - self.CONSTRAINT[i][j][q] <= 0
        # else:
        #     for i in range(self.m):
        #         for j in range(self.alphas[i]): 
        #             for q in range(self.k):
        #                 opt_prob += p.lpSum([pi[(s,a)] * self.Pstates[s] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) == 0

        # for s in self.states:
        #     opt_prob += p.lpSum([pi[(s,a)] for a in self.ACTIONS]) == 1

        # status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.001, msg = 0))

        # # print(p.LpStatus[status])   # The solution status
        # #print(opt_prob)
        # #print("printing best value constrained")
        # #print(p.value(opt_prob.objective))
                                                                                                                  
        # # for constraint in opt_prob.constraints:
        # #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                                                                                                                  
        # for s in self.states:
        #     for a in self.ACTIONS:                
        #         opt_policy[s][a] = pi[(s,a)].varValue
        #         if math.isnan(opt_policy[s][a]):
        #             print("isnan")
        #         elif opt_policy[s][a] > 1.0:
        #             print("The optimal policy contains an invalid value of: " + str(opt_policy[s][a]) + ", for the skill vector: " + str(s) + ", and the action: " + str(a))
                                                                                                                                                                  
        # if ep != 0:
        #     return opt_policy, 0, 0, 0, 0, 0
        
        # opt_gain = 0
        # opt_con = [[[0 for q in range(self.k)] for j in range(self.alphas[i])] for i in range(self.m)]
        # for s in self.states:
        #     for a in self.ACTIONS:
        #         if opt_policy[s][a] < 0.0:
        #             opt_policy[s][a] = 0
        #         elif opt_policy[s][a] > 1.0:
        #             opt_policy[s][a] = 1.0
                
        #         for i in range(self.m): 
        #             for j in range(self.alphas[i]): 
        #                 for q in range(self.k):
        #                     opt_con[i][j][q] += opt_policy[s][a] * self.Pstates[s] * self.C[s][a][i][j][q]
        #         opt_gain += opt_policy[s][a] * self.Pstates[s] * self.R[s][a]
        # # print("value from the conLPsolver")
        # # print(opt_gain)
        # # print(opt_con)

        # varphi_policy, val_policy_per_vec, con_policy_per_vec = self.PolicyEvaluation(self.Pstates, opt_policy, self.R, self.C)
                                                                                                                                                                          
        # return opt_policy, val_policy_per_vec, con_policy_per_vec, varphi_policy, opt_gain, opt_con
    


    def compute_opt_LP_Constrained_task(self, ep, q, is_tightened=False):
        opt_policy = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]
        opt_prob = p.LpProblem("OPT_LP_problem", p.LpMaximize)
        opt_varphi = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]
                                                                                  
        # Creating the problem's variables
        varphi_keys = [(s, a) for s in self.states for a in self.ACTIONS] # Occupancy Measure
        varphi = p.LpVariable.dicts("varphi", varphi_keys, lowBound=0, cat='Continuous')

        opt_prob += p.lpSum([varphi[(s,a)] * self.R[s][a] for s in self.states for a in self.ACTIONS])
            
        if is_tightened:
            for i in range(self.m):
                for j in range(self.alphas[i]): 
                    opt_prob += p.lpSum([varphi[(s,a)] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) - self.CONSTRAINT[i][j][q] <= 0
        else:
            for i in range(self.m):
                for j in range(self.alphas[i]): 
                    opt_prob += p.lpSum([varphi[(s,a)] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) == 0
            
        for s in self.states:
            opt_prob += p.lpSum([varphi[(s,a)] for a in self.ACTIONS]) - self.Pstates[s] == 0
        
        opt_prob += p.lpSum([varphi[(s,a)] for s in self.states for a in self.ACTIONS]) == 1

        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.001, msg = 0))

        #print(p.LpStatus[status])   # The solution status
        #print(opt_prob)
        #print("printing best value constrained")
        #print(p.value(opt_prob.objective))
                                                                                                                  
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                                                                                                                  
        for s in self.states:
            for a in self.ACTIONS:                
                opt_varphi[s][a] = varphi[(s,a)].varValue        
            opt_varphi_sum = np.sum([opt_varphi[s][act] for act in self.ACTIONS])             
            if opt_varphi_sum == 0:
                for a in self.ACTIONS:
                    opt_policy[s][a] = 1 / len(self.ACTIONS)
            else:
                for a in self.ACTIONS:
                    opt_policy[s][a] = opt_varphi[s][a] / opt_varphi_sum 
                    if math.isnan(opt_policy[s][a]):
                        opt_policy[s][a] = 1 / len(self.ACTIONS)
                    elif opt_policy[s][a] > 1.0:
                        print("The optimal policy contains an invalid value of: " + str(opt_policy[s][a]) + ", for the skill vector: " + str(s) + ", and the action: " + str(a))
                        #print(opt_policy[s,h,a])
            #probs = opt_policy[s,h,:]
            #optimal_policy[s,h] = int(np.argmax(probs))
                                                                                                                                                                  
        if ep != 0:
            return opt_policy, 0, 0, 0, 0, 0
        
        opt_gain = 0
        opt_con = [[[0 for q in range(self.k)] for j in range(self.alphas[i])] for i in range(self.m)]
        for s in self.states:
            for a in self.ACTIONS:
                if opt_policy[s][a] < 0.0:
                    opt_policy[s][a] = 0
                elif opt_policy[s][a] > 1.0:
                    opt_policy[s][a] = 1.0
                
                for i in range(self.m): 
                    for j in range(self.alphas[i]): 
                        for q in range(self.k):
                            opt_con[i][j][q] += opt_policy[s][a] * self.Pstates[s] * self.C[s][a][i][j][q]
                opt_gain += opt_policy[s][a] * self.Pstates[s] * self.R[s][a]
        # print("value from the conLPsolver")
        # print(opt_gain)
        # print(opt_con)

        varphi_policy, val_policy_per_vec, con_policy_per_vec = self.PolicyEvaluation(self.Pstates, opt_policy, self.R, self.C)
                                                                                                                                                                          
        return opt_policy, val_policy_per_vec, con_policy_per_vec, varphi_policy, opt_gain, opt_con
        
        # opt_policy = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]
        # opt_prob = p.LpProblem("OPT_LP_problem", p.LpMaximize)

        # # Creating the problem's variables
        # pi_keys = [(s, a) for s in self.states for a in self.ACTIONS] # Occupancy Measure
        # pi = p.LpVariable.dicts("pi", pi_keys, lowBound=0, upBound=1, cat='Continuous')

        # opt_prob += p.lpSum([pi[(s,a)] * self.Pstates[s] * self.R[s][a] for s in self.states for a in self.ACTIONS])
            
        # if is_tightened:
        #     for i in range(self.m):
        #         for j in range(self.alphas[i]): 
        #             for q in range(self.k):
        #                 opt_prob += p.lpSum([pi[(s,a)] * self.Pstates[s] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) - self.CONSTRAINT[i][j][q] <= 0
        # else:
        #     for i in range(self.m):
        #         for j in range(self.alphas[i]): 
        #             for q in range(self.k):
        #                 opt_prob += p.lpSum([pi[(s,a)] * self.Pstates[s] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) == 0

        # for s in self.states:
        #     opt_prob += p.lpSum([pi[(s,a)] for a in self.ACTIONS]) == 1

        # status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.001, msg = 0))

        # # print(p.LpStatus[status])   # The solution status
        # #print(opt_prob)
        # #print("printing best value constrained")
        # #print(p.value(opt_prob.objective))
                                                                                                                  
        # # for constraint in opt_prob.constraints:
        # #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                                                                                                                  
        # for s in self.states:
        #     for a in self.ACTIONS:                
        #         opt_policy[s][a] = pi[(s,a)].varValue
        #         if math.isnan(opt_policy[s][a]):
        #             print("isnan")
        #         elif opt_policy[s][a] > 1.0:
        #             print("The optimal policy contains an invalid value of: " + str(opt_policy[s][a]) + ", for the skill vector: " + str(s) + ", and the action: " + str(a))
                                                                                                                                                                  
        # if ep != 0:
        #     return opt_policy, 0, 0, 0, 0, 0
        
        # opt_gain = 0
        # opt_con = [[[0 for q in range(self.k)] for j in range(self.alphas[i])] for i in range(self.m)]
        # for s in self.states:
        #     for a in self.ACTIONS:
        #         if opt_policy[s][a] < 0.0:
        #             opt_policy[s][a] = 0
        #         elif opt_policy[s][a] > 1.0:
        #             opt_policy[s][a] = 1.0
                
        #         for i in range(self.m): 
        #             for j in range(self.alphas[i]): 
        #                 for q in range(self.k):
        #                     opt_con[i][j][q] += opt_policy[s][a] * self.Pstates[s] * self.C[s][a][i][j][q]
        #         opt_gain += opt_policy[s][a] * self.Pstates[s] * self.R[s][a]
        # # print("value from the conLPsolver")
        # # print(opt_gain)
        # # print(opt_con)

        # varphi_policy, val_policy_per_vec, con_policy_per_vec = self.PolicyEvaluation(self.Pstates, opt_policy, self.R, self.C)
                                                                                                                                                                          
        # return opt_policy, val_policy_per_vec, con_policy_per_vec, varphi_policy, opt_gain, opt_con
   

    def compute_extended_LP(self, is_tightened=False):
        opt_policy = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]
        opt_prob = p.LpProblem("OPT_LP_problem", p.LpMaximize)
        opt_varphi = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]

        # Creating the problem's variables
        
        varphi_keys = [(s, a) for s in self.states for a in self.ACTIONS] # Occupancy Measure
        varphi = p.LpVariable.dicts("varphi", varphi_keys, lowBound=0, cat='Continuous')
        
        # The variables beta(s) linearize the L1 constraint induced by the confidence set D_e.
        beta_keys = [(s) for s in self.states]
        beta = p.LpVariable.dicts("beta", beta_keys, lowBound=0, cat='Continuous')
        
        opt_prob += p.lpSum([varphi[(s,a)] * self.R[s][a] for s in self.states for a in self.ACTIONS])
            
        if is_tightened:
            for i in range(self.m):
                for j in range(self.alphas[i]): 
                    for q in range(self.k):
                        opt_prob += p.lpSum([varphi[(s,a)] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) - self.CONSTRAINT[i][j][q] <= 0
        else:
            for i in range(self.m):
                for j in range(self.alphas[i]): 
                    for q in range(self.k):
                        opt_prob += p.lpSum([varphi[(s,a)] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) == 0
            
        # for s in self.states:
        #     for a in self.ACTIONS:
        #         opt_prob += varphi[(s,a)] >= 0
        
        opt_prob += p.lpSum([varphi[(s,a)] for s in self.states for a in self.ACTIONS]) == 1
        
        for s in self.states:
            opt_prob += p.lpSum([varphi[(s,a)] for a in self.ACTIONS]) - (self.P_hat[s] + beta[(s)]) <= 0
            opt_prob += p.lpSum([varphi[(s,a)] for a in self.ACTIONS]) - (self.P_hat[s] - beta[(s)]) >= 0
            # for a in self.ACTIONS:
            opt_prob += beta[(s)] - self.beta_e <= 0
        
        
        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.01, msg = 0))
        # print(p.LpStatus[status])

        if p.LpStatus[status] != 'Optimal':
            for s in self.states:
                for a in self.ACTIONS:
                    opt_policy[s][a] = 1 / len(self.ACTIONS)
            return opt_policy, {s: 0 for s in self.states}, {s: 0 for s in self.states}, p.LpStatus[status], {s: {a: 0 for a in self.ACTIONS} for s in self.states}
                                                                                                                                                                                                                                                  
        # for h in range(self.EPISODE_LENGTH):
        for s in self.states:
            for a in self.ACTIONS:
                opt_varphi[s][a] = varphi[(s,a)].varValue
                if opt_varphi[s][a] < 0 and opt_varphi[s][a] > -0.001:
                    opt_varphi[s][a] = 0
                elif opt_varphi[s][a] < -0.001:
                    print("The optimal occupancy measure contains an invalid value of: " + str(opt_varphi[s][a]) + ", for the skill vector: " + str(s) + ", and the action: " + str(a))
                    sys.exit()

        for s in self.states:
            for a in self.ACTIONS:
                opt_varphi[s][a] = varphi[(s,a)].varValue
            opt_varphi_sum = np.sum([opt_varphi[s][act] for act in self.ACTIONS])
            if opt_varphi_sum == 0:
                # for s_1 in self.states:
                for a_1 in self.ACTIONS:
                    opt_policy[s][a_1] = 1 / len(self.ACTIONS)
                # break
            else:
                for a in self.ACTIONS:
                    opt_policy[s][a] = opt_varphi[s][a] / opt_varphi_sum
                    if math.isnan(opt_policy[s][a]):
                        # for s_1 in self.states:
                        for a_1 in self.ACTIONS:
                            opt_policy[s][a_1] = 1 / len(self.ACTIONS)
                        break
                    elif opt_policy[s][a] > 1.0:
                        # print("The optimal policy contains an invalid value of: " + str(opt_policy[s][a]) + ", for the skill vector: " + str(s) + ", and the action: " + str(a))
                        opt_policy[s][a] = 1.0
                    elif opt_policy[s][a] < 0.0:
                        opt_policy[s][a] = 0
                
                # if math.isnan(opt_policy[s][a]):
                #     break
        
        varphi_policy, val_policy_per_vec, con_policy_per_vec = self.PolicyEvaluation(self.Pstates, opt_policy, self.R, self.C)
        
        return opt_policy, val_policy_per_vec, con_policy_per_vec, p.LpStatus[status], varphi_policy
    
    def compute_extended_LP_task(self, q, is_tightened=False):
        opt_policy = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]
        opt_prob = p.LpProblem("OPT_LP_problem", p.LpMaximize)
        opt_varphi = {s: {a: 0 for a in self.ACTIONS} for s in self.states} #[s,a]

        # Creating the problem's variables
        
        varphi_keys = [(s, a) for s in self.states for a in self.ACTIONS] # Occupancy Measure
        varphi = p.LpVariable.dicts("varphi", varphi_keys, cat='Continuous')
        
        # The variables beta(s) linearize the L1 constraint induced by the confidence set D_e.
        # beta_keys = [(s) for s in self.states]
        # beta = p.LpVariable.dicts("beta", beta_keys, lowBound=0, cat='Continuous')
        
        opt_prob += p.lpSum([varphi[(s,a)] * self.R[s][a] for s in self.states for a in self.ACTIONS])
            
        if is_tightened:
            for i in range(self.m):
                for j in range(self.alphas[i]): 
                    opt_prob += p.lpSum([varphi[(s,a)] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) - self.CONSTRAINT[i][j][q] <= 0
        else:
            for i in range(self.m):
                for j in range(self.alphas[i]): 
                    opt_prob += p.lpSum([varphi[(s,a)] * self.C[s][a][i][j][q] for s in self.states for a in self.ACTIONS]) == 0
            
        # for s in self.states:
        #     for a in self.ACTIONS:
        #         opt_prob += varphi[(s,a)] >= 0
        
        opt_prob += p.lpSum([varphi[(s,a)] for s in self.states for a in self.ACTIONS]) == 1
        
        for s in self.states:
            opt_prob += p.lpSum([varphi[(s,a)] for a in self.ACTIONS]) - (self.P_hat[s] + self.beta_e) <= 0
            opt_prob += p.lpSum([varphi[(s,a)] for a in self.ACTIONS]) - (self.P_hat[s] - self.beta_e) >= 0
            # for a in self.ACTIONS:
            # opt_prob += beta[(s)] - self.beta_e <= 0
        
        
        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.01, msg = 0))
        # print(p.LpStatus[status])

        if p.LpStatus[status] != 'Optimal':
            for s in self.states:
                for a in self.ACTIONS:
                    opt_policy[s][a] = 1 / len(self.ACTIONS)
            return opt_policy, {s: 0 for s in self.states}, {s: 0 for s in self.states}, p.LpStatus[status], {s: {a: 0 for a in self.ACTIONS} for s in self.states}
                                                                                                                                                                                                                                                  
        # for h in range(self.EPISODE_LENGTH):
        for s in self.states:
            for a in self.ACTIONS:
                opt_varphi[s][a] = varphi[(s,a)].varValue
                if opt_varphi[s][a] < 0 and opt_varphi[s][a] > -0.001:
                    opt_varphi[s][a] = 0
                elif opt_varphi[s][a] < -0.001:
                    print("The optimal occupancy measure contains an invalid value of: " + str(opt_varphi[s][a]) + ", for the skill vector: " + str(s) + ", and the action: " + str(a))
                    sys.exit()

        for s in self.states:
            for a in self.ACTIONS:
                opt_varphi[s][a] = varphi[(s,a)].varValue
            opt_varphi_sum = np.sum([opt_varphi[s][act] for act in self.ACTIONS])
            if opt_varphi_sum == 0:
                # for s_1 in self.states:
                for a_1 in self.ACTIONS:
                    opt_policy[s][a_1] = 1 / len(self.ACTIONS)
                # break
            else:
                for a in self.ACTIONS:
                    opt_policy[s][a] = opt_varphi[s][a] / opt_varphi_sum
                    if math.isnan(opt_policy[s][a]):
                        # for s_1 in self.states:
                        for a_1 in self.ACTIONS:
                            opt_policy[s][a_1] = 1 / len(self.ACTIONS)
                        # break
                    elif opt_policy[s][a] > 1.0:
                        print("The optimal policy contains an invalid value of: " + str(opt_policy[s][a]) + ", for the skill vector: " + str(s) + ", and the action: " + str(a))
                
                # if math.isnan(opt_policy[s][a]):
                #     break
        
        varphi_policy, val_policy_per_vec, con_policy_per_vec = self.PolicyEvaluation(self.Pstates, opt_policy, self.R, self.C)
        
        return opt_policy, val_policy_per_vec, con_policy_per_vec, p.LpStatus[status], varphi_policy
    
    
    # def compute_LP_Tao(self, ep, cb):
    #     opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
    #     opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
    #     opt_varphi = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES)) #[h,s,a]
                                                                                  
    #     #create problem variables
    #     varphi_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in self.states for a in self.ACTIONS]
                                                                                          
    #     q = p.LpVariable.dicts("q",varphi_keys,lowBound=0,cat='Continuous')
        
        
        
        
    #     alpha_r = 1 + self.N_STATES*self.EPISODE_LENGTH + 4*self.EPISODE_LENGTH*(1+self.N_STATES*self.EPISODE_LENGTH)/(self.CONSTRAINT-cb)
    #     for s in self.states:
    #         l = len(self.ACTIONS)
    #         self.R_Tao[s] = np.zeros(l)
    #         for a in self.ACTIONS:
    #             self.R_Tao[s][a] = self.R[s][a] - alpha_r * self.beta_prob_T[s][a]
                
       
        
    #     for s in self.states:
    #         l = len(self.ACTIONS)
    #         self.C_Tao[s] = np.zeros(l)
    #         for a in self.ACTIONS:
    #             self.C_Tao[s][a] = self.C[s][a] + (1 + self.EPISODE_LENGTH * self.N_STATES)*self.beta_prob_T[s][a]

    #     opt_prob += p.lpSum([q[(h,s,a)]*self.R_Tao[s][a] for  h in range(self.EPISODE_LENGTH) for s in self.states for a in self.ACTIONS])
            
    #     opt_prob += p.lpSum([q[(h,s,a)]*self.C_Tao[s][a] for h in range(self.EPISODE_LENGTH) for s in self.states for a in self.ACTIONS]) - self.CONSTRAINT <= 0
            
    #     for h in range(1,self.EPISODE_LENGTH):
    #         for s in self.states:
    #             q_list = [q[(h,s,a)] for a in self.ACTIONS]
    #             pq_list = [self.P_hat[s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in self.states for a_1 in self.ACTIONS[s_1]]
    #             opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0

    #     for s in self.states:
    #         q_list = [q[(0,s,a)] for a in self.ACTIONS]
    #         opt_prob += p.lpSum(q_list) - self.mu[s] == 0

    #     status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.001, msg = 0))
    #     #print(p.LpStatus[status])   # The solution status
    #     #print(opt_prob)
    #     # print("printing best value constrained")
    #     # print(p.value(opt_prob.objective))
                                                                                                                  
    #     # for constraint in opt_prob.constraints:
    #     #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                                                                                                                  
    #     for h in range(self.EPISODE_LENGTH):
    #         for s in self.states:
    #             for a in self.ACTIONS:
    #                 opt_varphi[h,s,a] = q[(h,s,a)].varValue
    #             for a in self.ACTIONS:
    #                 if np.sum(opt_varphi[h,s,:]) == 0:
    #                     opt_policy[s,h,a] = 1/len(self.ACTIONS)
    #                 else:
    #                     opt_policy[s,h,a] = opt_varphi[h,s,a]/np.sum(opt_varphi[h,s,:])
    #                     if math.isnan(opt_policy[s,h,a]):
    #                         opt_policy[s,h,a] = 1/len(self.ACTIONS)
    #                     elif opt_policy[s,h,a] > 1.0:
    #                         print("invalid value printing")
    #                         #print(opt_policy[s,h,a])
    #             #probs = opt_policy[s,h,:]
    #             #optimal_policy[s,h] = int(np.argmax(probs))
                                                                                                                                                                  
    #     if ep != 0:
    #         return opt_policy, 0, 0, 0
        
        
    #     for h in range(self.EPISODE_LENGTH):
    #      for s in self.states:
    #         for a in self.ACTIONS:
    #             if opt_varphi[h,s,a] < 0:
    #                     opt_varphi[h,s,a] = 0
    #             elif opt_varphi[h,s,a] > 1:
    #                 opt_varphi[h,s,a] = 1.0
                    

    #     varphi_policy, val_policy_per_vec, con_policy_per_vec = self.PolicyEvaluation(self.Pstates, opt_policy, self.R, self.C)
                                                                                                                                                                          
    #     return opt_policy, val_policy_per_vec, con_policy_per_vec, varphi_policy
    


    def PolicyEvaluation(self, Px, policy, R, C):
        varphi = {s: {a: 0 for a in self.ACTIONS} for s in self.states}
        v = {s: 0 for s in self.states}
        c = [[[{s: 0 for s in self.states} for q in range(self.k)] for j in range(self.alphas[i])] for i in range(self.m)]
        P_policy = {s: 0 for s in self.states}
        R_policy = {s: 0 for s in self.states}
        C_policy = [[[{s: 0 for s in self.states} for q in range(self.k)] for j in range(self.alphas[i])] for i in range(self.m)]

        for s in self.states:
            for i in range(self.m): 
                for j in range(self.alphas[i]): 
                    for q in range(self.k):
                        x = 0
                        for a in self.ACTIONS:
                            x += policy[s][a] * C[s][a][i][j][q]
                
                        c[i][j][q][s] = x 

            for a in self.ACTIONS:
                varphi[s][a] = R[s][a]
            v[s] = np.dot(np.array([varphi[s][a] for a in self.ACTIONS]), np.array([policy[s][a] for a in self.ACTIONS]))

        for s in self.states:
            x = 0
            z = 0
            for a in self.ACTIONS:
                x += policy[s][a] * R[s][a]
                z += policy[s][a] * Px[s]
            R_policy[s] = x
            P_policy[s] = z
            
            for i in range(self.m): 
                for j in range(self.alphas[i]): 
                    for q in range(self.k):
                        y = 0
                        for a in self.ACTIONS:
                            y += policy[s][a] * C[s][a][i][j][q]
                        C_policy[i][j][q][s] = y

        for s in self.states:
            for i in range(self.m): 
                for j in range(self.alphas[i]): 
                    for q in range(self.k):
                        c[i][j][q][s] = C_policy[i][j][q][s] + np.dot(np.array([P_policy[s] for s_1 in self.states]), np.array([c[i][j][q][s_1] for s_1 in self.states]))
                        # print("constraint " + str(i) + str(j) + str(q) + str(s), c[i][j][q][s])
            for a in self.ACTIONS:
                z = 0
                for s_1 in self.states:
                    z += Px[s] * v[s_1]
                varphi[s][a] = R[s][a] + z
            v[s] = np.dot(np.array([varphi[s][a] for a in self.ACTIONS]), np.array([policy[s][a] for a in self.ACTIONS]))
            # print("evaluation " + str(s), v[s])
        # print("evaluation", v[s])
        # print("constraint", c[i][j][q][s])
                
        return varphi, v, c