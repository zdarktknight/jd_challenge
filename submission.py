#!/usr/bin/env python
'''
sample submission
'''

# import all modules been used 
import pandas as pd
import numpy as np
import scipy.stats as sp
import copy as cp

class UserPolicy:
    def __init__(self, initial_inventory, inventory_replenishment, sku_demand_distribution, sku_cost):
        self.inv = [initial_inventory]
        self.replenish = inventory_replenishment
        self.distribution = sku_demand_distribution
        self.cost = sku_cost
        self.sku_limit = np.asarray([200, 200, 200, 200, 200])
        self.extra_shipping_cost_per_unit = 0.01
        self.capacity_limit = np.asarray([3200, 1600, 1200, 3600, 1600])
        self.abandon_rate =np.asarray([1./100, 7./100, 10./100, 9./100, 8./100])
        # Define additional arrays for convenience
        self.sku_list = [i for i in range(1,1001)]
        self.dc_list = list(range(6))
        self.cnt = 0
        # Tuning parameters
        self.docheck = False # If or not check the feasibility after each iteration
        self.consider_demand = False # If or not use the worst demand for setting w_up
        self.show_obj = False # If or not calculate objective
        self.demand_percentage_lo = 0.5 # Suggested minimum percentage of stocks for FDC
        self.demand_percentage_up = 0.8 # Suggested maximum percentage of stocks for FDC
        self.percentile = 0.999 # Percentage for the worst case
        # Preprocessing input data
        for i in range(0, len(self.inv)):
            self.inv[i] = self.inv[i].sort_values(['dc_id','item_sku_id'])
        self.distribution = self.distribution.sort_values(['dc_id', 'item_sku_id'])
        self.replenish = self.replenish.sort_values(['item_sku_id', 'date'])
        self.cost_value = np.asarray(self.cost)[:,1]
        self.cost = self.cost.sort_values('stockout_cost')
        self.cost_sorted_index = np.asarray(self.cost)[:,0]

        # tong: Assign mean values as deterministic sku_demand
        self.dR_it = np.zeros((1000)).astype(int)
        self.d_ijt = np.zeros((1000, 5)).astype(int)
        [self.dR_it, self.d_ijt] = self.demand_quantile()
        
    # Tong: calculate 0.999 percentile (06/05)
    def demand_quantile(self, my_percentile = 0):
        if my_percentile == 0:
            my_percentile=self.percentile
        for sku_id in self.sku_list:
            for dc_id in self.dc_list:
                row = self.distribution.loc[(self.distribution.item_sku_id == sku_id) & (self.distribution.dc_id == dc_id)]
                dist_type = row.dist_type.iloc[0]
                para1 = row.para1.astype(float).iloc[0]
                para2 = row.para2.astype(float).iloc[0]             
                if dist_type == 'N':
                    ng_bi = sp.nbinom(para1, para2)
                    if dc_id == 0:
                        self.dR_it[sku_id-1] = np.ceil(ng_bi.ppf(my_percentile) )
                    else:
                        self.d_ijt[sku_id-1, dc_id-1] = np.ceil(ng_bi.ppf(my_percentile))
                elif dist_type == 'G':
                    g = sp.gamma(para1, scale = para2)
                    if dc_id == 0:
                        self.dR_it[sku_id-1] = np.ceil(g.ppf(my_percentile))
                    else:
                        self.d_ijt[sku_id-1, dc_id-1] = np.ceil(g.ppf(my_percentile))
        
        # # Assign mean values as deterministic sku_demand
        # self.dR_it = np.zeros((1000)).astype(int)
        # self.d_ijt = np.zeros((1000, 5)).astype(int)
        # for sku_id in self.sku_list:
        #     for dc_id in self.dc_list:
        #         row = self.distribution.loc[(self.distribution.item_sku_id == sku_id) & (self.distribution.dc_id == dc_id)]
        #         dist_type = row.dist_type.iloc[0]
        #         para1 = row.para1.astype(float).iloc[0]
        #         para2 = row.para2.astype(float).iloc[0]
        #         if dist_type == 'N':
        #             if dc_id == 0:
        #                 self.dR_it[sku_id-1] = para1*(1-para2)/para2
        #             else:
        #                 self.d_ijt[sku_id-1, dc_id-1] = para1*(1-para2)/para2
        #         elif dist_type == 'G':
        #             if dc_id == 0:
        #                 self.dR_it[sku_id-1] = para1*para2
        #             else:
        #                 self.d_ijt[sku_id-1, dc_id-1] = para1*para2
        #         else:
        #             if dc_id == 0:
        #                 print ("No distribution for sku", sku_id, "in RDC")
        #             else:
        #                 print ("No distribution for sku", sku_id, "in FDC", dc_id)
        return (self.dR_it, self.d_ijt)
    # Tong: end

    "Infer the demand in the last period by the current inventory and previous information"
    def demand_infer(self, inv_last, w_last, alpha_j):
        inv_current = self.invFilter(False)
        d_last = np.zeros((1000, 5)).astype(int)
        for i in range(0,1000):
            for j in range(0, 5):
                if w_last[i,j] > inv_current[i, j]:
                    d_last[i, j] = np.ceil((w_last[i,j] - inv_current[i, j] + 1)/(1 - alpha_j[j]) + inv_last[i, j])
                else:
                    d_last[i, j] = inv_last[i, j] + w_last[i,j] - inv_current[i, j]
        return d_last

    "Round to positive numer"
    def trans2pos(self,param):
        param *= (param > 0)
        return param
    
    "Calculate the obj"    
    def objective(self, w):
        alpha_j = self.abandon_rate
        d_ijt = self.d_ijt
        I_ijt = self.invFilter(False)
        IR_it = self.invFilter(True)
        dR_it = self.dR_it
        p = self.cost_value
        q = 0.01
        c1 = np.zeros((1000, 5)).astype(float)
        c2 = [0.0]*1000
        c3 = c2
        for j in range(1,6):
            for i in self.sku_list:
                c1[i-1, j-1] = p[i-1]*np.ceil(np.multiply(alpha_j[j-1], self.trans2pos(d_ijt[i-1, j-1]-I_ijt[i-1, j-1])))
        for i in self.sku_list:
            c2[i-1] = np.multiply(p[i-1], self.trans2pos(dR_it[i-1] - IR_it[i-1] + np.sum(self.trans2pos(np.floor(np.multiply(1-alpha_j, self.trans2pos(d_ijt[i-1, :]-I_ijt[i-1, :])))) - w[i-1,:])))
            c3[i-1] = np.multiply(q, min(self.trans2pos(IR_it[i-1]-dR_it[i-1]), np.sum(self.trans2pos(np.floor(np.multiply(1-alpha_j, self.trans2pos(d_ijt[i-1, :]-I_ijt[i-1, :]))) - w[i-1,:]))))
        self.cnt += 1
        return np.sum(c1) + np.sum(c2) + np.sum(c3)      

    # ====================================================================
    "Calculate LHS: The FDC inventory must be nonnegative"
    def cons_FDC_inventory(self, w, i, FDC, t):
        alpha_j = self.abandon_rate[FDC-1]
        d_ijt = self.d_ijt[i-1, FDC-1]
        I_ijt = self.invFilter(False)[i-1, FDC-1]
        if I_ijt - d_ijt >= 0:
            I_ijt_plus1 = I_ijt - d_ijt + w[i-1,(FDC-1)]
        else:
            term2 = w[i-1,(FDC-1)] - np.floor((1-alpha_j)*(d_ijt-I_ijt))
            I_ijt_plus1 = self.trans2pos(term2)
            if term2 >= 0:
                I_ijt_plus1 = term2
            else:
                I_ijt_plus1 = 0
        return -I_ijt_plus1

    "Calculate LHS: The RDC inventory must be nonnegative"
    def cons_RDC_inventory(self, w, i, t):
        IR_it = self.invFilter(True)[i-1]
        dR_it = self.dR_it[i-1]
        alpha_j = self.abandon_rate
        d_ijt = self.d_ijt[i-1]
        I_ijt = self.invFilter(False)[i-1]
        try:
            r_it = self.replenish.loc[(self.replenish.item_sku_id == i) & (self.replenish.date == t)].replenish_quantity.iloc[0]
        except:
            r_it = 0
        IR_it_plus1 = self.trans2pos(IR_it - dR_it - np.sum(self.trans2pos(np.floor(np.multiply(1-alpha_j, self.trans2pos(d_ijt-I_ijt)))-w[(i-1),:]))) + r_it - np.sum(w[(i-1),:])
        return -IR_it_plus1

    "Calculate LHS: Limited type of sku can be scheduled to each FDC"
    # 5 inequality constraints, FDC = [1, 2, 3, 4, 5]
    def cons_FDC_skutype(self, w, FDC, t):
        int_w = 0
        int_w = sum(1 for i in self.sku_list if w[(i-1), (FDC-1)]>0)
        return int_w - self.sku_limit[FDC-1]

    "Calculate LHS: Maximum capacity of sku for each lane to FDC"
    # 5 inequality constraints
    def cons_capacity(self, w, FDC, t):
        return np.sum(w[:,(FDC-1)]) - self.capacity_limit[FDC-1]

    "Check if constraint is >0 return Ture"
    def check_constraints(self, w, t):
        for i in self.sku_list:
            if self.cons_RDC_inventory(w,i,t)>0:
                print ("constraint violated: RDC_inventory", i, t)
                return False
            for j in [1, 2, 3, 4, 5]:
                if self.cons_FDC_inventory(w,i,j,t)>0:
                    print ("constraint violated: FDC_inventory", i, j, t)
                    return False
        for j in [1, 2, 3, 4, 5]:
            if self.cons_FDC_skutype(w,j,t)>0:
                print ("constraint violated: FDC_skutype", j, t)
                return False
            if self.cons_capacity(w,j,t)>0:
                print ("constraint violated: capacity", j, t)
                return False           
        return True

    # ====================================================================
    "Update inventory"
    def info_update(self,end_day_inventory,t):
        '''
        input values: inventory information at the end of day t
        '''        
        self.inv.append(end_day_inventory)

    "Return last inventory"
    def invFilter(self, dc_type, inv_last = False):
        if not inv_last:
            if dc_type:
                return self.inv_array[:,0]
            else:
                return self.inv_array[:,1:6]
        else:
            if dc_type:
                return self.inv_array_last[:,0]
            else:
                return self.inv_array_last[:,1:6]

    "Multiple solution: select the best"    
    def sol_filter(self, feasible_solution):
        if self.show_obj:
            Final_cost = 200000
            for i in range(0, len(feasible_solution)):
                cost = self.objective(feasible_solution[i])
                if cost < Final_cost:
                    Final_cost = cost
                    Final_solution = feasible_solution[i]
            print ("Final cost:", Final_cost)
            return Final_solution
        else:
            return feasible_solution[0]

    # ====================================================================
    "Our decision policy"
    # ====================================================================
    "Calculate UB of w"
    def calUpperbound(self, t):
        w_up = np.empty((1000,5))
        if self.consider_demand:
            # alpha_j = self.abandon_rate
            # w_last = self.w_last
            # I_last = self.invFilter(False, True)
            # d_last = self.demand_infer(I_last, w_last, alpha_j)
            # for i in self.sku_list:
            #     try:
            #         r_it = self.replenish.loc[(self.replenish.item_sku_id == i) & (self.replenish.date == t-1)].replenish_quantity.iloc[0]
            #     except:
            #         r_it = 0
            #     IR_it = self.invFilter(True)[i-1]
            #     dR_it = self.dR_it[i-1]
            #     sku_up = self.trans2pos(IR_it - dR_it-np.sum(self.trans2pos(np.floor(np.multiply(1-alpha_j, self.trans2pos(d_last[i-1,:]-I_last[i-1,:])))-w_last[(i-1),:]))) + r_it
            #     w_up[i-1,:] = int(sku_up/5)
            if t >= 2:
                I_ijt = self.invFilter(False)
                d_ijt = self.d_ijt
                alpha_j = self.abandon_rate
                for i in self.sku_list:
                    try:
                        r_it = self.replenish.loc[(self.replenish.item_sku_id == i) & (self.replenish.date == t)].replenish_quantity.iloc[0]
                    except:
                        r_it = 0
                    IR_it = self.invFilter(True)[i-1]               
                    dR_it = self.dR_it[i-1]
                    w_last = self.w_last
                    sku_up = self.trans2pos(IR_it - dR_it-np.sum(self.trans2pos(np.floor(np.multiply(1-alpha_j, self.trans2pos(d_ijt[i-1,:]-I_ijt[i-1,:])))))) + r_it
                    w_up[i-1,:] = int(sku_up/5)
            else:
                I_ijt = self.invFilter(False)
                d_ijt = self.d_ijt
                alpha_j = self.abandon_rate
                for i in self.sku_list:
                    try:
                        r_it = self.replenish.loc[(self.replenish.item_sku_id == i) & (self.replenish.date == t)].replenish_quantity.iloc[0]
                    except:
                        r_it = 0
                    IR_it = self.invFilter(True)[i-1]               
                    dR_it = self.dR_it[i-1]
                    sku_up = self.trans2pos(IR_it - dR_it-np.sum(self.trans2pos(np.floor(np.multiply(1-alpha_j, self.trans2pos(d_ijt[i-1,:]-I_ijt[i-1,:])))))) + r_it
                    w_up[i-1,:] = int(sku_up/5)
        else:
            for i in self.sku_list:
                try:
                    r_it = self.replenish.loc[(self.replenish.item_sku_id == i) & (self.replenish.date == t-1)].replenish_quantity.iloc[0]
                except:
                    r_it = 0
                sku_up = r_it
                if sku_up < 0:
                    sku_up = 0
                w_up[i-1,:] = int(sku_up/5)
        return w_up

    "Our policy: Set w to UB"
    def sol_generate(self, w_up):
        w_temp = cp.copy(w_up)
        w_temp *= (w_temp > 0)
        return w_temp

    "Decrease w by FDC capacity"    
    def sol_adjust_FDC(self, w, w_up, FDC_demand_lo, FDC_demand_up):
        print ("Adjusting for FDC inventory...")
        dR_it_lo = np.floor(FDC_demand_lo*self.dR_it)
        d_ijt_lo = np.floor(FDC_demand_lo*self.d_ijt)
        dR_it_up = np.floor(FDC_demand_up*self.dR_it)
        d_ijt_up = np.floor(FDC_demand_up*self.d_ijt)
        # [dR_it_lo, d_ijt_lo] = self.demand_quantile(FDC_demand_lo)
        # [dR_it_up, d_ijt_up] = self.demand_quantile(FDC_demand_up)
        I_ijt = self.invFilter(False)
        
        for j in range(0,5):
            for i in range(0,1000):
                if I_ijt[i, j] >= d_ijt_up[i, j]:
                    w[i, j] = 0
                elif (I_ijt[i,j] <= d_ijt_lo[i, j] and self.cost_value[i] >= 0.50):
                    w[i, j] = min(d_ijt_lo[i, j], w_up[i,j])
        return w

    "Decrease w by sku type capacity"    
    def sol_adjust_skutype(self, w):
        print ("Adjusting for skutype limit...")
        for j in range(0,5):
            w_j_sorted = np.sort(w[:,j])[::-1]
            w_j_kth = w_j_sorted[self.sku_limit[j]]
            for i in range(0, 1000):
                if (w[i, j] <= w_j_kth and self.cost_value[i] <= 0.16):
                    w[i, j] = 0               
        return w

    "Decrease w by maximum capacity for each w_ijt"        
    def sol_adjust_capacity(self, w):
        print ("Adjusting for the maximum capacity...")
        # for j in range(0,5):
        #     cnt = 0
        #     while np.sum(w[:,j]) - self.capacity_limit[j] > 0:
        #         cnt += 1
        #         if cnt >= 50:
        #             break
        #         w_vio_gap = np.sum(w[:,j]) - self.capacity_limit[j]
        #         # print (w_vio_gap)
        #         w_j_nonzero = sum(i > 0 for i in w[:,j])
        #         w_j_reduce = int(w_vio_gap / w_j_nonzero)
        #         if w_j_reduce < 1:
        #             w_j_reduce = 1
        #         for i in range(0,1000):
        #             if w[i,j] >= w_j_reduce and self.cost_value[i] <= 0.50:
        #                 w[i,j] = w[i,j] - w_j_reduce
        #             elif w[i,j] >= int(w_j_reduce/2):
        #                 w[i,j] = max(0, (w[i,j] - int(w_j_reduce/2)))
        #         #    else:
        #         #        w[i,j] = 0
        for j in range(0,5):
            w_vio_gap = np.sum(w[:,j]) - self.capacity_limit[j]
            item_cnt = 0
            if w_vio_gap > 0:
                for i in self.cost_sorted_index.astype(int):
                    if w[i-1,j] > 0:
                        item_cnt += w[i-1,j]
                        if item_cnt > w_vio_gap:
                            w[i-1,j] = item_cnt - w_vio_gap
                            break
                        else:
                            w[i-1,j] = 0
        return w
            
    def daily_decision(self,t):
        '''
        daily decision of inventory allocation
        input values:
            t: decision date
        return values:
            allocation decision, 2-D numpy array, shape (5,1000), type integer
        '''
        
        # Your algorithms here
        # simple rule: no transshipment at all
        print ("Day:", t)
        transshipment_decision = np.zeros((5, 1000)).astype(int)
        df_inv = self.inv[t-1].sort_values(['item_sku_id','dc_id'])
        self.inv_array = np.asarray([df_inv.loc[df_inv.item_sku_id == sku_id].stock_quantity.values for sku_id in self.sku_list])
        if t >= 2:
            df_inv_last = self.inv[t-2].sort_values(['item_sku_id','dc_id'])
            self.inv_array_last = np.asarray([df_inv_last.loc[df_inv_last.item_sku_id == sku_id].stock_quantity.values for sku_id in self.sku_list])
        feasible_solution = []
        # warm start
        w_up = self.calUpperbound(t)
        # searching
        feas_cnt = 0
        for ite in range(0,1000):
            if len(feasible_solution) >= 1:
                print ("A feasible solution has been found!")
                break
        #    print ("Searching feasible solutions: iteration -", ite )
            print ("Searching ... ")
            w = self.sol_generate(w_up)
            w = self.sol_adjust_FDC(w, w_up, self.demand_percentage_lo, self.demand_percentage_up)
            w = self.sol_adjust_capacity(w)
            w = self.sol_adjust_skutype(w)
            if self.docheck:
                if self.check_constraints(w, t):
                    feas_cnt += 1
                    print ("Found a feasible solution:", feas_cnt)
                    feasible_solution.append(w)
            else:
            #    print ("Found a feasible solution.")
                feasible_solution.append(w)
        Daily_solution = self.sol_filter(feasible_solution)
        self.w_last = Daily_solution
        transshipment_decision = np.transpose(Daily_solution)
        return transshipment_decision

# ====================================================================
    "Dictionary for scipy"    
    def const(self,t):
        const_dict_F = tuple({'type':'ineq','fun': lambda w: self.cons_FDC_inventory(w,i,j,t)} for i in self.sku_list for j in [1, 2, 3, 4, 5])
        print ("1) const_dict_F: finished!")
        const_dict_R = tuple({'type':'ineq','fun': lambda w: self.cons_RDC_inventory(w,i,t)} for i in self.sku_list)
        print ("2) const_dict_R: finished!")
        const_FDC_dict = tuple({'type':'ineq','fun': lambda w: self.cons_FDC_skutype(w,j,t)} for j in [1, 2, 3, 4, 5])
        print ("3) const_FDC_dict: finished!")
        const_capacity = tuple({'type':'ineq','fun': lambda w: self.cons_capacity(w,j,t)} for j in [1, 2, 3, 4, 5])
        print ("4) const_capacity: finished!")
        const_total = const_dict_F + const_dict_R + const_FDC_dict + const_capacity
        return const_total
