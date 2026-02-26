import numpy as np
import matplotlib.pyplot as plt
import time as time_lib
import random
import os

from tqdm import tqdm

import multiprocessing

from scripts.dynamics import vecField
from scripts.utils import sample_ab_node_centered, uniformPoints, lobattoPoints
from scripts.parareal import parallel_solver

def run_experiment(args,return_nets=False,verbose=False,setup_only=False,n_x_override=None,L_override=None):
        
        if len(args) == 2:
                system, nodes = args
                ab_init = "uniform"
                a_min = None
                a_max = None
        elif len(args) == 3:
                system, nodes, ab_init = args
                a_min = None
                a_max = None
        elif len(args) == 5:
                system, nodes, ab_init, a_min, a_max = args
        else:
                raise ValueError("run_experiment expects 2, 3, or 5 arguments in args tuple/list.")
        if system=="BurgerQ" or system=="Burger":
                system="Burger"
                ic = "quadratic"
        elif system=="Burger1W":
                system="Burger"
                ic = "single_wave"
        elif system=="BurgerSW":
                system="Burger"
                ic = "sum_of_waves"
        
        vecRef = vecField(system)
        number_processors = 5 if system=="Rober" or system=="Burger" else 1
        
        if system=="BurgerQ" or system=="Burger" or system=="Burger1W" or system=="BurgerSW":
                n_x = 5#3
                L = 5#3
        else:
                n_x = 5
                L = 5
        if n_x_override is not None:
                n_x = int(n_x_override)
        if L_override is not None:
                L = int(L_override)
        else:
                L = n_x
        if n_x < 2 or L < 2:
                raise ValueError("n_x and L must be >= 2.")
        if ab_init == "centred" and n_x != L:
                raise ValueError("For centred init, n_x must equal L.")
        n_t = 2 #we do all the experiments with n_t = 2, which means we really just have coarse intervals
                #we do not further split them to simplify the problem. On the other hand, the code is flexible
                #also to this additional splitting.

        LB = -1.
        UB = 1.

        if system=="Rober":
                #t_max = 100.
                t_max = 10.
                num_t = 101
                #L = 5
                vecRef.dt_fine = 1e-4
        elif system=="SIR":
                t_max = 100.
                num_t = 101
                #L = 3
                vecRef.dt_fine = 1e-2
        elif system=="Brusselator":
                t_max = 12.
                num_t = 33
                #L = 5
                vecRef.dt_fine = t_max/640
        elif system=="Arenstorf":
                t_max = 17 #17.0652165601579625588917206249 #One period
                num_t = 126
                #L = 3
                vecRef.dt_fine = t_max / 80000
        elif system=="Lorenz":
                t_max = 10.
                num_t = 251
                #L = 3
                vecRef.dt_fine = t_max / 14500
        elif system=="Burger":
                t_max = 1.
                num_t = 51
                #L = 3
                vecRef.dt_fine = t_max / 500
        else:
                print("Dynamics not implemented")
        
        if system=="Rober":
                time = np.concatenate((np.linspace(0,1,101)[:-1],np.linspace(1,t_max,34)))
        else:
                time = np.linspace(0,t_max,num_t)
                
        dts = np.diff(time)
        
        if system=="Rober":
                y0 = np.array([1.,0.,0])
        elif system=="SIR":
                y0 = np.array([0.3,0.5,0.2])
        elif system=="Brusselator":
                y0 = np.array([0.,1.])
        elif system=="Arenstorf":
                y0 = np.array([0.994,0,0.,-2.00158510637908252240537862224])
        elif system=="Lorenz":
                y0 = np.array([20.,5,-5])
        elif system=="Burger":
                if ic=="quadratic":
                        y0 = vecRef.x*(1.-vecRef.x)
                elif ic=="single_wave":
                        y0 = np.sin(2*np.pi*vecRef.x)
                else:
                        y0 = np.sin(2*np.pi*vecRef.x) + np.cos(4*np.pi*vecRef.x) - np.cos(8*np.pi*vecRef.x) 
                
        else:
                print("Dynamics not implemented")
                

        if ab_init == "centred":
                if a_min is None:
                        a_min = abs(LB)
                if a_max is None:
                        a_max = abs(UB)
                if nodes == "uniform":
                        tau = uniformPoints(n_x)
                elif nodes == "lobatto":
                        tau = lobattoPoints(n_x)
                else:
                        raise ValueError(f"Unsupported nodes='{nodes}' for centred initialization.")
                weight, bias = sample_ab_node_centered(
                        tau=tau,
                        a_min=a_min,
                        a_max=a_max,
                        jitter=0.1,
                )
        else:
                weight = np.random.uniform(low=LB,high=UB,size=(L))
                bias = np.random.uniform(low=LB,high=UB,size=(L))

        data = {"vecRef":vecRef,
                "LB" : LB,
                "time" : time,
                "dts" : dts,
                "UB" : UB,
                "L" : L,
                "y0" : y0,
                "n_x" : n_x,
                "n_t" : n_t,
                "num_t" : num_t,
                "system" : system,
                "nodes" : nodes,
                "act_name" : "tanh",
                "number_processors":number_processors,
                "t_max":t_max,
                "lsq_skip_tol":1e-10,
                "use_warm_start_gate":True,
                "warm_start_activation_tol":1e-2,
                "use_active_prefix":True,
                "use_selective_retrain":False,
                "selective_retrain_tol":1e-4,
                "selective_retrain_state_max_norm":1e6,
                "weight":weight,
                "bias":bias}
        
        if setup_only:
                return data,time,dts,vecRef,number_processors
                
        coarse_approx,networks,total_time,_,avg_coarse_step = parallel_solver(time,data,dts,vecRef,number_processors,verbose=verbose)

        if return_nets:
                return total_time,avg_coarse_step,coarse_approx,networks,data
        else:
                return total_time, avg_coarse_step
