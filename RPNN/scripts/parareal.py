import multiprocessing
import numpy as np
import random
import time as time_lib

from scripts.ode_solvers import solver
from scripts.utils import flowMap

def fine_integrator(ics,dts,vecRef,number_processors):
    if number_processors==1:
        output = np.zeros_like(ics)
        args = [(args,vecRef) for args in zip(ics,dts)]
        for i in range(len(dts)):
            output[i] = solver(args[i])
        return output
    else:
        with multiprocessing.Pool(processes=number_processors) as pool:
            output = pool.map(solver,[(args,vecRef) for args in zip(ics,dts)])
        return np.array(output)


def getCoarse(time,data,previous=[],networks=[]):
    
    LB = data["LB"]
    UB = data["UB"]
    L = data["L"]
    y0 = data["y0"]
    weight = data["weight"]
    bias = data["bias"]
    n_x = data["n_x"]
    n_t = data["n_t"]
    system = data["system"]
    y0 = data["y0"]
    
    dts = np.diff(time)
    
    coarse_approx = np.zeros((len(time),len(y0)))
    coarse_approx[0] = y0
    
    initial_proj = np.kron(y0,np.ones(L))
    
    if len(previous)==0:
        for i in range(len(time)-1):
            flow = flowMap(y0=coarse_approx[i],initial_proj=initial_proj,weight=weight,bias=bias,dt=dts[i],n_t=n_t,n_x=n_x,L=L,LB=LB,UB=UB,system=system,act_name="Tanh")
            flow.approximate_flow_map()
            coarse_approx[i+1] = flow.sol[-1]
            networks.append(flow)

    else:
        for i in range(len(time)-1):
            flow = flowMap(y0=previous[i],initial_proj=initial_proj,weight=weight,bias=bias,dt=dts[i],n_t=n_t,n_x=n_x,L=L,LB=LB,UB=UB,system=system,act_name="Tanh")
            if len(networks)>0:
                flow.computed_projection_matrices = networks[i].computed_projection_matrices.copy()
            flow.approximate_flow_map()
            coarse_approx[i+1] = flow.sol[-1]
            networks[i] = flow
            
    return coarse_approx, networks
    
def getNextCoarse(time,y,i,data,networks=[]):
    
    dts = np.diff(time)
    
    LB = data["LB"]
    UB = data["UB"]
    L = data["L"]
    weight = data["weight"]
    bias = data["bias"]
    n_x = data["n_x"]
    n_t = data["n_t"]
    system = data["system"]
    y0 = data["y0"]
    
    initial_proj = np.kron(y0,np.ones(L)).reshape(1,-1)
    
    flow = flowMap(y0=y,initial_proj=initial_proj,weight=weight,bias=bias,dt=dts[i],n_t=n_t,n_x=n_x,L=L,LB=LB,UB=UB,system=system,act_name="Tanh")
    if len(networks)>0:
        flow.computed_projection_matrices = networks[i].computed_projection_matrices.copy()
        flow.y0 = y
    flow.approximate_flow_map()
    networks[i] = flow
    return flow.analyticalApproximateSolution(dts[i]), networks

def parallel_solver(time,data,dts,vecRef,number_processors,verbose=False):
    
    np.random.seed(17)
    random.seed(17)
    
    max_it = 20 #maximum number of parareal iterates
    tol = 1e-4
    computational_times_per_iterate = []
    it = 0
    is_converged = False
    networks = []
    
    number_processors = min(number_processors,multiprocessing.cpu_count())

    initial_full = time_lib.time()
    
    while it<max_it and is_converged==False:

        norm_difference = []
        
        if it==0:
            initial_time = time_lib.time()
            coarse_approx, networks = getCoarse(previous=[],time=time,data=data,networks=networks)
            coarse_values_parareal = coarse_approx.copy()            
            cost = time_lib.time()-initial_time
            computational_times_per_iterate.append(cost)
            if verbose:
                print("Average cost per one coarse step : ",cost/len(dts))
        else:
            initial_time = time_lib.time()
            
            start_fine = time_lib.time()
            fine_int = fine_integrator(coarse_values_parareal,dts,vecRef,number_processors)
            if verbose:
                print("Time required for the fine solver : ",time_lib.time()-start_fine)
            for i in range(len(time)-1): 
                previous = coarse_values_parareal[i+1].copy()
                next,networks = getNextCoarse(y=coarse_values_parareal[i],i=i,time=time,data=data,networks=networks)
                coarse_values_parareal[i+1] = fine_int[i] + next - coarse_approx[i+1]
                coarse_approx[i+1] = next.copy()
                norm_difference.append(np.linalg.norm(coarse_values_parareal[i+1]-previous,2))
            
            if verbose:
                print("Difference norms : ",norm_difference)   
            computational_times_per_iterate.append(time_lib.time()-initial_time)
            if verbose:
                print("Maximum norm of difference :",np.round(np.max(norm_difference),10))
            is_converged = np.max(norm_difference)<tol
        it+=1
        if verbose:
            print(f"Iterate {it} completed")
            print(f"Time for iterate {it} is {computational_times_per_iterate[-1]}")
    
    total_time = time_lib.time()-initial_full

    return coarse_approx,networks,total_time,number_processors,cost/len(dts)