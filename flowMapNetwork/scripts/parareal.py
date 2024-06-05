import multiprocessing
import numpy as np
import torch
import time as time_lib

from scripts.ode_solvers import solver
from numba import jit   

def fine_integrator(ics,dts,vecRef,number_processors):
    if number_processors==1:
        output = np.zeros_like(ics)
        time = np.zeros_like(dts)
        args = [(args,vecRef) for args in zip(ics,dts)]
        for i in range(len(dts)):
            output[i],time[i] = solver(args[i])
        
        return output,np.sum(time)
    else:
        with multiprocessing.Pool(processes=number_processors) as pool:
            output = pool.map(solver,[(args,vecRef) for args in zip(ics,dts)])
        
        usol,time = zip(*output)
        
        return np.array(usol), np.sum(np.array(time))

def getCoarse(time,data,coarsePropagator,previous=[]):
    
    #coarsePropagator is the trained neural network
    
    y0 = data["y0"]
    device = data["device"]
    dtype = data["dtype"]
    
    dts = np.diff(time)
    
    coarse_approx = np.zeros((len(time),len(y0)))
    coarse_approx[0] = y0
        
    if len(previous)==0:
        for i in range(len(dts)):
            if dtype==torch.float32:
                previous = torch.from_numpy(coarse_approx[i:i+1].astype(np.float32)).to(device)
            else:
                previous = torch.from_numpy(coarse_approx[i:i+1]).to(device)
            dtt = torch.tensor([[dts[i]]],dtype=dtype).to(device)
            coarse_approx[i+1] = coarsePropagator(previous,dtt)[0].detach().cpu().numpy()

    else:
        if dtype==torch.float32:
            previous = torch.from_numpy(previous.astype(np.float32)).to(device).unsqueeze(0)
            dtt = torch.from_numpy(dts.astype(np.float32)).to(device)
        else:
            previous = torch.from_numpy(previous).to(device).unsqueeze(0)
            dtt = torch.from_numpy(dts).to(device)
        coarse_approx[1:] = coarsePropagator(previous,dtt).detach().cpu().numpy()
        
        
    return coarse_approx
    
def getNextCoarse(i,time,data,coarsePropagator,previous):
    
    dts = np.diff(time)
    
    dtype = data["dtype"]
    device = data["device"]
    
    if dtype==torch.float32:
        previous = torch.from_numpy(previous.astype(np.float32)).to(device).unsqueeze(0)
    else:
        previous = torch.from_numpy(previous).to(device).unsqueeze(0)
    dtt = torch.tensor([[dts[i]]],dtype=dtype).to(device)
    
    return coarsePropagator(previous,dtt)[0].detach().cpu().numpy()
    
    
def parallel_solver(time,data,dts,vecRef,number_processors,coarsePropagator,verbose=False):
    max_it = 20 #maximum number of parareal iterates
    tol = 1e-4
    computational_times_per_iterate = []
    overhead_costs = []
    it = 0
    is_converged = False
    networks = []
    
    number_processors = min(number_processors,multiprocessing.cpu_count())

    initial_full = time_lib.time()
        
    norm_differences = np.zeros((len(time)))
    
    while it<max_it and is_converged==False:
        
        if it==0:
            initial_time = time_lib.time()
            coarse_values_parareal = getCoarse(previous=[],time=time,data=data,coarsePropagator=coarsePropagator)
            coarse_approx = coarse_values_parareal.copy()
            cost_first_iterate = time_lib.time()-initial_time
            computational_times_per_iterate.append(cost_first_iterate)
            if verbose:
                print("Average cost first iterate : ",cost_first_iterate/len(dts))
        else:
            initial_time = time_lib.time()
            start_fine = time_lib.time()
            fine_int,fine_cost = fine_integrator(coarse_values_parareal,dts,vecRef,number_processors)
            time_fine = time_lib.time()-start_fine
            overhead_costs.append(time_fine-fine_cost)
            if verbose:
                print(f"\n\nTime required for the fine solver at iterate {it+1}: ",time_fine,"\n\n")
                print(f"Solution time for the fine integrator : {fine_cost}")
                print(f"Overhead cost : {time_fine-fine_cost}\n\n")
                
            for i in range(len(time)-1): 
                previous = coarse_values_parareal[i+1].copy()
                pp = time_lib.time()
                
                next = getNextCoarse(previous=coarse_values_parareal[i],i=i,time=time,data=data,coarsePropagator=coarsePropagator)
                coarse_values_parareal[i+1] = fine_int[i] + next - coarse_approx[i+1] 
                coarse_approx[i+1] = next
                norm_differences[i+1] = np.linalg.norm(coarse_values_parareal[i+1]-previous,ord=2)
            if verbose:
                print("Norms differences : ",norm_differences)   
            computational_times_per_iterate.append(time_lib.time()-initial_time)
            if verbose:
                print("Maximum value of difference :",np.round(np.max(norm_differences),10))
            is_converged = np.max(norm_differences)<tol
        it+=1
        if verbose:
            print(f"\n\nTime for iterate {it} is {computational_times_per_iterate[-1]}")
            if it>1:
                print(f"Time for coarse and parareal iteration is hence {computational_times_per_iterate[-1]-time_fine}\n\n")
    
    total_time = time_lib.time()-initial_full

    return coarse_values_parareal,networks,total_time,number_processors,overhead_costs,cost_first_iterate/len(dts)