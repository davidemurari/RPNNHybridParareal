import numpy as np
import matplotlib.pyplot as plt
import torch
import time as time_lib
from tqdm import tqdm
import pickle

from scipy.integrate import solve_ivp
from scripts.network import network
from scripts.training import trainModel
from scripts.dynamics import vecField,vecField_np
from scripts.parareal import parallel_solver
from scripts.ode_solvers import solver
from scripts.plotting import plot_results

torch.manual_seed(1)
np.random.seed(1)
dtype=torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

system = "SIR"

vecRef = vecField_np(system=system)

if system=="SIR":
        beta,gamma,N = vecRef.beta,vecRef.gamma,vecRef.N
        t_max = 100.
        num_t = 101
        L = 3
        n_t = 2
        vecRef.dt_fine = 1e-2
elif system=="Brusselator":
        A,B = vecRef.A,vecRef.B 
        t_max = 12.
        num_t = 25
        L = 3
        n_t = 5
        vecRef.dt_fine = t_max/640
else:
        print("Dynamics not implemented")
        
if system=="SIR":
        y0 = np.array([0.3,0.5,0.2])
        d = 3
elif system=="Brusselator":
        y0 = np.array([0.,1.])
        d = 2
else:
        print("Dynamics not implemented")

#Domain details
t0 = 0. #initial time

file_name = input("Write the name of the .pkl file where the results are saved\n")

#To generate results in the paper : file_name = trained_model_SIR_20240424_161900
with open(f'trainedModels/{file_name}.pkl', 'rb') as f:
    params = pickle.load(f)
    
dt = params["dt"]
nlayers = params["nlayers"]
hidden_nodes = params["hidden_nodes"]
act = params["act"]
dim_t = params["dim_t"]
device = params["device"]
file_name = params["file_name"]

model = network(neurons=hidden_nodes,d=d,dt=dt,act_name=act,nlayers=nlayers,dtype=dtype,system=system) #If you want a different network
model.to(device);

model.load_state_dict(torch.load(f"trainedModels/{file_name}.pt",map_location=device))
model.eval();

data = {"y0" : y0,
        "system" : system,
        "device":device,
        "dtype":dtype}

number_points = int(t_max / dt)
time = np.linspace(0,100,number_points+1)
dts = np.diff(time)

avg_total_cost = 0.
avg_cost_first_iterate = 0.

N = int(input("How many runs of the hybrid Parareal method do you want to do?\n"))

number_processors = 1

for i in tqdm(range(N)):
    coarse_values_parareal,networks, total_time, number_processors, overhead_costs, cost_first_it = parallel_solver(time,data,dts,vecRef,number_processors,model,verbose=False)
    avg_total_cost += total_time
    avg_cost_first_iterate += cost_first_it

avg_total_cost /= N
avg_cost_first_iterate /= N
print("Average total cost hybrid Parareal algorithm : ",avg_total_cost)
print("Average cost zeroth iterate : ",avg_cost_first_iterate)

def get_detailed_solution():
    num_steps = np.rint((dts[0] / vecRef.dt_fine)).astype(int)
    time_plot = np.linspace(0,dts[0],num_steps+1)
    sol = model.plotOverTimeRange(y0,time_plot,data)
    total_time = time_plot
    for i in np.arange(len(dts)):
        num_steps = int((dts[i] / vecRef.dt_fine))
        time_plot = np.linspace(0,dts[i],num_steps)[1:]
        sol = np.concatenate((sol,model.plotOverTimeRange(coarse_values_parareal[i+1],time_plot,data)),axis=1)
        total_time = np.concatenate((total_time,time_plot+total_time[-1]),axis=0)
    return sol,total_time

network_sol, time_plot = get_detailed_solution()

np.savetxt(f"network_sol_{system}_flow.txt",network_sol)

initial = time_lib.time()
arg = [[y0,time_plot[-1],time_plot],vecRef]
output,time_plot_sequential = solver(arg,final=False)
#output, _ = dop853(funcptr, y0, t_eval=time_plot, rtol=1e-11, atol=1e-10)
final = time_lib.time()
print(f"Computational time sequential: {final-initial}")
print(f"Average computational time parallel with {number_processors} processors: {avg_total_cost}")

if len(y0)==2:
        list_of_labels = [r"${x}_1$",r"${x}_2$"]
elif len(y0)==3:
        list_of_labels = [r"${x}_1$",r"${x}_2$",r"${x}_3$"]
elif len(y0)==4:
        list_of_labels = [r"${x}_1$",r"${x}_1'$",r"${x}_2$",r"${x}_2'$"]
else:
        list_of_labels = []
plot_results(y0,system,time_plot,time_plot_sequential,output,network_sol,list_of_labels,avg_total_cost,time)

