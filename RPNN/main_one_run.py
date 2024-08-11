import numpy as np
import matplotlib.pyplot as plt
import time as time_lib
import random
import os
import multiprocessing
import datetime

from scripts.dynamics import vecField
from scripts.plotting import plot_results
from scripts.ode_solvers import solver
from scripts.parareal import parallel_solver
from scripts.repeated_experiments import run_experiment

if __name__ == '__main__':
        
        cwd = os.getcwd()
        os.chdir(cwd+"/RPNN")
        print("Current working directory: ",os.getcwd())
        
        system = input("Which among the following systems do you want to consider?\n .\
        Write one among the following: SIR, Lorenz, Brusselator, Arenstorf, Rober, BurgerQ, Burger1W, BurgerSW\n .\
                BurgerQ stands for Burgers with Quadratic initial condition,\n .\
                Burger1W stands for Burgers with a sinusoidal initial condition,\n .\
                BurgerSW stands for Burgers with a sum of waves as initial condition.\n")

        print(f"You chose SYSTEM = {system}")

        nodes = "uniform"

        if not os.path.exists("savedReports/"):
                os.mkdir("savedReports")

        btype = None
                
        btype = None
        network_sol_lobatto = []
        network_sol_SIR_flow = []
        avg_time_SIR_flow = None
        avg_time_lobatto=None
        
        
        computational_time = time_lib.time()
        _,_,coarse_approx,networks,data = run_experiment([system,nodes],return_nets=True,verbose=False)
        computational_time = time_lib.time() - computational_time
        
        if system=="Burger" or system=="BurgerQ":
                btype = "quadratic"
        elif system=="Burger1W":
                btype = "single_wave"
        elif system=="BurgerSW":
                btype = "sum_of_waves"

        system = data["system"]
        LB = data["LB"]
        UB = data["UB"]
        L = data["L"]
        y0 = data["y0"]
        n_x = data["n_x"]
        n_t = data["n_t"]
        system = data["system"]
        nodes = data["nodes"]
        act_name = data["act_name"]
        t_max = data["t_max"]
        vecRef = data["vecRef"]
        num_t = data["num_t"]
        number_processors = data["number_processors"]
        time = data["time"]

        def get_detailed_solution():
                num_steps = np.rint(networks[0].dt / vecRef.dt_fine).astype(int)
                time_plot = np.linspace(0.,networks[0].dt,num_steps+1)
                sol = networks[0].plotOverTimeRange(time_plot)
                total_time = time_plot
                for i in np.arange(1,len(networks)):
                        num_steps = np.rint(networks[i].dt / vecRef.dt_fine).astype(int)
                        time_plot = np.linspace(0.,networks[i].dt,num_steps+1)[1:]
                        sol = np.concatenate((sol,networks[i].plotOverTimeRange(time_plot)),axis=1)
                        total_time = np.concatenate((total_time,time_plot+total_time[-1]),axis=0)
                return sol,total_time

        network_sol, time_plot = get_detailed_solution()

        num_fine_steps_per_coarse = ( t_max / (num_t-1) ) / vecRef.dt_fine
        num_fine_steps = t_max / vecRef.dt_fine

        arg = [[y0,time_plot[-1],time_plot],vecRef]
        output,time_plot_sequential = solver(arg,final=False)

        if len(y0)==2:
                list_of_labels = [r"${x}_1$",r"${x}_2$"]
        elif len(y0)==3:
                list_of_labels = [r"${x}_1$",r"${x}_2$",r"${x}_3$"]
        elif len(y0)==4:
                list_of_labels = [r"${x}_1$",r"${x}_1'$",r"${x}_2$",r"${x}_2'$"]
        else:
                list_of_labels = []

        if system=="Lorenz":
                nodes = "lobatto"
                avg_time_lobatto = time_lib.time()
                _,_,coarse_approx_lob,networks_lob,data_lob = run_experiment([system,nodes],return_nets=True,verbose=False)
                avg_time_lobatto = time_lib.time() - avg_time_lobatto
                def get_detailed_solution_lob():
                        num_steps = np.rint(networks_lob[0].dt / vecRef.dt_fine).astype(int)
                        time_plot = np.linspace(0.,networks_lob[0].dt,num_steps+1)
                        sol = networks_lob[0].plotOverTimeRange(time_plot)
                        total_time = time_plot
                        for i in np.arange(1,len(networks)):
                                num_steps = np.rint(networks[i].dt / vecRef.dt_fine).astype(int)
                                time_plot = np.linspace(0.,networks_lob[i].dt,num_steps+1)[1:]
                                sol = np.concatenate((sol,networks_lob[i].plotOverTimeRange(time_plot)),axis=1)
                                total_time = np.concatenate((total_time,time_plot+total_time[-1]),axis=0)
                        return sol,total_time

                network_sol_lobatto, time_plot = get_detailed_solution_lob()

                num_fine_steps_per_coarse = ( t_max / (num_t-1) ) / vecRef.dt_fine
                num_fine_steps = t_max / vecRef.dt_fine

        comp_time = -1

        plot_results(system=system,
                     time_plot=time_plot,
                     time_plot_sequential=time_plot_sequential,
                     output=output,
                     network_sol=network_sol,
                     list_of_labels=list_of_labels,
                     n_x=n_x,L=L,vec=vecRef,
                     number_iterates=1,
                     btype=btype,
                     avg_time = computational_time,
                     network_sol_lobatto=network_sol_lobatto,
                     avg_time_lobatto=avg_time_lobatto,
                     network_sol_SIR_flow=network_sol_SIR_flow,
                     avg_time_SIR_flow=avg_time_SIR_flow)
