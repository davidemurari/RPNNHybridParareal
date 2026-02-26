import numpy as np
import matplotlib.pyplot as plt
import time as time_lib
import random
import os
import argparse
import csv

from tqdm import tqdm
import datetime
import os
import multiprocessing

from scripts.dynamics import vecField
from scripts.plotting import plot_results
from scripts.ode_solvers import solver
from scripts.parareal import parallel_solver
from scripts.repeated_experiments import run_experiment

from joblib import Parallel, delayed,parallel_config
from tqdm import tqdm  

if __name__ == '__main__':
        
        system_names = ["SIR","Lorenz","Brusselator","Arenstorf","BurgerQ","Burger1W","BurgerSW"]

        parser = argparse.ArgumentParser(description="Run repeated experiments for selected dynamical systems.")
        parser.add_argument(
                "--system_name",
                default="all",
                choices=system_names + ["all", "all_but_burgers"],
                help="System to run: one of the listed systems, 'all', or 'all_but_burgers'.",
        )
        parser.add_argument(
                "--ab_init",
                choices=["uniform", "centred"],
                default="uniform",
                help="Initialization for a,b: 'uniform' or 'centred'.",
        )
        parser.add_argument(
                "--a_min",
                type=float,
                default=1.0,
                help="Lower bound for |a_i| when using centred AB init.",
        )
        parser.add_argument(
                "--a_max",
                type=float,
                default=1.0,
                help="Upper bound for |a_i| when using centred AB init.",
        )
        args = parser.parse_args()
        
        if args.a_min > args.a_max:
                parser.error("--a_min must be <= --a_max.")
        
        ab_folder = "centred" if args.ab_init == "centred" else "uniform"
        
        if args.system_name == "all":
                selected_systems = system_names
        elif args.system_name == "all_but_burgers":
                selected_systems = [s for s in system_names if not s.startswith("Burger")]
        else:
                selected_systems = [args.system_name]
        
        cwd = os.getcwd()
        os.chdir(cwd+"/RPNN")
        print("Current working directory: ",os.getcwd())
        
        number_iterates = 100 #int(input("How many repeated experiments do you want to perform?\n")) 
        
        for system in selected_systems:

                if not os.path.exists("savedReports/"):
                        os.mkdir("savedReports")
                reports_dir = os.path.join("savedReports", ab_folder)
                if not os.path.exists(reports_dir):
                        os.mkdir(reports_dir)

                btype = None
                        
                if system=="Lorenz":
                        nodes_list = ["uniform","lobatto"]
                else:
                        nodes_list = ["uniform"]
                
                for nodes in nodes_list:

                        print(f"\n\n Generating experiments for {system} and {nodes} collocation points\n\n")

                        with parallel_config(backend="loky", inner_max_num_threads=2):
                                output = Parallel(n_jobs=2)(
                                        delayed(run_experiment)((system,nodes,args.ab_init,args.a_min,args.a_max))
                                        for _ in tqdm(range(number_iterates))
                                )

                        total_time, avg_coarse_step = list(zip(*output))
                        total_time = np.array(total_time)
                        avg_coarse_step = np.array(avg_coarse_step)

                        average_cost = np.mean(total_time)
                        average_coarse_steps = np.mean(avg_coarse_step)
                        print("\n\n")
                        print(f"The average cost over {number_iterates} of the hybrid parareal is {average_coarse_steps}")

                        if system=="BurgerQ":
                                name_file = os.path.join(reports_dir, "report_Burger_quadratic.txt")
                        elif system=="Burger1W":
                                name_file = os.path.join(reports_dir, "report_Burger_one_wave.txt")
                        elif system=="BurgerSW":
                                name_file = os.path.join(reports_dir, "report_Burger_sum_waves.txt")
                        else:
                                if nodes!="uniform":
                                        name_file = os.path.join(reports_dir, f"report_{system}_{nodes}.txt")
                                else:
                                        name_file = os.path.join(reports_dir, f"report_{system}.txt")
                                
                        with open(name_file, "w") as file1:
                                # Writing data to a file
                                file1.write('=========================================================================\n')
                                file1.write('New experiment run at: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()))
                                file1.write('\n=========================================================================\n')
                                file1.write(f"The average cost over {number_iterates} of the hybrid parareal is {average_cost}\n")
                                file1.write(f"The average cost coarse over {number_iterates} of the hybrid parareal is {average_coarse_steps}\n")

                        _,_,coarse_approx,networks,data = run_experiment(
                                [system,nodes,args.ab_init,args.a_min,args.a_max],
                                return_nets=True,
                                verbose=False
                        )

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

                        initial = time_lib.time()
                        arg = [[y0,time_plot[-1],time_plot],vecRef]
                        output,time_plot_sequential = solver(arg,final=False)
                        final = time_lib.time()
                        print(f"Computational time sequential: {final-initial}")
                        print(f"Computational time parallel with {number_processors} processors: {average_cost}")

                        with open(name_file, "a") as file1:
                                # Writing data to a file
                                file1.write(f"Experiment with n_x={n_x}, H={L}, nodes={nodes}\n")
                                file1.write(f"Computational time sequential: {final-initial}\n")
                                file1.write(f"Computational time parallel with {number_processors} processors: {average_cost}\n")

                        csv_file = os.path.splitext(name_file)[0] + ".csv"
                        with open(csv_file, "w", newline="") as file_csv:
                                writer = csv.writer(file_csv)
                                writer.writerow([
                                        "timestamp",
                                        "system",
                                        "nodes",
                                        "ab_init",
                                        "a_min",
                                        "a_max",
                                        "num_runs",
                                        "avg_parallel_time",
                                        "avg_coarse_step_time",
                                        "n_x",
                                        "H",
                                        "sequential_time",
                                        "parallel_time_reported",
                                        "number_processors",
                                ])
                                writer.writerow([
                                        datetime.datetime.now().isoformat(timespec="seconds"),
                                        system,
                                        nodes,
                                        args.ab_init,
                                        args.a_min,
                                        args.a_max,
                                        number_iterates,
                                        average_cost,
                                        average_coarse_steps,
                                        n_x,
                                        L,
                                        final - initial,
                                        average_cost,
                                        number_processors,
                                ])

                        if len(y0)==2:
                                list_of_labels = [r"${x}_1$",r"${x}_2$"]
                        elif len(y0)==3:
                                list_of_labels = [r"${x}_1$",r"${x}_2$",r"${x}_3$"]
                        elif len(y0)==4:
                                list_of_labels = [r"${x}_1$",r"${x}_1'$",r"${x}_2$",r"${x}_2'$"]
                        else:
                                list_of_labels = []
                        
                        node_type = nodes if nodes!="uniform" else None
