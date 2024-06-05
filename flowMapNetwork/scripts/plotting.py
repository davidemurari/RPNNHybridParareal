import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.dynamics import *
import time as time_lib
import os

import matplotlib

#Setting the plotting parameters
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['font.size']= 45
matplotlib.rcParams['font.family']= 'ptm' #'Times New Roman

def testPlot(vec,y0,tf,model,dtype,device):

    if dtype==torch.float32:
        yy0 = torch.from_numpy(y0.astype(np.float32)).to(device)
        
    y0_np = y0.copy()
    t_eval = np.linspace(0,tf,int(tf*10+1)) 

    sol_network = approximate_solution(yy0,model,t_eval,dtype,vec,device).detach().cpu().numpy().T
    sol_scipy = solution_scipy(y0_np,t_eval=t_eval,vec=vec)

    if vec.system=="SIR":
        labels = ["x","y","z"]
    elif vec.system=="Brusselator":
        labels = ["x","y"]
    else:
        print("Dynamics not implemented")

    fig = plt.figure(figsize=(20,10))

    for it in range(len(y0)):
        plt.plot(t_eval,sol_scipy[it],'-',label=f'{labels[it]} reference', linewidth=5) 
        plt.plot(t_eval,sol_network[it],'--',label=f'{labels[it]} parareal', linewidth=5)

    plt.xlabel("Time")
    plt.ylabel("Solution")
    plt.legend()
    plt.savefig
    plt.show();
    
def plot_results(y0,system,time_plot,time_plot_sequential,output,network_sol,list_of_labels,total_time,time):

    fig = plt.figure(figsize=(20,10))
    
    for i in range(len(y0)):
        if i==0:
            plt.plot(time_plot_sequential,output[:,i],'-',label=f"{list_of_labels[i]} reference", linewidth=5)
            plt.plot(time_plot,network_sol[i],'--',label=f"{list_of_labels[i]} parareal", linewidth=5)
        else:
            plt.plot(time_plot_sequential,output[:,i],'-',label=f"{list_of_labels[i]} reference", linewidth=5)
            plt.plot(time_plot,network_sol[i],'--',label=f"{list_of_labels[i]} parareal", linewidth=5)
  
    plt.legend(loc='center',bbox_to_anchor=(1.17, 0.5))
    plt.xlabel(r"$t$")
    plt.title(f"Comparison of solutions\n Computational time = {np.round(total_time,2)}s")
    
    if os.path.exists("savedPlots/"):
        plt.savefig(f"savedPlots/FLOW_pararealPlot_{system}.pdf",bbox_inches='tight')
    else:
        os.mkdir("savedPlots")
        plt.savefig(f"savedPlots/FLOW_pararealPlot_{system}.pdf",bbox_inches='tight')
    plt.show();