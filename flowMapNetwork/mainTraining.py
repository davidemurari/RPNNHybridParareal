import numpy as np
import matplotlib.pyplot as plt
import torch
import time as time_lib
import os

from scripts.network import network
from scripts.training import trainModel
from scripts.dynamics import vecField,vecField_np

torch.manual_seed(1)
np.random.seed(1)
dtype=torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

system = "SIR"

vecRef = vecField(system=system)

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
elif system=="Brusselator":
        y0 = np.array([0.,1.])
else:
        print("Dynamics not implemented")
        
#Domain details
t0 = 0.
dt = 1.
if system=="SIR":
    lb = 0.
    ub = 1.
    d = 3
elif system=="Brusselator":
    lb = 0.
    ub = 5.
    d = 2

n_train, epochs = 1000, int(1e5)
vec = vecField(system,d)

lr = 5e-3

nlayers = 4
hidden_nodes = 10
act = "tanh"
dim_t = hidden_nodes
device = 'cpu'
bounds = [lb,ub]

model = network(neurons=hidden_nodes,d=d,dt=dt,act_name=act,nlayers=nlayers,dtype=dtype,system=system) #If you want a different network
model.to(device);

tt = time_lib.time()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 
file_name = f"trained_model_{system}_{timestamp}"

path = f"trainedModels/{file_name}.pt"
Loss = trainModel(model,n_train,y0,dt,t_max,bounds,vec,epochs,device,dtype,optimizer)
if os.path.exists("trainedModels/"):
    torch.save(model.state_dict(), path)
else:
    os.mkdir("trainedModels")
    torch.save(model.state_dict(), path)
    
model.eval();

training_time = time_lib.time() - tt

params = {
    "dt":dt,
    "n_train":n_train,
    "epochs":epochs,
    "nlayers":nlayers,
    "lr":lr,
    "act":act,
    "hidden_nodes":hidden_nodes,
    "dim_t":dim_t,
    "device":device,
    "file_name":file_name,
    "training_time":training_time
}

import pickle 

with open(f'trainedModels/{file_name}.pkl', 'wb') as f:
    pickle.dump(params, f)
    
print(f"The required training time was : {training_time}")