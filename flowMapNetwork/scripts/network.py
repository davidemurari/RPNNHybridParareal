import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scripts.dynamics import vecField

#Custom activation functions   
class sinAct(nn.Module):
        def __init__(self):
            super().__init__()  
        def forward(self,x):
            return torch.sin(x) 

class swishAct(nn.Module):
        def __init__(self):
            super().__init__()  
        def forward(self,x):
            return torch.sigmoid(x)*x
        
class integralTanh(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,x):
            return torch.log(torch.cosh(x)) #this is the primitive of tanh(x)

class network(nn.Module):
        def __init__(self,neurons=100,dt=1.,act_name='tanh',d=4,nlayers=3,dtype=torch.float,system="harmonic-oscillator"):
            super().__init__()
            
            ##The outputs of this network are to be intended as the variables
            ##q1,q2,pi1,pi2, i.e. the physical configuration variables together
            ##with the non-conservative momenta.
            
            torch.manual_seed(1)
            np.random.seed(1)
            
            self.dtype = dtype
            self.vec = vecField(system=system)
            self.dt = dt
            
            self.seq = nn.Sequential(
                nn.Linear(1,10,bias=True,dtype=dtype),
                sinAct(),
                nn.Linear(10,4,bias=False,dtype=dtype)
            )
            
            self.seq[0].weight.data = torch.ones_like(self.seq[0].weight.data,dtype=dtype)
            vv = torch.randint(0,2,size=self.seq[0].bias.data.shape,dtype=dtype)*2-1 #so -1 or 1
            self.seq[0].bias.data = vv * torch.pi/2
            
            activations = {
                "tanh": nn.Tanh(),
                "sin": sinAct(),
                "sigmoid": nn.Sigmoid(),
                "swish": swishAct(),
                "intTanh":integralTanh()
            }
            
            self.act = activations[act_name]
            
            self.H = neurons
            self.d = d
            self.nlayers = nlayers
            
            
            self.lift = nn.Linear(self.d+1,self.H,dtype=dtype)
            self.linears = nn.ModuleList([nn.Linear(self.H,self.H,dtype=dtype) for _ in range(self.nlayers)])
            self.proj = nn.Linear(self.H,self.d,dtype=dtype)
            
            self.f = lambda t: 1-torch.exp(-t)
        
        def parametric(self,y0,t):
            input = torch.cat((y0 / 5,t),dim=1)
            H = self.act(self.lift(input))

            for i in range(self.nlayers):
                H = self.act(self.linears[i](H))
            res = self.proj(H)
            return res     
        
        def plotOverTimeRange(self,y0,time_plot,data):
            device = data["device"]
            dtype = data["dtype"]
            if dtype==torch.float32:
                yy0 = torch.from_numpy(y0.astype(np.float32)).to(device).unsqueeze(0).repeat(len(time_plot),1)
                time = torch.from_numpy(time_plot.astype(np.float32)).to(device)
            else:
                yy0 = torch.from_numpy(y0).to(device).unsqueeze(0).repeat(len(time_plot),1)
                time = torch.from_numpy(time_plot).to(device)
            
            preds = self.forward(yy0,time).detach().cpu().numpy().T
            return preds
            
        
        def forward(self,y0,t):
            
            y0 = y0.reshape(-1,self.d)
            t = t.reshape(-1,1)
            
            t0 = torch.zeros_like(t)
            res = self.parametric(y0,t)
            res = y0 + self.f(t)*res
            return res