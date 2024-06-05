import torch
from torch.func import vmap, jacfwd
from scripts.dynamics import *
from scripts.plotting import *

def trainModel(model,n_train,y0,dt,max_t,bounds,vec,epochs,device,dtype,optimizer):

    #isSymplectic(model,device,dtype) #check if the model is symplectic
    #isVolumePreserving(model,device,dtype) #check if the model is volume preserving
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=int(0.4*epochs),gamma=0.1)

    Loss_history = []
    
    lb,ub = bounds
    d = model.d
    epoch = 1
        
    for epoch in range(epochs):
        
        optimizer.zero_grad()    
        
        t = torch.rand((n_train,1),dtype=dtype)*dt*1.2
        z = torch.rand((n_train,d))*(ub-lb)+lb
        
        z /= z.sum(dim=1,keepdims=True)
        z += torch.rand_like(z)*2e-3-1e-3
        
        z,t = z.to(device),t.to(device)
        #z[-20:,1] *= 0 #so also equilibria are considered 
        z = z.to(device);
                                
        derivative = lambda x,t : vmap(jacfwd(model,argnums=1))(x,t)
        z_d = derivative(z,t).squeeze()            
        z = model(z,t)    
        Loss = vec.residualLoss(z,z_d)
                    
        Loss.backward(retain_graph=False);
        optimizer.step();
        scheduler.step();
            
        Loss_history.append(Loss.item())
        if epoch%2000==0:
            print(f"Epoch {epoch}, Loss {Loss.item()}")

        best = 1.
        if epoch>0.8*epochs:
            if Loss.item()<best:
                path = f"trainedModels/best.pt"
                torch.save(model.state_dict(), path)
                best = Loss.item()
        epoch+=1
    
    return Loss.item()