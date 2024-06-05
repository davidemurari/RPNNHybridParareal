import numpy as np
from scipy.optimize import least_squares

def RK4_step(x0,dt,vecRef):
    k1 = vecRef.eval(x0)
    k2 = vecRef.eval(x0+k1*dt/2)
    k3 = vecRef.eval(x0+k2*dt/2)
    k4 = vecRef.eval(x0+k3*dt)
    return x0 + 1/6*dt*(k1+2*k2+2*k3+k4)

def solver(args,final=True):
        
    args,vecRef = args[0],args[1]
    
    t_eval = []
    if len(args)==2:
        u0,tf=args
    if len(args)==3:
        u0,tf,t_eval=args
        
    t0 = 0.
        
    if len(t_eval)==0:
        n_steps = int(tf/vecRef.dt_fine + 1)
        time = np.linspace(t0,tf,n_steps)
    else:
        time = t_eval

    h = time[1]-time[0]
    sol = np.zeros((len(time),len(u0)))
    sol[0] = u0
    
    if vecRef.system=="Rober" or vecRef.system=="Burger":
        for i in range(len(time)-1):
            objective = lambda u: (u - sol[i] - h*vecRef.eval(u))
            euler_guess = sol[i]+h*vecRef.eval(sol[i])
            if vecRef.system=="Rober":
                sol[i+1] = least_squares(objective,x0=euler_guess,method='trf',xtol=1e-5).x
            else:
                sol[i+1] = least_squares(objective,x0=euler_guess,method='lm',xtol=1e-5).x
    else:
        for i in range(len(time)-1):
            sol[i+1] = RK4_step(sol[i],h,vecRef)
    
    if final:
        return sol[-1]
    else:
        return sol,time