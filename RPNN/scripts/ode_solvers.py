import numpy as np
from scipy.optimize import least_squares

def RK4_step(x0,t0,dt,vecRef):
    k1 = vecRef.eval(t0,x0)
    k2 = vecRef.eval(t0+dt/2,x0+k1*dt/2)
    k3 = vecRef.eval(t0+dt/2,x0+k2*dt/2)
    k4 = vecRef.eval(t0+dt,x0+k3*dt)
    return x0 + 1/6*dt*(k1+2*k2+2*k3+k4)

def solver(args,final=True):
        
    args,vecRef = args[0],args[1]
    
    t_eval = []
    if len(args)==2:
        u0,tf=args
        t0 = 0.0
    elif len(args)==3:
        u0,tf,t_eval=args
        if len(t_eval) > 0:
            t0 = float(np.asarray(t_eval)[0])
            tf = float(np.asarray(t_eval)[-1])
        else:
            t0 = 0.0
    elif len(args)==4:
        u0,t0,tf,t_eval=args
    else:
        raise ValueError("solver expects args=(u0,tf), (u0,tf,t_eval), or (u0,t0,tf,t_eval).")
        
    if len(t_eval)==0:
        if tf <= t0:
            if final:
                return np.asarray(u0, dtype=float)
            return np.asarray([u0], dtype=float), np.asarray([t0], dtype=float)
        n_steps = int(np.ceil((tf-t0)/vecRef.dt_fine + 1))
        n_steps = max(2, n_steps)
        time = np.linspace(t0,tf,n_steps)
    else:
        time = np.asarray(t_eval, dtype=float)
        if len(time) > 0:
            t0 = float(time[0])
            tf = float(time[-1])

    if len(time) < 2:
        if tf <= t0:
            if final:
                return np.asarray(u0, dtype=float)
            return np.asarray([u0], dtype=float), np.asarray([t0], dtype=float)
        time = np.array([t0, tf], dtype=float)

    sol = np.zeros((len(time),len(u0)))
    sol[0] = u0
    
    if vecRef.system=="Rober" or vecRef.system=="Burger":
        for i in range(len(time)-1):
            h = time[i+1]-time[i]
            t_next = time[i+1]
            objective = lambda u: (u - sol[i] - h*vecRef.eval(t_next,u))
            euler_guess = sol[i]+h*vecRef.eval(time[i],sol[i])
            if vecRef.system=="Rober":
                sol[i+1] = least_squares(objective,x0=euler_guess,method='trf',xtol=1e-5).x
            else:
                sol[i+1] = least_squares(objective,x0=euler_guess,method='lm',xtol=1e-5).x
    else:
        for i in range(len(time)-1):
            h = time[i+1]-time[i]
            sol[i+1] = RK4_step(sol[i],time[i],h,vecRef)
    
    if final:
        return sol[-1]
    else:
        return sol,time
