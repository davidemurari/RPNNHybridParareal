from scripts.dynamics import *
import time as time_lib
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator as linOp

import numpy as np


def act(t,w,b,act_name="Tanh"):
    if act_name=="Tanh":
        return np.tanh(w*t+b), (1-np.tanh(w*t+b)**2)*w #function at the node, derivative at the node    
    elif act_name=="Sigmoid":
        func = lambda x : 1/(1+np.exp(-x))
        func_p = lambda x : func(x)*(1-func(x))
        return func(w*t+b), func_p(w*t+b)*w
    else: #sin
        func = lambda x : np.sin(x)
        func_p = lambda x: np.cos(x)
        return func(w*t+b), func_p(w*t+b)*w

#https://mathworld.wolfram.com/LobattoQuadrature.html
def lobattoPoints(n):
    #Nodes in [-1,1]
    if n==3:
        nodes = np.array([-1,0.,1.])
    elif n==4:
        nodes = np.array([-1,-np.sqrt(5)/5,np.sqrt(5)/5,1.])
    elif n==5:
        nodes = np.array([-1,-np.sqrt(21)/7,0.,np.sqrt(21)/7,1.])
    #Transforming them into [0,1]
    return nodes/2+0.5

def uniformPoints(n):
    return np.linspace(0,1,n)
    
def sample_ab_node_centered(tau, a_min=5.0, a_max=10.0, jitter=0.01, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    C = len(tau)
    signs = rng.choice([-1.0, 1.0], size=C)
    mags  = rng.uniform(a_min, a_max, size=C)
    a = signs * mags

    # permutation = "shuffle assignment"; set perm = np.arange(C) for identity
    perm = rng.permutation(C)
    centers = tau[perm] + rng.uniform(-jitter, jitter, size=C)
    b = -a * centers
    return a, b

class flowMap:
    def __init__(self,y0,initial_proj,weight,bias,dt=1,t_start=0.,n_t=2,n_x=5,L=5,LB=-1.,UB=1.,system="Rober",act_name="tanh",nodes="uniform",verbose=False,vec=None,lsq_skip_tol=1e-10,profile_timing=True,enable_fast_lsq=True,lsq_fast_max_nfev=10,lsq_fast_rms_tol=1e-6):
        
        self.system = system
        self.vec = vec if vec is not None else vecField(system)
        self.act = lambda t,w,b : act(t,w,b,act_name=act_name)
        self.dt = dt
        self.t_start = t_start
        self.d = len(y0) #dimension phase space
        
        self.verbose = verbose
        self.lsq_skip_tol = lsq_skip_tol
        self.profile_timing = profile_timing
        self.enable_fast_lsq = enable_fast_lsq
        self.lsq_fast_max_nfev = lsq_fast_max_nfev
        self.lsq_fast_rms_tol = lsq_fast_rms_tol
        
        self.n_x = n_x
        self.h = np.zeros((n_x,L))
        self.hd = np.zeros((n_x,L))
        
        self.iter = 0
        
        self.y0 = y0
        self.y0_supp = y0
        
        self.L = L #number of neurons
        self.LB = LB #Lower boundary for weight and bias sampling 
        self.UB = UB #Upper boundary for weight and bias sampling
                
        self.n_t = n_t
        self.t_tot = np.linspace(0,dt,self.n_t)
        self.t_abs = np.linspace(self.t_start,self.t_start+dt,self.n_t)
        if nodes=="uniform":
            self.x = uniformPoints(self.n_x)
        elif nodes=="lobatto":
            self.x = lobattoPoints(self.n_x)
        
        
        if len(weight)==0:
            self.weight = np.random.uniform(low=self.LB,high=self.UB,size=(self.L))
        else:
            self.weight = weight
        if len(bias)==0:
            self.bias = np.random.uniform(low=self.LB,high=self.UB,size=(self.L))
        else:
            self.bias = bias

        x_eval = self.x.reshape(-1,1)
        w_eval = np.asarray(self.weight).reshape(1,-1)
        b_eval = np.asarray(self.bias).reshape(1,-1)
        self.h, self.hd = self.act(x_eval,w_eval,b_eval)

        self.h0 = self.h[0] #at the initial time, i.e. at x=0.
        self.hd0 = self.hd[0]
        self.H = self.h - self.h0
        if self.system == "Burger":
            # Workspace to avoid reallocating padded Burger coefficients in residual().
            self._burger_w_full = np.zeros(self.L * self.d)
        
        self.computational_time = None        
        self.computed_projection_matrices = np.tile(initial_proj,(self.n_t-1,1)) #one per time subintreval       
        self.computed_initial_conditions = np.zeros((self.n_t-1,self.d)) #one per time subinterval
        self.training_err_vec = np.zeros((self.n_t,1))
        self.sol = np.zeros((self.n_t,self.d))
        self.profile_last = {}
        self.profile_cumulative = {
            "train_calls": 0,
            "residual_calls": 0,
            "residual_time": 0.0,
            "jac_calls": 0,
            "jac_time": 0.0,
            "lsq_calls": 0,
            "lsq_fast_calls": 0,
            "lsq_full_calls": 0,
            "lsq_fallbacks": 0,
            "lsq_time": 0.0,
            "lsq_skips": 0,
            "total_time": 0.0,
        }
    
    def to_mat(self,y,a,b):
        return y.reshape((a,b),order='F')
    
    def residual(self,c_i,xi_i,t_nodes=None):
        #if system=Burger we suppose xi_i to have only weights for internal nodes and the rest is set to 0
        H = self.H
        if self.system=="Burger":
            w_full = self.to_mat(self._burger_w_full, self.L, self.d)
            w_full.fill(0.0)
            w_full[:, 1:-1] = self.to_mat(xi_i, self.L, self.d - 2)
            y = H @ w_full + self.y0_supp.reshape(1,-1)
            y_dot = c_i * (self.hd @ w_full)
        else:
            W = self.to_mat(xi_i, self.L, self.d)
            y = H @ W + self.y0_supp.reshape(1,-1)
            y_dot = c_i * (self.hd @ W)
        
        if t_nodes is None:
            vecValue = self.vec.eval(0.0,y)
        else:
            vecValue = self.vec.eval(t_nodes,y)
        Loss = (y_dot - vecValue)
        if self.system=="Burger":
            Loss = Loss[:,1:-1]
        return Loss.reshape(-1,order='F')
    
    def re(self,a,H):
        return np.einsum('i,ij->ij',a,H)

    def jac_residual(self,c_i,xi_i,t_nodes=None):
        H = self.H
        c_hd = c_i * self.hd
        weight = xi_i

        if self.system=="Burger":
            W = self.to_mat(weight,self.L,self.d-2)
            y = H@W + self.y0_supp[1:-1].reshape(1,-1)
        else:
            W = self.to_mat(weight,self.L,self.d)
            y = H@W + self.y0_supp.reshape(1,-1)

        def rs(i):
            return slice(i * self.n_x, (i + 1) * self.n_x)

        def cs(i):
            return slice(i * self.L, (i + 1) * self.L)
                
        if self.system=="Rober":
            y2,y3 = y[:,1],y[:,2]
            k1,k2,k3 = self.vec.k1, self.vec.k2, self.vec.k3
            J = np.zeros((self.n_x * self.d, self.L * self.d))
            y2H = y2[:, None] * H
            y3H = y3[:, None] * H
            J[rs(0), cs(0)] = c_hd + k1 * H
            J[rs(0), cs(1)] = -k3 * y3H
            J[rs(0), cs(2)] = -k3 * y2H
            J[rs(1), cs(0)] = -k1 * H
            J[rs(1), cs(1)] = c_hd + (2 * k2 * y2 + k3 * y3)[:, None] * H
            J[rs(1), cs(2)] = k3 * y2H
            J[rs(2), cs(1)] = -2 * k2 * y2H
            J[rs(2), cs(2)] = c_hd
            return J
        
        elif self.system=="SIR":
            y1,y2,y3 = y[:,0],y[:,1],y[:,2]
            beta,gamma,N = self.vec.beta, self.vec.gamma, self.vec.N
            J = np.zeros((self.n_x * self.d, self.L * self.d))
            by1H = (beta / N) * (y1[:, None] * H)
            by2H = (beta / N) * (y2[:, None] * H)
            J[rs(0), cs(0)] = c_hd + by2H
            J[rs(0), cs(1)] = by1H
            J[rs(1), cs(0)] = -by2H
            J[rs(1), cs(1)] = c_hd - by1H + gamma * H
            J[rs(2), cs(1)] = -gamma * H
            J[rs(2), cs(2)] = c_hd
            return J
    
        elif self.system=="Brusselator":
            xx,yy = y[:,0],y[:,1]
            A,B = self.vec.A, self.vec.B
            _ = A
            J = np.zeros((self.n_x * self.d, self.L * self.d))
            xyH = (xx * yy)[:, None] * H
            xx2H = (xx ** 2)[:, None] * H
            J[rs(0), cs(0)] = c_hd - 2 * xyH + (B + 1) * H
            J[rs(0), cs(1)] = -xx2H
            J[rs(1), cs(0)] = -B * H + 2 * xyH
            J[rs(1), cs(1)] = c_hd + xx2H
            return J
    
        elif self.system=="Arenstorf":
            xx,xxp,yy,yyp = y[:,0],y[:,1],y[:,2],y[:,3]
            a,b = self.vec.a,self.vec.b
            _ = (xxp, yyp)
            
            D1 = ((xx+a)**2+yy**2)**(3/2)
            D2 = ((xx-b)**2+yy**2)**(3/2)
            D1_dx = 3*np.sqrt(a**2+2*a*xx+xx**2+yy**2)*(xx+a)
            D2_dx = 3*np.sqrt(b**2-2*xx*b+xx**2+yy**2)*(xx-b)
            D1_dy = 3*np.sqrt(a**2+2*xx*a+xx**2+yy**2)*yy
            D2_dy = 3*np.sqrt(b**2-2*xx*b+xx**2+yy**2)*yy
           
            dxpp_dx = 1-b/D1+b*(xx+a)*D1_dx/D1**2 - a/D2 + a*(xx-b)*D2_dx/D2**2
            dxpp_dy = b*(xx+a)*D1_dy/D1**2 + a*(xx-b)*D2_dy/D2**2
            
            dypp_dx = yy*(b*D1_dx/D1**2+a*D2_dx/D2**2)
            dypp_dy = 1-b/D1+b*yy*D1_dy/D1**2-a/D2+a*yy*D2_dy/D2**2
            J = np.zeros((self.n_x * self.d, self.L * self.d))
            J[rs(0), cs(0)] = c_hd
            J[rs(0), cs(1)] = -H
            J[rs(1), cs(0)] = -(dxpp_dx[:, None] * H)
            J[rs(1), cs(1)] = c_hd
            J[rs(1), cs(2)] = -(dxpp_dy[:, None] * H)
            J[rs(1), cs(3)] = -2 * H
            J[rs(2), cs(2)] = c_hd
            J[rs(2), cs(3)] = -H
            J[rs(3), cs(0)] = -(dypp_dx[:, None] * H)
            J[rs(3), cs(1)] = 2 * H
            J[rs(3), cs(2)] = -(dypp_dy[:, None] * H)
            J[rs(3), cs(3)] = c_hd
            return J
        
        elif self.system=="Lorenz":
            y1,y2,y3 = y[:,0],y[:,1],y[:,2]
            sigma,r,b = self.vec.sigma, self.vec.r, self.vec.b
            J = np.zeros((self.n_x * self.d, self.L * self.d))
            J[rs(0), cs(0)] = c_hd + sigma * H
            J[rs(0), cs(1)] = -sigma * H
            J[rs(1), cs(0)] = y3[:, None] * H - r * H
            J[rs(1), cs(1)] = c_hd + H
            J[rs(1), cs(2)] = y1[:, None] * H
            J[rs(2), cs(0)] = -(y2[:, None] * H)
            J[rs(2), cs(1)] = -(y1[:, None] * H)
            J[rs(2), cs(2)] = c_hd + b * H
            return J
        
        elif self.system=="Duffing":
            # x' = v
            # v' = -delta*v - alpha*x - beta*x^3 + gamma*cos(omega*t)
            xx,vv = y[:,0],y[:,1]
            delta = self.vec.delta
            alpha = self.vec.alpha
            beta = self.vec.beta
            _ = vv
            J = np.zeros((self.n_x * self.d, self.L * self.d))
            J[rs(0), cs(0)] = c_hd
            J[rs(0), cs(1)] = -H
            J[rs(1), cs(0)] = (alpha + 3 * beta * (xx ** 2))[:, None] * H
            J[rs(1), cs(1)] = c_hd + delta * H
            return J
        
        elif self.system=="Burger":
            
            D2 = self.vec.D2[1:-1,1:-1]
            D1 = self.vec.D1[1:-1,1:-1]
                                    
            def vec(Y):
                return Y.reshape((-1),order='F')
            
            def mv(v):
                V = self.to_mat(v,self.L,self.d-2)
                hV = H@V
                Mat_Version = c_i*self.hd@V-(-(y@D1.T)*hV-y*(H@(V@D1.T))+self.vec.nu*H@(V@D2.T))
                res = vec(Mat_Version)
                return res
            def rmv(v):
                V = self.to_mat(v,self.n_x,self.d-2)
                Mat_Version = c_i*self.hd.T@V-H.T@[-(y@D1.T)*V - (y*V)@D1 + self.vec.nu*V@D2]
                return vec(Mat_Version)
            
            shape = (self.n_x*(self.d-2),(self.d-2)*self.L)
            return linOp(shape,matvec=mv,rmatvec=rmv)

        else:
            pass
    
    def approximate_flow_map(self):
        self.training_err_vec[0] = 0.        
        self.sol[0] = self.y0_supp
        
        initial_time = time_lib.perf_counter()
        profile = {
            "residual_calls": 0,
            "residual_time": 0.0,
            "jac_calls": 0,
            "jac_time": 0.0,
            "lsq_calls": 0,
            "lsq_fast_calls": 0,
            "lsq_full_calls": 0,
            "lsq_fallbacks": 0,
            "lsq_time": 0.0,
            "lsq_skips": 0,
        }

        def do_lsq(x0, method, jac_fun, kwargs):
            # Fast first pass: capped work budget, then fallback to full solve only if needed.
            if self.enable_fast_lsq and self.lsq_fast_max_nfev is not None and self.lsq_fast_max_nfev > 0:
                lsq_t0 = time_lib.perf_counter()
                try:
                    x_fast = least_squares(
                        timed_residual,
                        x0=x0,
                        method=method,
                        jac=jac_fun,
                        max_nfev=self.lsq_fast_max_nfev,
                        **kwargs,
                    ).x
                except ValueError:
                    x_fast = x0
                if self.profile_timing:
                    profile["lsq_calls"] += 1
                    profile["lsq_fast_calls"] += 1
                    profile["lsq_time"] += time_lib.perf_counter() - lsq_t0

                loss_fast = timed_residual(x_fast)
                if np.all(np.isfinite(loss_fast)):
                    loss_fast_rms = np.sqrt(np.mean(loss_fast**2))
                    if loss_fast_rms <= self.lsq_fast_rms_tol:
                        return x_fast, loss_fast
                if self.profile_timing:
                    profile["lsq_fallbacks"] += 1
                x0 = x_fast

            lsq_t0 = time_lib.perf_counter()
            try:
                x_full = least_squares(
                    timed_residual,
                    x0=x0,
                    method=method,
                    jac=jac_fun,
                    **kwargs,
                ).x
            except ValueError:
                x_full = x0
            if self.profile_timing:
                profile["lsq_calls"] += 1
                profile["lsq_full_calls"] += 1
                profile["lsq_time"] += time_lib.perf_counter() - lsq_t0
            return x_full, timed_residual(x_full)
        
        for i in range(self.n_t-1):
            
            self.iter = 1
            
            c_i = (self.x[-1]-self.x[0]) / (self.t_tot[i+1]-self.t_tot[i])
            t_nodes = np.linspace(self.t_abs[i],self.t_abs[i+1],self.n_x)
            xi_i = self.computed_projection_matrices[i] 
            self.computed_initial_conditions[i] = self.y0_supp
            
            def timed_residual(x):
                t0 = time_lib.perf_counter()
                out = self.residual(c_i,x,t_nodes=t_nodes)
                if self.profile_timing:
                    profile["residual_calls"] += 1
                    profile["residual_time"] += time_lib.perf_counter() - t0
                return out

            def timed_jac(x):
                t0 = time_lib.perf_counter()
                out = self.jac_residual(c_i,x,t_nodes=t_nodes)
                if self.profile_timing:
                    profile["jac_calls"] += 1
                    profile["jac_time"] += time_lib.perf_counter() - t0
                return out
                
            if self.system=="Burger":
                func = timed_residual
                initial_condition = xi_i[self.L:-self.L]
                initial_condition = np.nan_to_num(initial_condition, nan=0.0, posinf=0.0, neginf=0.0)
                jac = timed_jac
                loss0 = func(initial_condition)
                if np.all(np.isfinite(loss0)):
                    loss0_rms = np.sqrt(np.mean(loss0**2))
                    if self.lsq_skip_tol > 0 and loss0_rms <= self.lsq_skip_tol:
                        Loss = loss0
                        if self.profile_timing:
                            profile["lsq_skips"] += 1
                    else:
                        xi_i, Loss = do_lsq(
                            x0=initial_condition,
                            method="trf",
                            jac_fun=jac,
                            kwargs={"verbose": 0, "xtol": 1e-5, "gtol": 1e-8},
                        )
                        self.computed_projection_matrices[i,self.L:-self.L] = xi_i
                else:
                    Loss = np.nan_to_num(loss0, nan=1e12, posinf=1e12, neginf=-1e12)
                if not np.all(np.isfinite(Loss)):
                    Loss = np.nan_to_num(Loss, nan=1e12, posinf=1e12, neginf=-1e12)
            else:
                func = timed_residual
                jac = timed_jac
                xi_i = np.nan_to_num(xi_i, nan=0.0, posinf=0.0, neginf=0.0)
                loss0 = func(xi_i)
                if np.all(np.isfinite(loss0)):
                    loss0_rms = np.sqrt(np.mean(loss0**2))
                    if self.lsq_skip_tol > 0 and loss0_rms <= self.lsq_skip_tol:
                        Loss = loss0
                        if self.profile_timing:
                            profile["lsq_skips"] += 1
                    else:
                        if self.system=="Rober":
                            xi_new, Loss = do_lsq(
                                x0=xi_i,
                                method="lm",
                                jac_fun=jac,
                                kwargs={"verbose": 0, "xtol": 1e-8, "gtol": 1e-8},
                            )
                        else:
                            xi_new, Loss = do_lsq(
                                x0=xi_i,
                                method="lm",
                                jac_fun=jac,
                                kwargs={"verbose": 0, "xtol": 1e-5},
                            )
                        self.computed_projection_matrices[i] = xi_new
                else:
                    Loss = np.nan_to_num(loss0, nan=1e12, posinf=1e12, neginf=-1e12)
                if not np.all(np.isfinite(Loss)):
                    Loss = np.nan_to_num(Loss, nan=1e12, posinf=1e12, neginf=-1e12)
                
            y = (self.h-self.h0)@self.to_mat(self.computed_projection_matrices[i],self.L,self.d) + self.y0_supp.reshape(1,-1)
            self.y0_supp = y[-1]
            self.sol[i+1] = self.y0_supp
            self.training_err_vec[i+1] = np.sqrt(np.mean(Loss**2))
        final_time = time_lib.perf_counter()
        
        self.computational_time = final_time-initial_time
        self.profile_last = {
            **profile,
            "total_time": self.computational_time,
            "n_slabs": self.n_t - 1,
        }
        self.profile_cumulative["train_calls"] += 1
        self.profile_cumulative["total_time"] += self.computational_time
        for k in ("residual_calls","residual_time","jac_calls","jac_time","lsq_calls","lsq_fast_calls","lsq_full_calls","lsq_fallbacks","lsq_time","lsq_skips"):
            self.profile_cumulative[k] += self.profile_last[k]
        if self.verbose:
            print(f"Training complete. Required time {self.computational_time}")
    
    def analyticalApproximateSolution(self,t):
        
        j = np.searchsorted(self.t_tot,t,side='left') #determines the index of the largest number in t_tot that is smaller than t
        #In other words, it finds where to place t in t_tot in order to preserve its increasing ordering
        j = j if j>0 else 1 #so if t=0 we still place it after the first 0.
        
        y_0 = self.sol[j-1]        
        x = np.array([(t - self.t_tot[j-1]) / (self.t_tot[j]-self.t_tot[j-1])])
        h,_ = act(x,self.weight,self.bias)
        h0,_ = act(0*x,self.weight,self.bias)
        y = self.to_mat(self.computed_projection_matrices[j-1],self.L,self.d).T@(h-h0) + y_0
        return y
    
    def plotOverTimeRange(self,time):
        sol_approximation = np.zeros((self.d,len(time)))
        for i,t in enumerate(time):
            sol_approximation[:,i] = self.analyticalApproximateSolution(t)
        return sol_approximation
