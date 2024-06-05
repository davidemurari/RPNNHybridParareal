from scripts.dynamics import *
import time as time_lib
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator as linOp
from scipy.sparse import diags

import random
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
    
class flowMap:
    def __init__(self,y0,initial_proj,weight,bias,dt=1,n_t=2,n_x=5,L=5,LB=-1.,UB=1.,system="Rober",act_name="tanh",nodes="uniform",verbose=False):
        
        self.system = system
        self.vec = vecField(system)
        self.act = lambda t,w,b : act(t,w,b,act_name=act_name)
        self.dt = dt
        self.d = len(y0) #dimension phase space
        
        self.verbose = verbose
        
        self.n_x = n_x
        self.h = np.zeros((n_x,L))
        self.hd = np.zeros((n_x,L))
        
        self.iter = 0
        
        self.y0 = y0
        self.y0_supp = y0
        
        self.L = L #number of neurons
        self.LB = LB #Lower boundary for weight and bias sampling 
        self.UB = UB #Upper boundary for weight and bias sampling
        
        if len(weight)==0:
            self.weight = np.random.uniform(low=self.LB,high=self.UB,size=(self.L))
        else:
            self.weight = weight
        if len(bias)==0:
            self.bias = np.random.uniform(low=self.LB,high=self.UB,size=(self.L))
        else:
            self.bias = bias
        
        self.n_t = n_t
        self.t_tot = np.linspace(0,dt,self.n_t)
        if nodes=="uniform":
            self.x = uniformPoints(self.n_x)
        elif nodes=="lobatto":
            self.x = lobattoPoints(self.n_x)

        for i in range(n_x):
            self.h[i], self.hd[i] = self.act(self.x[i],self.weight,self.bias)

        self.h0 = self.h[0] #at the initial time, i.e. at x=0.
        self.hd0 = self.hd[0]
        
        self.computational_time = None        
        self.computed_projection_matrices = np.tile(initial_proj,(self.n_t-1,1)) #one per time subintreval       
        self.computed_initial_conditions = np.zeros((self.n_t-1,self.d)) #one per time subinterval
        self.training_err_vec = np.zeros((self.n_t,1))
        self.sol = np.zeros((self.n_t,self.d))
    
    def to_mat(self,y,a,b):
        return y.reshape((a,b),order='F')
    
    def residual(self,c_i,xi_i):
        #if system=Burger we suppose xi_i to have only weights for internal nodes and the rest is set to 0
        if self.system=="Burger":
            zero = np.zeros((self.L,1))
            w = np.concatenate((zero,xi_i.reshape((self.L,self.d-2),order='F'),zero),axis=1).reshape(-1,order='F')
            y = (self.h-self.h0)@self.to_mat(w,self.L,self.d) + self.y0_supp.reshape(1,-1)
            y_dot = c_i * self.hd @ self.to_mat(w,self.L,self.d)
        else:
            y = (self.h-self.h0)@self.to_mat(xi_i,self.L,self.d) + self.y0_supp.reshape(1,-1)
            y_dot = c_i * self.hd @ self.to_mat(xi_i,self.L,self.d)
        
        vecValue = self.vec.eval(y)
        Loss = (y_dot - vecValue)
        if self.system=="Burger":
            Loss = Loss[:,1:-1]
        return Loss.reshape(-1,order='F')
    
    def re(self,a,H):
        return np.einsum('i,ij->ij',a,H)

    def jac_residual(self,c_i,xi_i):
        
        np.random.seed(17)
        random.seed(17)

        H = self.h - self.h0    
        weight = xi_i

        if self.system=="Burger":
            W = self.to_mat(weight,self.L,self.d-2)
            y = H@W + self.y0_supp[1:-1].reshape(1,-1)
        else:
            W = self.to_mat(weight,self.L,self.d)
            y = H@W + self.y0_supp.reshape(1,-1)
                
        if self.system=="Rober":
            #Verified
            y1,y2,y3 = y[:,0],y[:,1],y[:,2]
            k1,k2,k3 = self.vec.k1, self.vec.k2, self.vec.k3
            zz = np.zeros((self.n_x,self.L))
            row1 = np.concatenate((c_i*self.hd+k1*H,-k3*self.re(y3,H),-k3*self.re(y2,H)),axis=1)
            row2 = np.concatenate((-k1*H,c_i*self.hd+self.re(2*k2*y2+k3*y3,H),k3*self.re(y2,H)),axis=1)
            row3 = np.concatenate((zz,-2*k2*self.re(y2,H),c_i*self.hd),axis=1)
            return np.concatenate((row1,row2,row3),axis=0)
        
        elif self.system=="SIR":
            #Verified
            y1,y2,y3 = y[:,0],y[:,1],y[:,2]
            zz = np.zeros((self.n_x,self.L))
            beta,gamma,N = self.vec.beta, self.vec.gamma, self.vec.N
            row1 = np.concatenate((c_i*self.hd+beta*self.re(y2,H)/N,beta*self.re(y1,H)/N,zz),axis=1)
            row2 = np.concatenate((-beta*self.re(y2,H)/N,c_i*self.hd-beta*self.re(y1,H)/N+gamma*H,zz),axis=1)
            row3 = np.concatenate((zz,-gamma*H,c_i*self.hd),axis=1)
            return np.concatenate((row1,row2,row3),axis=0)
    
        elif self.system=="Brusselator":
            #Verified
            xx,yy = y[:,0],y[:,1]
            zz = np.zeros((self.n_x,self.L))
            A,B = self.vec.A, self.vec.B
            
            row1 = np.concatenate((c_i*self.hd-2*self.re(xx*yy,H)+(B+1)*H,-self.re(xx**2,H)),axis=1)
            row2 = np.concatenate((-B*H+2*self.re(xx*yy,H),c_i*self.hd+self.re(xx**2,H)),axis=1)
            return np.concatenate((row1,row2),axis=0)
    
        elif self.system=="Arenstorf":
            #Verified
            xx,xxp,yy,yyp = y[:,0],y[:,1],y[:,2],y[:,3]
            zz = np.zeros((self.n_x,self.L))
            a,b = self.vec.a,self.vec.b
            
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
            
            row1 = np.concatenate((c_i*self.hd,-H,zz,zz),axis=1)
            row2 = np.concatenate((-self.re(dxpp_dx,H),c_i*self.hd,-self.re(dxpp_dy,H),-2*H),axis=1)
            row3 = np.concatenate((zz,zz,c_i*self.hd,-H),axis=1)
            row4 = np.concatenate((-self.re(dypp_dx,H),2*H,-self.re(dypp_dy,H),c_i*self.hd),axis=1)
            return np.concatenate((row1,row2,row3,row4),axis=0)
        
        elif self.system=="Lorenz":
            #Verified
            y1,y2,y3 = y[:,0],y[:,1],y[:,2]
            zz = np.zeros((self.n_x,self.L))
            sigma,r,b = self.vec.sigma, self.vec.r, self.vec.b
            row1 = np.concatenate((c_i*self.hd+sigma*H,-sigma*H,zz),axis=1)
            row2 = np.concatenate((self.re(y3,H)-r*H,c_i*self.hd+H,self.re(y1,H)),axis=1)
            row3 = np.concatenate((-self.re(y2,H),-self.re(y1,H),c_i*self.hd+b*H),axis=1)
            return np.concatenate((row1,row2,row3),axis=0)
        
        elif self.system=="Burger":
            
            N = self.d     
            dx = 1/(N-1)
            vv = np.ones(N-1)
            Shift_forward = diags([vv], [1], shape=(N, N))#np.diag(vv,k=1)
            Shift_backward = diags([vv], [-1], shape=(N, N))#np.diag(vv,k=-1)

            Shift_backward = Shift_backward.tolil()
            Shift_backward[-1, -2] = 0  # Adjust for sparse matrix indexing
            Shift_forward = Shift_forward.tolil()
            Shift_forward[0, 1] = 0
            Shift_backward = Shift_backward.tocsr()
            Shift_forward = Shift_forward.tocsr()

            D2 = (Shift_forward + Shift_backward - 2*np.eye(N))/(dx**2)
            D1 = 1/(2*dx) * (Shift_forward-Shift_backward)
            
            D2 = D2[1:-1,1:-1]
            D1 = D1[1:-1,1:-1]
                                    
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
        
        np.random.seed(17)
        random.seed(17)
        
        self.training_err_vec[0] = 0.        
        self.sol[0] = self.y0_supp
        
        initial_time = time_lib.time()
        
        for i in range(self.n_t-1):
            
            self.iter = 1
            
            c_i = (self.x[-1]-self.x[0]) / (self.t_tot[i+1]-self.t_tot[i])
            xi_i = self.computed_projection_matrices[i] 
            self.computed_initial_conditions[i] = self.y0_supp
                
            if self.system=="Burger":
                func = lambda x : self.residual(c_i,x)
                initial_condition = xi_i[self.L:-self.L]
                jac = lambda x : self.jac_residual(c_i,x)
                xi_i = least_squares(func,x0=initial_condition,verbose=0,xtol=1e-5,gtol=1e-8,method='trf',jac=jac).x               
                self.computed_projection_matrices[i,self.L:-self.L] = xi_i
                Loss = func(self.computed_projection_matrices[i,self.L:-self.L])
            else:
                func = lambda x : self.residual(c_i,x)
                jac = lambda x : self.jac_residual(c_i,x)
                if self.system=="Rober":
                    self.computed_projection_matrices[i] = least_squares(func,x0=xi_i,verbose=0,xtol=1e-8,gtol=1e-8,method='lm',jac=jac).x
                else:
                    self.computed_projection_matrices[i] = least_squares(func,x0=xi_i,verbose=0,xtol=1e-5,method='lm',jac=jac).x
                Loss = func(self.computed_projection_matrices[i])
                
            y = (self.h-self.h0)@self.to_mat(self.computed_projection_matrices[i],self.L,self.d) + self.y0_supp.reshape(1,-1)
            self.y0_supp = y[-1]
            self.sol[i+1] = self.y0_supp
            self.training_err_vec[i+1] = np.sqrt(np.mean(Loss**2))
        final_time = time_lib.time()
        
        self.computational_time = final_time-initial_time
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