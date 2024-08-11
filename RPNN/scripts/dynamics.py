import numpy as np

class vecField:
    def __init__(self,system="Rober"):
        self.system = system 
                
        if self.system=="Rober":
            self.k1 = 0.04
            self.k2 = 3e7
            self.k3 = 1e4
            self.dt_fine = 0.
        elif self.system=="SIR":
            self.beta = 0.1
            self.gamma = 0.1
            self.N = 1.
            self.dt_fine = 0.
        elif self.system=="Brusselator":
            self.A = 1.
            self.B = 3.
            self.dt_fine = 0.
        elif self.system=="Arenstorf":
            self.a = 0.012277471
            self.b = 1.-self.a
            self.dt_fine = 0.
        elif self.system=="Lorenz":
            self.sigma = 10.
            self.r = 28.
            self.b = 8/3
            self.dt_fine = 0.
        elif self.system=="Burger":
            self.nu = 1/50
            self.L = 1.
            self.N = 51
            self.x = np.linspace(0,self.L,self.N)
            self.dx = self.x[1]-self.x[0]
            self.dt_fine = 0.
        else:
            print("This dynamics is not implemented.")
            
    
    def eval(self,y):
        if self.system=="Rober":            
            if len(y.shape)==2:
                y1,y2,y3 = y[:,0:1],y[:,1:2],y[:,2:3]
                return np.concatenate([
                        -self.k1*y1 + self.k3*y2*y3,
                        self.k1*y1 - self.k2*(y2**2) - self.k3*y2*y3,
                        self.k2*(y2**2)
                    ],axis=1)
            else:
                y1,y2,y3 = y[0],y[1],y[2]
                return np.array([
                    -self.k1*y1 + self.k3*y2*y3,
                    self.k1*y1 - self.k2*(y2**2) - self.k3*y2*y3,
                    self.k2*(y2**2)
                ])
        elif self.system=="SIR":
            if len(y.shape)==2:
                y1,y2,y3 = y[:,0:1],y[:,1:2],y[:,2:3]
                return np.concatenate((
                    -self.beta*y2*y1/self.N,
                    self.beta*y1*y2/self.N - self.gamma*y2,
                    self.gamma*y2
                ),axis=1)
            else:
                y1,y2,y3 = y[0],y[1],y[2]
                return np.array([
                    -self.beta*y2*y1/self.N,
                    self.beta*y1*y2/self.N - self.gamma*y2,
                    self.gamma*y2
                ])
        
        elif self.system=="Brusselator":
            if len(y.shape)==2:
                xx,yy = y[:,0:1],y[:,1:2]
                return np.concatenate((
                    self.A+xx**2*yy - (self.B+1)*xx,
                    self.B*xx-xx**2*yy
                ),axis=1)
            else:
                xx,yy = y[0],y[1]
                return np.array([
                    self.A+xx**2*yy - (self.B+1)*xx,
                    self.B*xx-xx**2*yy
                ])
        
        elif self.system=="Arenstorf":
            if len(y.shape)==2:
                xx,xxp,yy,yyp = y[:,0:1],y[:,1:2],y[:,2:3],y[:,3:4]
                
                D1 = ((xx+self.a)**2+yy**2)**(3/2)
                D2 = ((xx-self.b)**2+yy**2)**(3/2)
                
                return np.concatenate((
                    xxp,
                    xx+2*yyp-self.b*(xx+self.a)/D1-self.a*(xx-self.b)/D2,
                    yyp,
                    yy-2*xxp-self.b*yy/D1-self.a*yy/D2
                ),axis=1)
            else:
                xx,xxp,yy,yyp = y[0],y[1],y[2],y[3]
                D1 = ((xx+self.a)**2+yy**2)**(3/2)
                D2 = ((xx-self.b)**2+yy**2)**(3/2)
                
                return np.array([
                    xxp,
                    xx+2*yyp-self.b*(xx+self.a)/D1-self.a*(xx-self.b)/D2,
                    yyp,
                    yy-2*xxp-self.b*yy/D1-self.a*yy/D2
                ])
        elif self.system=="Lorenz":
            if len(y.shape)==2:
                xx,yy,zz = y[:,0:1],y[:,1:2],y[:,2:3]
                
                return np.concatenate((
                    -self.sigma*xx+self.sigma*yy,
                    -xx*zz+self.r*xx-yy,
                    xx*yy-self.b*zz
                ),axis=1)
            else:
                xx,yy,zz = y[0],y[1],y[2]
                return np.array([
                    -self.sigma*xx+self.sigma*yy,
                    -xx*zz+self.r*xx-yy,
                    xx*yy-self.b*zz
                ])
        
        elif self.system=="Burger":
            if len(y.shape)==2:    
                
                N = self.N
                dx = self.dx
                vv = np.ones(N-1)
                Shift_forward = np.diag(vv,k=1)
                Shift_backward = np.diag(vv,k=-1)
                #For the boundary conditions
                Shift_backward[-1]*=0
                Shift_forward[0]*=0
                D2 = (Shift_forward + Shift_backward - 2*np.eye(N))/(dx**2)
                D1 = 1/(2*dx) * (Shift_forward-Shift_backward)
                
                vec = -y * (y@D1.T) + self.nu * (y@D2.T)
                return vec
            else:
                N = self.N
                dx = self.dx
                vv = np.ones(N-1)
                Shift_forward = np.diag(vv,k=1)
                Shift_backward = np.diag(vv,k=-1)
                #For the boundary conditions
                Shift_backward[-1]*=0
                Shift_forward[0]*=0
                D2 = (Shift_forward + Shift_backward - 2*np.eye(N))/(dx**2)
                D1 = 1/(2*dx) * (Shift_forward-Shift_backward)
                
                vec = -y * (D1@y) + self.nu * (D2@y)
                return vec
                
        
        else:
            pass