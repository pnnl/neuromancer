
import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap

from torch.func import grad
from torch.func import vmap
from torch.func import jacrev
from torch.func import hessian



class ProxObjectivePlusEqualityConstraint(torch.nn.Module):
    '''
    Computes an approximation of the prox operator
    prox_g(x) = argmin_y g(y) + (1/2*gamma)|| x - y ||^2_{M}
    with respect to Metric M, defined by a positive definite matrix
    where g(x) is a function of the form
    g(x) = f(x) + i_{ x : F(x) = 0 }
    where F: R^{n}->R^{m}, and m <= n , and is assumed to be differentiable everywhere
          i_{} is the indicator function
          f: R^{n}-> R is scalar valued and assumed to be twice differentiable everywhere
    For a given x the prox operator is computed for the approximation
    f(x) + grad_f(x)^T(y - x) + (y-x)^T H_f(x) (y-x) + i_{ y: F(x) + J_F(x)*(y - x) = 0 }
    with grad_f(x) the gradient of f at x, H_f(x) the hessian of f at x, and  J_F(x) the Jacobian of F at x

    Note that if f is quadratic and F is linear this is exact.
    '''
    def __init__(self,f,F,metric = None,JF_fixed = False,Hf_fixed = False,gamma = 2.0):
        '''
        :param f: (functorch compatible function) a parameterized function f with input of the form (x,parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  f is defined unbatched, the method will call vmap, to "raise" f to batch dim

        :param F:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
        :param metric: (functorch compatible function) a parameterized function with inputs of the form (x,parms) and output a positive definite matrix with dimension equal to the dimsnsion of x
                  assumed to be defined unbatched, method will call vmap to "raise" to batch dim. If metric is None, will default to return the Identiy. 
        :param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed with call to self.prefactor()
        :param Hf_fixed: (Bool) Indicates if the Hessian of f, or metric needs to be computed as each iteration. Default is False, if True prox computation will be prefactored with call to self.prefactor()
        :param gamma : (float)
        :param n_dim: (int) the dimension of x for precomputing Jacobian
        :param parm_dim: (int) the dimension of parms for precomputing Jacobian
        '''
        super().__init__()
        self.f = vmap(f)
        self.f_grad = vmap(grad(f,argnums = 0))
        self.H_vec = vmap(hessian(f,argnums = 0))
        self.F = vmap(F)
        self.JF = vmap(jacrev(F,argnums = 0))
        self.gamma = gamma
        if metric != None:
            self.metric = metric
        else:
            self.metric = lambda x,parms : torch.eye(x.shape[-1])
        self.JF_fixed = JF_fixed
        self.Hf_fixed = Hf_fixed
    def prefactor(self,x,parms):
        '''
        This function computes a prefactoring of the prox solution for given problem parameters under assumption factoring is 
        not dependent on x, e.g. Constraints are linear, objective is quadratic, metric does not depend on x.
        '''
        if self.JF_fixed ==True:
            with torch.no_grad():
                # Prefactor Jacobian of the constraints for the given parms
                JFx = self.JF(x,parms)
                Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
                null_dim = Qc.shape[-1] - Rc.shape[-1]
                Rr = Rc[:,:-null_dim,:]
                Qr = Qc[:,:,:-null_dim]
                Qn = Qc[:,:,-null_dim:]
                self.JFx = JFx
                self.Qr = Qr
                self.Qn = Qn
                self.Rr = Rr
        if self.Hf_fixed == True:
            # Prefactor the Hessian of the objective for the given parms
            # Compute the Hessian of f
            Hf = self.H_vec(x,parms)
            ### Compute the gradient step and Projection Operation
            Pm = vmap(self.metric)(x,parms)
            Md =  (self.gamma/2)*Pm + Hf
            QTMdQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Md),Qn)
            Lcho = torch.linalg.cholesky(QTMdQ)
            self.Md = Md
            self.Lcho = Lcho
    def forward(self, x, parms):
        #print(self.metric.P_d)
        if self.JF_fixed == False:
            with torch.no_grad():
                JFx = self.JF(x,parms)
                ### Take a QR decomposition of the Jacobian
                Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
                null_dim = Qc.shape[-1] - Rc.shape[-1]
                Rr = Rc[:,:-null_dim,:]
                Qr = Qc[:,:,:-null_dim]
                Qn = Qc[:,:,-null_dim:]
        if self.JF_fixed == True:
            JFx = self.JFx
            Qr = self.Qr
            Qn = self.Qn
            Rr = self.Rr
        if self.Hf_fixed ==False:
            # Prefactor the Hessian of the objective for the given parms
            # Compute the Hessian of f
            Hf = self.H_vec(x,parms)
            ### Compute the gradient step and Projection Operation
            Pm = vmap(self.metric)(x,parms)
            Md =  (self.gamma/2)*Pm + Hf
            QTMdQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Md),Qn)
            Lcho = torch.linalg.cholesky(QTMdQ)
        if self.Hf_fixed ==True:
            Md = self.Md
            Lcho = self.Lcho
        Fx = self.F(x,parms)
        # Compute the RHS
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)
        #### Find a solution to RHS
        Rr_T = torch.transpose(Rr,1,2)
        zeta_r = torch.linalg.solve_triangular(Rr_T,b,upper=False,left = True)
        zeta = torch.bmm(Qr,zeta_r)
        # Compute the gradient of f
        fg = torch.unsqueeze(self.f_grad(x,parms),dim = -1)
        zq = torch.bmm(Md,x_mat - zeta) - (self.gamma/2)*fg
        zq = torch.bmm(torch.transpose(Qn,1,2),zq)
        zq = torch.linalg.solve_triangular(Lcho,zq,upper = False, left = True)
        zq = torch.linalg.solve_triangular(torch.transpose(Lcho,1,2),zq,upper = True, left = True)
        zq = torch.bmm(Qn,zq)
        x_new =  zq + zeta
        return torch.squeeze(x_new,dim = -1)

class ProxBoxConstraint(torch.nn.Module):
    '''
    Computes the projection onto the box constraints l_b <= x <= u_b for constants l_b and u_b.
    '''
    def __init__(self,f_lower_bound,f_upper_bound):
        '''
        :param f_lower_bound: (function) has form f(parms), returns tensor of lower bound constraints, is defined unbatched, will be 'raised' with vmap.
        :param f_upper_bound: (function) has form f(parms), returns tensor of upper bound constraints, is defined unbatched, will be 'raised' with vmap.
        '''
        super().__init__()
        self.lower_bound_func = vmap(f_lower_bound)
        self.upper_bound_func = vmap(f_upper_bound)
    def forward(self, x, parms):
        l_b = self.lower_bound_func(parms)
        u_b = self.upper_bound_func(parms)
        return l_b + torch.relu( u_b - l_b - torch.relu(u_b - x ))

class DRSolver(nn.Module):
    """
    Implementation of a Parameteric Douglas Rachford (DR) Solution routine for problems of the form
    
    min f(x)
    subject to:
    F_ineq(x) <= 0
    F_eq(x)= 0
    
    The problem is reformulated as
    
    min f(x)
    subject to:
    F(x,s) = 0
    s>=0
    
    for slack variables s, and F(x,s) defined as

    F(x,s) = [ F_eq(x) ; F_ineq(x) + s ]

    DR is an operator splitting approach, here applied to the splitting

    min g_1(x,s) + g_2(x,s)

    with
     g_1(x,s) = f(x) + i_{ (x,s) : F(x,s) = 0}
     g_2(x) = i_{ s : s>=0 }

    where i_{S} is the indicator function on set S.

    The solver uses a second order approximation of the objective and first order approximation of the constraints.
    """
    def __init__(self,f_obj = None,
                 F_ineq = None,
                 F_eq = None,
                 x_dim = 0,
                 n_ineq = 0,
                 n_eq = 0,
                 JF_fixed = False,
                 Hf_fixed = False,
                 num_steps=3,
                 metric = None,
                 state_slack_bound = 1e3):
        """
        :param f_obj: functorch compatible function) a parameterized function f with input of the form (x,parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  f is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the objective to be optimized
        :param F_ineq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the inequality constraints to satisfy, F_ineq(x) <= 0
        :param F_eq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the equality constraints to satisfy, F_eq(x) = 0
        :param x_dim: (int) dimension of the primal variables
        :param n_ineq: (int) number of inequality constraints
        :param n_eq: (int) number of equality constraints
        :param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed at x=0, parms = 0
        :param Hf_fixed: (Bool) Indicates if the Hessian of f should be re-computed at each iteration. Default is False, if True Hessian of f will be precomputed at x=0, parms = 0
        :param num_steps: (int) number of iteration steps for the Douglas Rachford method
        :param metric: (function) function from (x,parms) to a poitive definite matrix with dimension equal to dimension of x will be used as the metric for prox evaluations, if None will default to return the identity.
        :param state_slack_bound: (float) an approximate bound on the max absolute value for the state and slack variables, this bound must be satisfied but does not need to be exact, a closer bound is better.
        """
        super().__init__()
        self.x_dim = x_dim
        self.n_ineq = n_ineq
        self.n_eq = n_eq
        self.num_steps = num_steps
        self.JF_fixed = JF_fixed
        self.Hf_fixed = Hf_fixed
        self.state_slack_bound = state_slack_bound
        self.n_dim = self.x_dim + self.n_ineq
        self.f_obj = f_obj
        self.F_ineq = F_ineq
        self.F_eq = F_eq
        if n_eq > x_dim: print('ERROR: Equality constraints are overdetermined')
        #### Convert problem inputs to the standard form for the DR iterations
        #i.d. problem type
        #pid =
        #    = 1 only equality constraints
        #    = 2 only inequality constraints
        #    = 3 both equality and inequality constraints
        #    = 0 Error: no constraints
        pid = 2*(self.F_ineq != None) + (self.F_eq != None)
        self.pid = pid
        if pid == 0: print( 'ERROR: One of F_eq or F_ineq must be defined')
        if pid == 1:
            def f(xs, parms):
                return self.f_obj(xs,parms)
            def F(xs, parms):
                return self.F_eq(xs,parms)
        if pid == 2:
            def f(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.f_obj(x,parms)
            def F(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.F_ineq(x,parms) + s
        if pid ==3 :
            def Fs_ineq(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return torch.cat( (torch.zeros(self.n_eq),  self.F_ineq(x,parms) + s))
            def Fs_eq(xs,parms):
                x = xs[0:self.x_dim]
                return torch.cat( (self.F_eq(x,parms), torch.zeros(self.n_ineq) ))
            def F(xs, parms):
                return Fs_eq(xs,parms) + Fs_ineq(xs,parms)
            def f(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.f_obj(x,parms)
        self.F = F
        self.f = f
        self.metric = metric
        self.foF = ProxObjectivePlusEqualityConstraint(self.f,self.F,self.metric,JF_fixed = self.JF_fixed,Hf_fixed = self.Hf_fixed)
        ### Set the Prox of g_2(x,s)
        ## define the slack bounds
        upper_bound = self.state_slack_bound*torch.ones(self.n_dim)
        lower_bound = torch.cat( (-self.state_slack_bound*torch.ones(self.x_dim),torch.zeros(self.n_ineq)))
        def f_upper(parms):
            return upper_bound
        def f_lower(parms):
            return lower_bound
        self.sp = ProxBoxConstraint(f_lower,f_upper)
    def forward(self,x,parms):
        if self.n_ineq > 0:
            #add the slack variables
            x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        if (self.JF_fixed ==True) or (self.Hf_fixed == True):
            self.foF.prefactor(x,parms)
        x_k = x
        x_hist = []
        x_hist.append(x_k)
        for n in range(self.num_steps):
            y_k = self.sp(x_k,parms)
            z_k = self.foF(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
            cnv_gap = z_k - y_k
            x_hist.append(x_k)
        cnv_gap = (z_k - y_k)**2
        return x_k_new, cnv_gap

class ADMMSolver(nn.Module):
    """
    Implementation of an ADMM Solution routine for problems of the form
    min f(x)
    subject to:
    F_ineq(x) <= 0
    F_eq(x)= 0
    
    The problem is reformulated as
    min f(x)
    subject to:
    F(x,s) = 0
    s>=0
    
    for slack variables s, and F(x,s) defined as
    F(x,s) = [ F_eq(x) ; F_ineq(x) + s ]


    ADMM is an operator splitting approach, here applied to the splitting

    min g_1(x,s) + g_2(x,s)

    with
     g_1(x,s) = f(x) + i_{ (x,s) : F(x,s) = 0}
     g_2(x) = i_{ s : s>=0 }

    where i_{S} is the indicator function on set S.

    The solver uses a second order approximation of the objective and first order approximation of the constraints.
    """
    def __init__(self,f_obj = None,
                 F_ineq = None,
                 F_eq = None,
                 x_dim = 0,
                 n_ineq = 0,
                 n_eq = 0,
                 JF_fixed = False,
                 Hf_fixed = False,
                 num_steps=3,
                 metric = None,
                 state_slack_bound = 1e3,
                 alpha = 0.5):
        """
        :param f_obj: functorch compatible function) a parameterized function f with input of the form (x,parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  f is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the objective to be optimized
        :param F_ineq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the inequality constraints to satisfy, F_ineq(x) <= 0
        :param F_eq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the equality constraints to satisfy, F_eq(x) = 0
        :param x_dim: (int) dimension of the primal variables
        :param n_ineq: (int) number of inequality constraints
        :param n_eq: (int) number of equality constraints
        :param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed at x=0, parms = 0
        :param Hf_fixed: (Bool) Indicates if the Hessian of f should be re-computed at each iteration. Default is False, if True Hessian of f will be precomputed at x=0, parms = 0
        :param num_steps: (int) number of iteration steps for the Douglas Rachford method using Identity metric
        :param metric: (function) function from (x,parms) to a poitive definite matrix with dimension equal to dimension of x will be used as the metric for prox evaluations, if None will default to return the identity.
        :param state_slack_bound: (float) an approximate bound on the max absolute value for the state and slack variables, this bound must be satisfied but does not need to be exact, a closer bound is better.
        :param alpha: (scalar) value in (0,1), under or over relaxation parameter of ADMM, default for standard ADMM is 0.5
        """
        super().__init__()
        self.x_dim = x_dim
        self.n_ineq = n_ineq
        self.n_eq = n_eq
        self.num_steps = num_steps
        self.JF_fixed = JF_fixed
        self.Hf_fixed = Hf_fixed
        self.state_slack_bound = state_slack_bound
        self.n_dim = self.x_dim + self.n_ineq
        self.f_obj = f_obj
        self.F_ineq = F_ineq
        self.F_eq = F_eq
        self.alpha = alpha
        if n_eq > x_dim: print('ERROR: Equality constraints are overdetermined')
        #### Convert problem inputs to the standard form for the DR iterations
        #i.d. problem type
        #pid =
        #    = 1 only equality constraints
        #    = 2 only inequality constraints
        #    = 3 both equality and inequality constraints
        #    = 0 Error: no constraints
        pid = 2*(self.F_ineq != None) + (self.F_eq != None)
        self.pid = pid
        if pid == 0: print( 'ERROR: One of F_eq or F_ineq must be defined')
        if pid == 1:
            def f(xs, parms):
                return self.f_obj(xs,parms)
            def F(xs, parms):
                return self.F_eq(xs,parms)
        if pid == 2:
            def f(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.f_obj(x,parms)
            def F(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.F_ineq(x,parms) + s
        if pid ==3 :
            def Fs_ineq(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return torch.cat( (torch.zeros(self.n_eq),  self.F_ineq(x,parms) + s))
            def Fs_eq(xs,parms):
                x = xs[0:self.x_dim]
                return torch.cat( (self.F_eq(x,parms), torch.zeros(self.n_ineq) ))
            def F(xs, parms):
                return Fs_eq(xs,parms) + Fs_ineq(xs,parms)
            def f(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.f_obj(x,parms)
        self.F = F
        self.f = f
        self.metric = metric
        #adjust gamma for ADMM computation
        self.gamma = 1/2
        self.foF = ProxObjectivePlusEqualityConstraint(self.f,self.F,self.metric,JF_fixed = self.JF_fixed,Hf_fixed = self.Hf_fixed,gamma = self.gamma)
        ## define the slack bounds
        upper_bound = self.state_slack_bound*torch.ones(self.n_dim)
        lower_bound = torch.cat( (-self.state_slack_bound*torch.ones(self.x_dim),torch.zeros(self.n_ineq)))
        def f_upper(parms):
            return upper_bound
        def f_lower(parms):
            return lower_bound
        self.sp = ProxBoxConstraint(f_lower,f_upper)
    def forward(self,x,parms):
        if self.n_ineq>0 :
            #add the slack variables
            x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        if (self.JF_fixed ==True) or (self.Hf_fixed == True):
            self.foF.prefactor(x,parms)
        y_k = x
        w_k = torch.zeros(x.shape)
        x_hist = []
        x_hist.append(x)
        for n in range(self.num_steps):
            y_k_old = y_k
            x_k = self.foF(y_k - w_k,parms)
            xA_k = 2*self.alpha*x_k + (1-2*self.alpha)*y_k
            y_k = self.sp(xA_k + w_k,parms)
            w_k = w_k + xA_k - y_k
            r_gap = x_k - y_k
            s_gap = y_k - y_k_old
            cnv_gap = r_gap**2 + s_gap**2
            x_hist.append(x_k)
        return x_k, cnv_gap

class ParaMetricDiagonal(torch.nn.Module):
    def __init__(self,n_dim,parm_dim,upper_bound,lower_bound,scl_upper_bound = 0.2,scl_lower_bound = 0.05):
        super().__init__()
        self.n_dim = n_dim
        self.P_diag_upper_bound = upper_bound
        self.P_diag_lower_bound = lower_bound
        self.parm_dim = parm_dim
        self.hidden_dim = np.round(10*self.parm_dim).astype(int)
        self.DiagMap = nn.Sequential(
          nn.Linear(self.parm_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.n_dim),
        )
        self.scl_upper_bound = scl_upper_bound
        self.scl_lower_bound = scl_lower_bound
        self.ScaleMap = nn.Sequential(
                nn.Linear(self.parm_dim,self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,1)
                )
    def forward(self,x,parms):
        Pd = self.DiagMap(parms)
        scl = self.scl_lower_bound +  torch.sigmoid(self.ScaleMap(parms))*( self.scl_upper_bound - self.scl_lower_bound)
        P_diag = scl*( self.P_diag_lower_bound + torch.sigmoid(Pd)*(self.P_diag_upper_bound - self.P_diag_lower_bound) )
        Pm = torch.diag(P_diag)
        return Pm
    def scl_comp(self,x,parms):
        scl = torch.sigmoid(self.ScaleMap(parms))
        return scl