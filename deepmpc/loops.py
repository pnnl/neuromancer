"""
Open and closed loop dynamical models
x: states (x0 - initial conditions)
y: predicted outputs
ym: measured outputs
u: control inputs
d: uncontrolled inputs (measured disturbances)
r: reference signals

generic closed-loop dynamics:
x+ = fx(x) o fu(u) o fd(d)
y =  fy(x)
x = estim(yp,up,dp,x_prev)
u = policy(x,u,d,r)

Dynamical models from dynamics.py
estimator from estimators.py
policy from  policies.py
"""

# pytorch imports
import torch
import torch.nn as nn
#local imports
import estimators
import policies
import dynamics
from blocks import MLP

# TODO: custom loss functions with templates
# IDEA: user should have opportunity to define custom loss functions easily via high level API


class OpenLoop(nn.Module):
    def __init__(self, model=dynamics.BlockSSM, estim=estimators.LinearEstimator, Q_e=1.0, **linargs):
        """
        :param model: SSM mappings, see dynamics.py
        :param estim: state estimator mapping, see estimators.py

        input data trajectories:
        Y: measured outputs p (past)
        U: control inputs p (past), f (future)
        D: measured disturbances p (past), f (future)
        nsamples: prediction horizon length
        """
        super().__init__()
        self.model = model
        self.estim = estim
        self.Q_e = Q_e
        self.criterion =  torch.nn.MSELoss()

    def n_step(self, data):
        Yp, Yf, Up, Uf, Dp, Df = data.values()
        X_pred, Y_pred, reg_error = self.forward(Yp, Up, Uf, Dp, Df, nsamples=Yf.shape[0])
        U_pred = Uf
        loss = self.criterion(Y_pred.squeeze(), Yf.squeeze())
        return {f'{data.name}_nstep_obj_loss': loss,
                f'{data.name}_nstep_reg_error': reg_error,
                f'{data.name}_nstep_loss': loss + reg_error,
                'X_pred': X_pred,
                'Y_pred': Y_pred,
                'U_pred': U_pred,
                'Df': Df}

    def loop_step(self, data):
        Yp, Yf, Up, Uf, Dp, Df = data.values()
        X_pred, Y_pred, reg_error = self.forward(Yp, Up, Uf, Dp, Df, nsamples=Yf.shape[0])
        U_pred = Uf
        loss = self.criterion(Y_pred.squeeze(), Yf.squeeze())
        return {f'{data.name}_loop_obj_loss': loss,
                f'{data.name}_loop_reg_error': reg_error,
                f'{data.name}_loop_loss': loss + reg_error,
                'X_pred': X_pred,
                'Y_pred': Y_pred,
                'U_pred': U_pred,
                'Df': Df}

    def forward(self, Yp, Up, Uf, Dp, Df, nsamples=1):
        x0, reg_error_estim = self.estim(Yp, Up, Dp)
        Xf, Yf, reg_error_model = self.model(x=x0, U=Uf, D=Df, nsamples=nsamples)
        # Calculate mse for smoother state estimator predictions. Last prediction of SSM for a batch should equal
        # the state estimation of the next sequential batch. Warning: This will not perform as expected
        # if batches are shuffled in SGD (we are using full GD so we are okay here.
        estim_error = self.Q_e*torch.nn.functional.mse_loss(x0[1:], Xf[-1, :-1:, :])
        reg_error = reg_error_model + reg_error_estim + estim_error
        return Xf, Yf, reg_error


class ClosedLoop(nn.Module):
    def __init__(self, model=dynamics.BlockSSM, estim=estimators.LinearEstimator,
                 policy=policies.LinearPolicy, Q_e=1.0, **linargs):
        """
        :param model: SSM mappings, see dynamics.py
        :param estim: state estimator mapping, see estimators.py

        input data trajectories:
        Y: measured outputs p (past)
        U: control inputs  f (future)
        D: measured disturbances p (past), f (future)
        R: desired references f (future)
        """
        super().__init__()
        self.model = model
        self.estim = estim
        self.policy = policy
        self.Q_e = Q_e

    def forward(self, Yp, Up, Dp, Df, Rf, nsamples=1):
        x0, reg_error_estim = self.estim(Yp, Up, Dp)
        # TODO: how do we make policy design more flexible for the user?
        Uf, reg_error_policy = self.policy(x0, Df, Rf)
        Uf = Uf.unsqueeze(2).reshape(Uf.shape[0], self.model.nu, -1)
        Uf = Uf.permute(2,0,1)
        # Uf = Uf.reshape(Rf.shape[0], -1, self.model.nu)  # not sure if it does not shuffle
        Xf, Yf, reg_error_model = self.model(x=x0, U=Uf, D=Df, nsamples=nsamples)
        # Calculate mse for smoother state estimator predictions. Last prediction of SSM for a batch should equal
        # the state estimation of the next sequential batch. Warning: This will not perform as expected
        # if batches are shuffled in SGD (we are using full GD so we are okay here.
        estim_error = self.Q_e * torch.nn.functional.mse_loss(x0[1:], Xf[-1, :-1:, :])
        reg_error = reg_error_model + reg_error_policy + reg_error_estim + estim_error
        return Xf, Yf, Uf, reg_error

"""
CONSTRAINTS DESIGN IDEAS

# TODO: make internal variables from return attributes in models, estimators and policies
#  such that we can treat them with constraints class
# e.g., Sxmax = MaxPenalty(model.X, Xmax)
# Loop input arguments as a list of constraints classes: constr=[]
# example: constr = [MaxPenalty(model.X, Xmax), MinPenalty(model.X, Xmin),
#                    MaxPenalty(policy.U, Umax), MinPenalty(policy.U, Umin)]
# OR symbolic definition with parser:
# constr = [ X <= Xmax, X >= Xmin, U <= Umax, U >= Umin]
# TODO: what model binding mechamism we shall use for flexible design?
# shall we use symbolic variable definition as used in optimization toolboxes?
# based on used objects we can extract all model variables into a list of object attributes
"""


"""
OBJECTIVE DESIGN IDEAS

aggregate of torch.nn.functional expressions
objective = 0
objective += F.MSE(X-X_trg)
objective += F.MSE(U)
objective += f(X)           #  custom loss term
"""



class Model(nn.Module):
    def __init__(self, **linargs):
        """
        Base model class of the framework
        Allows flexible constriction of the loops (Open, Closed), and optimization problems

        Attributes:
        self.dynamics = system dynamics model
        self.estim = state estimator
        self.policy = control policy
        self.constraints = constraints
        self.parametric_map = solution map for constrained optimization problem, equivalent to policy for control
        self.objective = objective function
        self.variables = all model variables Xi - to be computed by the model
        self.parameters = all model parameters Theta - to be obtained from the dataset
        self.type = type of the model: 'OpenLoop', 'ClosedLoop', 'pOP' - parametric optimization problem
        self.variables - list of all variables of the model, each submodule should have this atribute

        Methods:
        add() method for stepwise construction of the loop
        delete() method for dropping unwanted parts of the model
        compile() method for putting things together
        forward() for forward pass

        -------------- VARIABLES AND PARAMETERS -------------------
        variables Xi - computed by the model
        parameters Theta - obtained from the dataset
        OpenLoop:
            variables Xi:
                states - X, outputs - Y, slacks - S
            parameters Theta:
                inputs - U, disturbances - D,
                min_states - X_min, max_states - X_max,
                min_outputs - Y_min, max_outputs - Y_max,
                target_states - X_trg, target_outputs - Y_trg,
        ClosedLoop:
            variables Xi:
                states - X, outputs - Y, inputs - U, slacks - S
            parameters Theta:
                disturbances - D,
                min_states - X_min, max_states - X_max,
                min_outputs - Y_min, max_outputs - Y_max,
                min_inputs - U_min, max_inputs - U_max,
                target_states - X_trg, target_outputs - Y_trg
        pOP:
            variables Xi:
                states - X, slacks - S
            parameters Theta:
                min_states - X_min, max_states - X_max, target_states - X_trg,

        --------------- MODEL TYPES ------------------
        OpenLoop:
            min objective(Xi,Theta)
            s.t.
                X = estim(X,Y,U,D)
                X, Y = dynamics(X,U,D)
                S = constraints(Xi,Theta)

        ClosedLoop:
            min objective(Xi,Theta)
            s.t.
                X = estim(X,Y,U,D)
                U = policy(Xi)
                X, Y = dynamics(X,U,D)
                S = constraints(Xi,Theta)

        pOP: parametric optimization problem
            min objective(X,S,Theta)
            s.t. S = constraints(X,Theta)
                 X = parametric_map(Theta)

            more specific optimization problem formulation with hard constraints:
            https://en.wikipedia.org/wiki/Parametric_programming
                min_W f(X,Theta)
                s.t. g_max(X) <= X_max
                     g_min(X) >= X_min
                     g_eq(X) = X_trg
                     X = h_W(Theta)
                f(X, Theta): objective class
                g(X, Theta): constraints class
                h_W(Theta): parametric_map

        """
        super().__init__()
        self.dynamics = None
        self.estim = None
        self.policy = None
        self.constraints = []
        self.parametric_map = None
        self.objective = None
        self.variables = None
        self.parameters = None
        self.type = None    # 'OpenLoop', 'ClosedLoop', 'pOP'

    def add(self, dynamics=None, estim=None, policy=None, constraints=None,
                 parametric_map=None, objective=None, **linargs):
        # add part of the model
        self.dynamics = dynamics
        self.estim = estim
        self.policy = policy
        self.parametric_map = parametric_map
        self.objective = objective
        if constraints is not None:
            for i in len(constraints):
                self.constraints.append(constraints[i])

    def delete(self, **linargs):
        pass
        # delete part of the model, e.g., policy or particular constraint

    def compile(self):
        # TODO: step 1, check if comensions of components hold
        # TODO: setp 2, assign variables and parameters based on the current components
        # TODO: step 3, define type of the model based on defined components
        # TODO: step 4, construct compiled model for forward pass based on the current type

        # define type of the model based on its components
        if self.policy == None and not all((self.dynamics, self.estim, self.objective)):
            self.type = 'OpenLoop'
        elif not all((self.dynamics, self.estim, self.policy, self.objective)):
            self.type = 'ClosedLoop'
        elif all((self.dynamics, self.estim, self.policy)) \
                and not all((self.parametric_map, self.objective, self.constraints)):
            self.type = 'pOP'

        self.variables = None
        self.parameters = None  # e.g., constraints bounds, references, disturbances - obtained from dataset
        self.model_compiled = None

        
    def forward(self, **linargs):
        # TODO: how to handle varying arguments based on the model type?
        # define specific forward passes based on the model type? e.g., open loop, closed loop, optimization problem?
        if self.type == 'OpenLoop':
            pass
        elif self.type == 'ClosedLoop':
            pass
        elif self.type == 'pOP':
            pass


# TODO: alternative to single model class is definition of constraints and objectives as arguments in loops objects

if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    Np = 2
    Nf = 10
    samples = 100
    # Data format: (N,samples,dim)
    x = torch.rand(samples, nx)
    Yp = torch.rand(Np, samples, ny)
    Up = torch.rand(Np, samples, nu)
    Uf = torch.rand(Nf, samples, nu)
    Dp = torch.rand(Np, samples, nd)
    Df = torch.rand(Nf, samples, nd)
    Rf = torch.rand(Nf, samples, ny)
    x0 = torch.rand(samples, nx)

    # block  SSM
    fx, fu, fd = [MLP(insize, nx, hsizes=[64, 64, 64]) for insize in [nx, nu, nd]]
    fy = MLP(nx, ny, hsizes=[64, 64, 64])
    model1 = dynamics.BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)
    model_out = model1(x0, Uf, Df)

    # black box SSM
    fxud = MLP(nx + nu + nd, nx, hsizes=[64, 64, 64])
    model2 = dynamics.BlackSSM(nx, nu, nd, ny, fxud, fy)
    model_out = model2(x0, Uf, Df)

    # TODO: issue with the estimator switching 0th index with 1st index
    est = estimators.LinearEstimator(ny, nx)
    est_out = est(Yp, Up, Dp)

    pol = policies.LinearPolicy(nx, nu, nd, ny, Nf)
    pol_out = pol(x0, Df, Rf)

    ol = OpenLoop(model1,est)
    ol_out = ol(Yp, Up, Uf, Dp, Df)

    cl = ClosedLoop(model1, est, pol)
    cl_out = cl(Yp, Up, Dp, Df, Rf)

    ol = OpenLoop(model2, est)
    ol_out = ol(Yp, Up, Uf, Dp, Df)

    cl = ClosedLoop(model2, est, pol)
    cl_out = cl(Yp, Up, Dp, Df, Rf)
