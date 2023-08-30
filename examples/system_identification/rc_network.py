# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from neuromancer.psl.perturb import Periodic, WhiteNoise
from scipy.integrate import solve_ivp
from scipy import interpolate
import h5py
from neuromancer.dynamics import integrators, ode, physics, interpolation
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.system import Node, System
from collections import OrderedDict
from neuromancer.loggers import BasicLogger
from neuromancer.trainer import Trainer
np.random.seed(200)
torch.manual_seed(0)
device = 'cpu'

plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times"]
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 10})


params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10}
plt.rcParams.update(params)


# %%
def H5DataLoader(path):
    f_base = h5py.File(path,'r')
    Keys_base = list(f_base.keys())
    Data_Dict_base = {}
    for K in Keys_base:
        Data_Dict_base[K] = np.asarray(f_base[K]).T
    return Data_Dict_base,Keys_base

def SelectData(Data_weather, Key_weather, startday, numdays):
    start_minute = (startday-1) * 24 * 60 + 1
    end_minute = start_minute + numdays * 24 * 60
    Data_new = Data_weather[Key_weather[1]][start_minute:end_minute,1]
    
    return Data_new

def External_Data():
    """_summary_
    This function load the external weather data file
    to create the external forcing intepolant.
    """
    path_weather = (os.getcwd())
    Weather_data = os.path.join(path_weather,'/Users/koch404/Library/CloudStorage/OneDrive-PNNL/Documents/LocalCode/nm-dev/dae/koch_dev/data/OAT_Chicago_TMY.h5')
    Data_weather, Key_weather = H5DataLoader(Weather_data)
    # print(Key_weather)
    # print(Data_weather[Key_weather[1]].shape)

    S_day = 130
    E_day = 243
    D= np.arange(S_day,E_day)
    Days = len(D)

    # print(len(Days))
    # Data_store = 

    # for i, val in enumerate(Days)
    WData = SelectData(Data_weather,Key_weather, S_day, Days)

    WData_reshape = WData.reshape(Days, 1440)
    s_day = 10
    # print(WData_reshape.shape)
    # print(WData_reshape[70].shape)
    plt.plot(WData_reshape[s_day][:]+273.14)
    plt.show()
    return WData_reshape[s_day][:]+273.14

class RC_Network:
    def __init__(self, 
                 R = None, 
                 C = None,
                 U=None, 
                 adj=None,
                 nx=2,
                 x0=None,
                 Seed=59):
        """_summary_

        :param R: [float, np.array], Coupled Resistances
        :param C: [float, np.array], Room Capacitance
        :param nsim: [int], length of simulation, defaults to 1001
        :param ninit: [int], starting time of simulation, defaults to 0
        :param ts: [float], rate of sampling, defaults to 0.1
        :param adj: [np.array, shape=(2,*) or (nx,nx)], adjacency list or matrix, defaults to None
        :param nx: (int), number of nodes, defaults to 1
        :param seed: seed for random number generator, defaults to 59
        """
        self.adj_list = adj if adj is not None else self.get_adj()
        self.R = R if R is not None else self.get_R(
            self.adj_list, amax=20, amin=5, symmetric=True)
        self.C = C if C is not None else self.get_C(nx)
        self.R_ext = self.get_R(np.tile(np.arange(nx),(2,1)), amax=15, symmetric=False)
        self.R_int = self.get_R(np.tile(np.arange(nx),(2,1)), Rval=1.0, amax=15, symmetric=False)
        self.x0 = x0 if x0 is not None else self.get_x0(nx)
        
        self.R_extCi = (1.0 / (self.R_ext * self.C))
        self.R_intCi = (1.0 / (self.R_int * self.C))
        self.seed = Seed
        self.f_ext = None
        self.f_int = None

    def get_x0(self, nx = None, rseed=None):
        if rseed is not None:
            np.random.seed(rseed)
        nx = nx if nx is not None else self.nx
        x0 = np.random.uniform(low=20, high=25, size=(nx,)) + 273.14
        return x0

    def get_U(self, nsim=None, nx=None, periods = None, rseed=1):
        period_length = 500
        nsim = nsim if nsim is not None else self.nsim
        nx = nx if nx is not None else self.nx
        if periods is None:
            periods = int(np.ceil(nsim / period_length))
        global_source = Periodic(nsim=nsim, xmin=280.0, xmax=300.0, numPeriods=periods)
        global_source += WhiteNoise(nsim=nsim, xmax=1, xmin=-1, rseed=rseed)
        
        #Generate individual heat sources in each room, with random noise, and offset periods
        ind_sources = Periodic(nx=nx, nsim=nsim+period_length, numPeriods=periods*2, xmin = 288, xmax=300)
        offsets = np.random.randint(0,period_length,nx)
        offsets = np.linspace(offsets, offsets + nsim-1, nsim, dtype=int)
        ind_sources = np.take_along_axis(ind_sources, offsets, axis=0)
        ind_sources += WhiteNoise(nx=nx, nsim=nsim, xmax=0.5, xmin=-0.5, rseed=rseed)
        return np.hstack([global_source,ind_sources])

    def get_R(self, adj_list, Rval=3.5, amax=20, amin=0, symmetric=True):
        #Default Rval is fiberglass insulation
        num = adj_list.shape[1]
        m2 = np.random.rand(num) * (amax-amin) + amin #surface area
        if symmetric:
            edge_map = {(i,j) : idx for idx, (i,j) in enumerate(adj_list.T)}
            edge_map = [edge_map[(d, s)] for (s,d) in adj_list.T]
            m2 = (m2 + m2[edge_map]) / 2.0  
        m2 = np.maximum(m2, 0.0000001)
        R = Rval / m2
        return R

    def get_C(self, num=1):
        C_air = 700
        d_air = 1.2
        v_max = 10.0 * 10.0 * 5.0
        v_min = 3.0 * 3.0 * 3.0
        V = np.random.rand(num) * (v_max - v_min) + v_min
        return C_air * d_air * V / 100.0
    
    def get_adj(self):
        return np.array([[0,1],[0,2],[0,3],[1,0],[1,3],[1,4],[2,0],[2,3],[3,0],[3,1],[3,2],[3,4],[4,1],[4,3]]).T

    def message_passing(self, receivers, senders, t, u):
        R = self.R        
        if type(self.C) is np.ndarray:
            C = self.C[self.adj_list[0]]
        else:
            C = self.C
        messages = (1.0 / (R*C)) * (senders - receivers)
        return messages
    
    def equations(self, t, x, f_ext,f_int):
        # external_source = u[0]
        # internal_sources = u[1:]
    
        dx = np.zeros_like(x)
                
        #Internal heat transfer
        u = None
        messages = self.message_passing(x[self.adj_list[0]], x[self.adj_list[1]], t, u)
        np.add.at(dx, self.adj_list[0], messages)
        
        #Outside heat transfer
        external_source = np.ones_like(x)* f_ext(t)
        deltas = external_source - x
        dx += self.R_extCi * deltas
        
        #Internal heat sources
        internal_sources = []
        for i in range(len(x)):
            internal_sources.append(f_int[f'node-{i}'](t))

        deltas = np.array(internal_sources) - x
        dx += self.R_intCi * deltas
        return dx

    def make_5_room(self,ts_tf=[0,1441]):
        adj = np.array([[0,1],[0,2],[0,3],[1,0],[1,3],[1,4],[2,0],[2,3],[3,0],[3,1],[3,2],[3,4],[4,1],[4,3]]).T
        RC_node  = RC_Network(nx=5, adj=adj)

        ext_data = np.tile(External_Data(),200)
        t_ = np.arange(0,len(ext_data))
        self.f_ext = interpolate.interp1d(t_, ext_data)

        tmp = Periodic(nx=5, nsim=len(t_)+500, numPeriods=4*2, xmin = 288, xmax=300)
        offset = np.random.randint(0,400,5)
        off_ = np.linspace(offset, offset + (len(t_)-1), len(t_), dtype=int)
        ind_sources = np.take_along_axis(tmp, off_, axis=0)
        ind_sources +=WhiteNoise(nx=5, nsim=len(t_), xmax=0.5, xmin=-0.5, rseed=self.seed)
        # print(ind_sources.shape, t_.shape, ">>")
        # plt.plot(ind_sources)

        # This is just to demonstrate that each zone can have it's own
        # internal load data. We are simply creating a data that is in functional form.
        self.f_int = {}
        for i in range(5):
            # print(ind_sources[:,i].shape,"***")
            self.f_int[f'node-{i}'] = interpolate.interp1d(t_, ind_sources[:,i])
        
        sol = solve_ivp(RC_node.equations, ts_tf,self.x0,method='RK45',args=[self.f_ext,self.f_int])
        return sol
    
    def get_external(self):
        return self.f_ext

    def get_internal(self):
        return self.f_int

    def get_ambAdj(self):
        # Here all the nodes are connected with external forcing state:5
        return np.array([[0,5],[1,5],[2,5],[3,5],[4,5]])
    
    def get_IntSourceAdj(self):
        return np.array([[0,6],[1,7],[2,8],[3,9],[4,10]])

# %% 
nx_node = 5
n_days = 1
start_endTime =[0,1440*n_days]
Time_cont = np.arange(start_endTime[0],start_endTime[1])
RC_net = RC_Network(nx=nx_node)
sol_ =RC_net.make_5_room(ts_tf=start_endTime)

interp_sol = np.zeros((len(Time_cont),nx_node))
fig, ax = plt.subplots(figsize=[8,6])
for i in range(nx_node):
    interp_sol[:,i] = interpolate.interp1d(sol_.t,sol_.y[i])._evaluate(Time_cont)
    ax.plot(Time_cont,interp_sol[:,i]-273.14,label=f'Node-{i}')
    # ax.plot(sol_.t, sol_.y[i]-273.14, label=f'Node-{i}')
ax.legend()

# Collecting the external and internal 
Sources_ = []
Ext_ = RC_net.get_external()._evaluate(Time_cont)
Sources_.append(Ext_)
for item in RC_net.get_internal().keys():
    Sources_.append(RC_net.get_internal()[item]._evaluate(Time_cont))

# Zone temperature, Sources data. 
Gen_traindata = np.hstack((interp_sol,np.array(Sources_).T))
U = np.array(Sources_).T

# %% Dataset construction:

# Prepare the generated data from the previous cell for Training...
nsim = interp_sol.shape[0]
nstep = 10 #int(interp_sol.shape[0]*0.01) 

train_data = {'Y': interp_sol, 'X': interp_sol, 'U': U[:,:2]}
dev_data = train_data
test_data = train_data
for d in [train_data, dev_data]:
    d['X'] = d['X'].reshape(nsim//nstep, nstep, nx_node)
    d['Y'] = d['Y'].reshape(nsim//nstep, nstep, nx_node)
    d['U'] = d['U'].reshape(nsim//nstep, nstep, 2)
    d['xn'] = d['X'][:, 0:1, :]

train_dataset, dev_dataset, = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
train_loader, dev_loader, test_loader = [DataLoader(d, batch_size=nsim//nstep, collate_fn=d.collate_fn, shuffle=True) for d in [train_dataset, dev_dataset, dev_dataset]]


# %% Helper function:
def generate_parameterized_edges(physics,edge_list):
    """
    Quick helper function to construct edge physics/objects from adj. list:
    """

    couplings = []
    if isinstance(physics,nn.Module): # is "physics" an instance or a class?
        # If we're in here, we expect one instance of "physics" for all edges in edge_list (homogeneous edges)
        physics.pins = edge_list
        couplings.append(physics)
        print(f'Broadcasting {physics} to all elements in edge list.')
    else: 
        # If we're in here, we expect different "physics" for each edge in edge_list (heterogeneous edges)
        for edge in edge_list:
            agent = physics(R=nn.Parameter(torch.tensor(50.0)),pins=[edge])
            couplings.append(agent)

        print(f'Assuming new {physics} for each element in edge list.')

    return couplings

# %% Instantiation/Model construction
adj_dT = np.array([[0,1],[0,2],[0,3],[1,0],[1,3],[1,4],[2,0],[2,3],[3,0],[3,1],[3,2],[3,4],[4,1],[4,3]]).T
nsim = 100
ts = 20.0
nx = 5
nu = 2

# %%
adjacency = list(adj_dT.T)                                                      # Interaction physics
outside = physics.SourceSink()                                                  # Constant/non-constant outdoor temp?
heater = physics.SourceSink()                                                   # Constant/non-constant outdoor temp?

# Define agents in network:
agents = [physics.RCNode(C=nn.Parameter(torch.tensor(5.0)),scaling=1.0e-5) for i in range(5)]  # heterogeneous population w/ identical physics

agents.append(outside) # Agent #5
agents.append(heater) # Agent #6
map = physics.map_from_agents(agents)                                           # Construct state mappings

# Define inter-node couplings:
couplings = generate_parameterized_edges(physics.DeltaTemp,adjacency)                   # Heterogeneous edges of same physics

# Couple w/ outside temp:
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[0,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[1,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[2,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[3,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[4,5]]))

# Couple w/ hvac:
couplings.append(physics.HVACConnection(pins=[[0,6],[1,6],[2,6],[3,6],[4,6]]))

# Define ODE system:
model_ode = ode.GeneralNetworkedODE(
    map = map,
    agents = agents,
    couplings = couplings,
    insize = nx+nu,
    outsize = nx,
    inductive_bias="compositional")

# %%
out = model_ode(torch.rand(20,7))
out.shape

# %%
fx_int = integrators.RK4(model_ode, interp_u = lambda tq, t, u: u, h=ts)

dynamics_model = System([Node(fx_int,['xn','U'],['xn'])])

# %%
x = variable("X")
xhat = variable("xn")[:, :-1, :]

reference_loss = ((xhat == x)^2)
reference_loss.name = "ref_loss"

objectives = [reference_loss]
constraints = []
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem([dynamics_model], loss)


optimizer = torch.optim.Adam(problem.parameters(), lr=0.01)
logger = BasicLogger(args=None, savedir='test', verbosity=1,
                     stdout=["dev_loss","train_loss"])

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_loader,
    optimizer,
    epochs=500,
    patience=100,
    warmup=5,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=logger,
)

# %%
best_model = trainer.train()

# %%
u = torch.from_numpy(U).to(device).float()
sol = torch.zeros((1440,5))
ic = torch.tensor(RC_net.x0)
for j in range(sol.shape[0]-1):
    if j==0:
        sol[[0],:] = ic.float()
        sol[[j+1],:] = fx_int(sol[[j],:],u[[j],:])
    else:
        sol[[j+1],:] = fx_int(sol[[j],:],u[[j],:])

# %%
plt.plot(sol.detach().numpy(),label='model')
# %%
