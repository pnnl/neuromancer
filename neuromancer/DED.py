



from glob import glob
import math
import os
from typing import Union
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch





#############################################################################


###      DATASET FUNCTIONS


##############################################################################



from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate






def _is_multisequence_data(data):
    return isinstance(data, list) and all([isinstance(x, dict) for x in data])


def _is_sequence_data(data):
    return isinstance(data, dict) and len({x.shape[0] for x in data.values()}) == 1


def _extract_var(data, regex):
    filtered = data.filter(regex=regex).values
    return filtered if filtered.size != 0 else None


SUPPORTED_EXTENSIONS = {".csv", ".mat"}


def read_file(file_or_dir):
    if os.path.isdir(file_or_dir):
        files = [
            os.path.join(file_or_dir, x)
            for x in os.listdir(file_or_dir)
            if os.path.splitext(x)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        return [_read_file(x) for x in sorted(files)]

    return _read_file(file_or_dir)


def _read_file(file_path):
    """Read data from MAT or CSV file into data dictionary.

    :param file_path: (str) path to a MAT or CSV file to load.
    """
    file_type = file_path.split(".")[-1].lower()
    if file_type == "mat":
        f = loadmat(file_path)
        Y = f.get("y", None)  # outputs
        X = f.get("x", None)
        U = f.get("u", None)  # inputs
        D = f.get("d", None)  # disturbances
        id_ = f.get("exp_id", None)  # experiment run id
    elif file_type == "csv":
        data = pd.read_csv(file_path)
        Y = _extract_var(data, "^y[0-9]+$")
        X = _extract_var(data, "^x[0-9]+$")
        U = _extract_var(data, "^u[0-9]+$")
        D = _extract_var(data, "^d[0-9]+$")
        id_ = _extract_var(data, "^exp_id")
    else:
        print(f"error: unsupported file type: {file_type}")

    assert any([v is not None for v in [Y, X, U, D]])

    if id_ is None:
        return {
            k: v for k, v in zip(["Y", "X", "U", "D"], [Y, X, U, D]) if v is not None
        }
    else:
        return [
            {k: v[id_.flatten() == i, ...] for k, v in zip(["Y", "X", "U", "D"], [Y, X, U, D]) if v is not None}
            for i in sorted(set(id_.flatten()))
        ]


def batch_tensor(x: torch.Tensor, steps: int, mh: bool = False):
    return x.unfold(0, steps, 1 if mh else steps)


def unbatch_tensor(x: torch.Tensor, mh: bool = False):
    return (
        torch.cat((x[:, :, :, 0], x[-1, :, :, 1:]), dim=0)
        if mh
        else torch.cat(torch.unbind(x, 0), dim=-1)
    )


def _get_sequence_time_slices(data):
    seq_lens = []
    for i, d in enumerate(data):
        seq_lens.append(None)
        for v in d.values():
            seq_lens[i] = seq_lens[i] or v.shape[0]
            assert seq_lens[i] == v.shape[0], \
                "sequence lengths within a dictionary must be equal"
    slices = []
    i = 0
    for seq_len in seq_lens:
        slices.append(slice(i, i + seq_len, 1))
        i += seq_len
    return slices


def _validate_keys(data):
    keys = set(data[0].keys())
    for d in data[1:]:
        other_keys = set(d.keys())
        assert len(keys - other_keys) == 0 and len(other_keys - keys) == 0, \
            "list of dictionaries must have matching keys across all dictionaries."
        keys = other_keys
    return keys













class fpSplitSequenceDataset(Dataset):
    def __init__(
        self,
        data,
        nsteps=1,
        psteps=1,
        moving_horizon=False,
        name="data",
    ):
        """Dataset for handling sequential data and transforming it into the dictionary structure
        used by NeuroMANCER models.

        :param data: (dict str: np.array) dictionary mapping variable names to tensors of shape
            (T, Dk), where T is number of time steps and Dk is dimensionality of variable k.
        :param nsteps: (int) n-step prediction horizon for batching data.
        :param psteps: (int) p-step past data available for n-step prediction.
        :param moving_horizon: (bool) if True, generate batches using sliding window with stride 1;
            else use stride N = nsteps + psteps.
        :param name: (str) name of dataset split.

        .. note:: To generate train/dev/test datasets and DataLoaders for each, see the
            `get_sequence_dataloaders` function.

        .. warning:: This dataset class requires the use of a special collate function that must be
            provided to PyTorch's DataLoader class; see the `collate_fn` method of this class.

        .. 
        """

        super().__init__()
        self.name = name

        self.multisequence = _is_multisequence_data(data)
        assert _is_sequence_data(data) or self.multisequence, \
            "data must be provided as a dictionary or list of dictionaries"

        if isinstance(data, dict):
            data = [data]

        keys = _validate_keys(data)

        # _sslices used to slice out sequences from a multi-sequence dataset
        self._sslices = _get_sequence_time_slices(data)
        assert all([nsteps+psteps < (sl.stop - sl.start) for sl in self._sslices]), \
            f"length of time series data must be greater than nsteps+psteps"

        self.nsteps = nsteps
        self.psteps = psteps
        self.Nsteps = nsteps+psteps

        self.variables = list(keys)
        self.full_data = torch.cat(
            [torch.cat([torch.tensor(d[k], dtype=torch.float) for k in self.variables], dim=1) for d in data],
            dim=0,
        )
        self.nsim = self.full_data.shape[0]
        self.dims = {k: (self.nsim, *data[0][k].shape[1:],) for k in self.variables}

        # _vslices used to slice out sequences of individual variables from full_data and batched_data
        i = 0
        self._vslices = {}
        for k, v in self.dims.items():
            self._vslices[k] = slice(i, i + v[1], 1)
            i += v[1]

        self.dims = {
            **self.dims,
            **{k + "p": (self.nsim - 1, v[1]) for k, v in self.dims.items()},
            **{k + "f": (self.nsim - 1, v[1]) for k, v in self.dims.items()},
            "nsim": self.nsim,
            "nsteps": nsteps,
        }

        self.batched_data = torch.cat(
            [batch_tensor(self.full_data[s, ...] , self.Nsteps, mh=moving_horizon) for s in self._sslices],
            dim=0,
        )
        self.batched_data = self.batched_data.permute(0, 2, 1)
        
        
    def __len__(self):
        """Gives the number of N-step batches in the dataset."""
        return len(self.batched_data) 

    def __getitem__(self, i):
        """Fetch a single N-step sequence from the dataset."""
        return {
            **{
                k + "p": self.batched_data[i, 0:self.psteps, self._vslices[k]]
                for k in self.variables
            },
            **{
                k + "f": self.batched_data[i , self.psteps:self.Nsteps, self._vslices[k]]
                for k in self.variables
            },
        }

    def _get_full_sequence_impl(self, start=0, end=None):
        """Returns the full sequence of data as a dictionary. Useful for open-loop evaluation.
        """
        if end is not None and end < 0:
            end = self.full_data.shape[0] + end
        elif end is None:
            end = self.full_data.shape[0]

        return {
            **{
                k + "p": self.full_data[start : self.psteps, self._vslices[k]].unsqueeze(1)
                for k in self.variables
            },
            **{
                k + "f": self.full_data[self.psteps : end, self._vslices[k]].unsqueeze(1)
                for k in self.variables
            },
            "name": "loop_" + self.name,
        }

    def get_full_sequence(self):
        return (
            [self._get_full_sequence_impl(start=s.start, end=s.stop) for s in self._sslices]
            if self.multisequence
            else self._get_full_sequence_impl()
        )

    def get_full_batch(self):
        return {
            **{
                k + "p": self.batched_data[:, 0:self.psteps, self._vslices[k]].transpose(0, 1)
                for k in self.variables
            },
            **{
                k + "f": self.batched_data[:, self.psteps:self.Nsteps, self._vslices[k]].transpose(0, 1)
                for k in self.variables
            },
            "name": "nstep_" + self.name,
        }

    def collate_fn(self, batch):
        """Batch collation for dictionaries of samples generated by this dataset. This wraps the
        default PyTorch batch collation function and does some light post-processing to transpose
        the data for NeuroMANCER models and add a "name" field.

        :param batch: (dict str: torch.Tensor) dataset sample.
        """
        batch = default_collate(batch)
        return {
            **{k: v.transpose(0, 1) for k, v in batch.items()},
            "name": "nstep_" + self.name,
        }

    def __repr__(self):
        varinfo = "\n    ".join([f"{x}: {d}" for x, d in self.dims.items() if x not in {"nsteps", "nsim"}])
        seqinfo = f"    nsequences: {len(self._sslices)}\n" if self.multisequence else ""
        return (
            f"{type(self).__name__}:\n"
            f"  multi-sequence: {self.multisequence}\n"
            f"{seqinfo}"
            f"  variables (shapes):\n"
            f"    {varinfo}\n"
            f"  nsim: {self.nsim}\n"
            f"  nsteps: {self.nsteps}\n"
            f"  psteps: {self.psteps}\n"
            f"  batches: {len(self)}\n"
        )














#############################################################################


###      ESTIMATOR FUNCTIONS

##############################################################################



# pytorch imports
import torch
import torch.nn as nn

# ecosystem imports
import slim




# local imports
from neuromancer.estimators import TimeDelayEstimator






class FullyObservable_fp(TimeDelayEstimator):
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.Identity, hsizes=[],
                 input_keys=['Y_dp'], linargs=dict(), name='fully_observable'):
        """
        Dummmy estimator to use consistent API for fully and partially observable systems
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys, name=name)
        self.net = nn.Identity()

    def features(self, data):
        return data['Y_dp'][0]

    def reg_error(self):
        return torch.tensor(0.0)


























#############################################################################


###     KOOPMAN DYNAMICS FUNCTIONS


##############################################################################


from neuromancer.component import Component
import torch.nn.functional as F





## KOOPMAN DYNAMICS MODEL


#partitioned into three components

# map from the state_space to the latent space

# update map in the latent space

# map from the latent space to the state space


###################################################################################
####################################################################################









class KoopmanUpdateModel(Component):
    DEFAULT_INPUT_KEYS = ["x0","Y_phif", "Uf", "Df"] 
    DEFAULT_OUTPUT_KEYS = ["X_pred","X_true", "reg_error"]

    OPTIONAL_INPUT_KEYS = ["Uf", "Df"]
    OPTIONAL_OUTPUT_KEYS = ["fE"]
    
    _ALL_INPUTS = DEFAULT_INPUT_KEYS + OPTIONAL_INPUT_KEYS
    _ALL_OUTPUTS = DEFAULT_OUTPUT_KEYS + OPTIONAL_OUTPUT_KEYS

    def __init__(self, fxud,fpsi, fe = None, residual=False, name='KO_UM',
                 input_keys=dict()):
        """
        
            
        :param fxud: (nn.Module) State transition function
        :param fpsi: (nn.Module) Map from state space to space of observables

        :param residual: (bool) Whether to make recurrence in state space model residual
        :param name: (str) Name for tracking output
        :param input_keys: (dict {str: str}) Mapping canonical expected input keys to alternate names
        """
        
        input_keys = input_keys or KoopmanUpdateModel.DEFAULT_INPUT_KEYS
        input_keys = KoopmanUpdateModel.add_optional_inputs(
            [
                k for k in self.OPTIONAL_INPUT_KEYS
                if k in input_keys or (isinstance(input_keys, dict) and k in input_keys.values())
            ],
            remapping=input_keys,
        )

        output_keys = KoopmanUpdateModel.add_optional_outputs(
            [x for x, c in zip(self.OPTIONAL_OUTPUT_KEYS, [fe]) if c is not None]
        )

        super().__init__(
            input_keys,
            output_keys,
            name,
        )

        
        self.fxud = fxud
        
        self.fpsi = fpsi
        
        
        self.residual = residual

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        x_in, y_out, u_in, d_in = self.DEFAULT_INPUT_KEYS
       
        nsteps = data[y_out].shape[0]
        X, X_true = [], []
        
        
        psi = data[x_in]
        
        for i in range(nsteps):
            psi_prev = psi
            psi = self.fxud(psi,data[u_in][i],data[d_in][i])
            if self.residual:
                psi += psi_prev
            X.append(psi)
            xtrue = data[y_out][i]
            xtrue = self.fpsi(xtrue)
            X_true.append(xtrue)
        output = dict()
        
        for tensor_list, name in zip([X, X_true],
                                     ['X_pred', 'X_true']):
            if tensor_list:
                output[name] = torch.stack(tensor_list)
        output['reg_error'] = self.reg_error()
        return output

        

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])















class StatetoObservableMap(Component):
    DEFAULT_INPUT_KEYS = ["x0"] 
    DEFAULT_OUTPUT_KEYS = ["psi_0", "reg_error"]

    OPTIONAL_INPUT_KEYS = ["Up", "Dp"]
    OPTIONAL_OUTPUT_KEYS = ["fE"]
    
    _ALL_INPUTS = DEFAULT_INPUT_KEYS + OPTIONAL_INPUT_KEYS
    _ALL_OUTPUTS = DEFAULT_OUTPUT_KEYS + OPTIONAL_OUTPUT_KEYS

    def __init__(self, fpsi, fe = None, residual=False, name='StOM',
                 input_keys=dict()):
               
        input_keys = input_keys or StatetoObservableMap.DEFAULT_INPUT_KEYS
        input_keys = StatetoObservableMap.add_optional_inputs(
            [
                k for k in self.OPTIONAL_INPUT_KEYS
                if k in input_keys or (isinstance(input_keys, dict) and k in input_keys.values())
            ],
            remapping=input_keys,
        )

        output_keys = StatetoObservableMap.add_optional_outputs(
            [x for x, c in zip(self.OPTIONAL_OUTPUT_KEYS, [fe]) if c is not None]
        )

        super().__init__(
            input_keys,
            output_keys,
            name,
        )

        
        self.fpsi = fpsi
        
        
        self.residual = residual

      
    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        #x_in, y_out, u_in, d_in = self.input_keys
        [x_in] = self.DEFAULT_INPUT_KEYS

        x = data[x_in]
        #print(x.shape)
        psi = self.fpsi(x)
        
        output = dict()
        
        for tensor, name in zip([psi],["psi_0"]):
                output[name] = tensor
                
                
        output['reg_error'] = self.reg_error()
        return output



    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])















class State_to_Obs_map(nn.Module):
    def __init__(self,state_dim,hidden_dim,nn_obs_dim,st_inclv_idx):
        super().__init__()
        self.state_dim = state_dim
        self.nn_obs_dim = nn_obs_dim
        self.hidden_dim = hidden_dim
        self.L1 = nn.Linear(self.state_dim,self.hidden_dim)
        self.L2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.L3 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.L4 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.L5 = nn.Linear(self.hidden_dim,self.nn_obs_dim)
        self.st_inclv_idx = st_inclv_idx

    def forward(self, x):
        phi = F.elu(self.L1(x))
        phi = F.elu(self.L2(phi))
        phi = F.elu(self.L3(phi))
        phi = F.elu(self.L4(phi))
        phi = F.elu(self.L5(phi))
        
        cat_dim = 1
        z = torch.reshape( x[:,self.st_inclv_idx],(x.shape[0],len(self.st_inclv_idx)) )
        psi = torch.cat( (z,phi),dim=cat_dim ) 
        return psi






class KO_state_transition(nn.Module):
    def __init__(self,obs_dim,tot_input_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.tot_input_dim = tot_input_dim
        self.KO = nn.Linear(self.obs_dim,self.obs_dim,bias=False)
        self.B = nn.Linear(self.tot_input_dim,self.obs_dim,bias = False)
        self.in_features = obs_dim
        
    def forward(self,x,u,d):
        ud_in = torch.cat((u,d),dim=1)
        Bu = self.B(ud_in)
        return self.KO(x-Bu) + Bu
        



class Domain_aware_KO_state_transition(nn.Module):
    def __init__(self,obs_dim,tot_input_dim,device):
        super().__init__()
        self.obs_dim = obs_dim
        self.tot_input_dim = tot_input_dim
        self.device = device
        self.KO = PowerBndLinear(self.obs_dim,self.obs_dim,self.device)
        self.bs_t = -torch.ones(1,self.tot_input_dim)
        self.bs_t = self.bs_t.to(self.device)
        self.bs_z = torch.zeros(2,self.tot_input_dim)
        self.bs_z = self.bs_z.to(self.device)
        self.register_buffer('B_top',torch.tensor(np.concatenate( (np.zeros((2,8)),np.reshape([-1,-1,0,-1,0,-1,0,-1],(1,8)) ) ),dtype=torch.float32) )
        self.B_top.to(self.device)
        self.B_bot_init = torch.rand(self.obs_dim - 3,self.tot_input_dim)
        self.B_bot = torch.nn.parameter.Parameter(self.B_bot_init,requires_grad=True)
        self.B_bot.to(self.device)
        self.in_features = obs_dim
        
    def forward(self,x,u,d):
        ud_in = torch.cat((u,d),dim=1)
        B = torch.cat((self.B_top,self.B_bot))
        Bu = torch.matmul(B, torch.transpose(ud_in,0,1))
        Bu = torch.transpose(Bu,0,1)
        return self.KO(x-Bu) + Bu
    
    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])


    
    

class Obs_to_state_map(nn.Module):
    def __init__(self,return_idx):
        super().__init__()
        self.return_idx = return_idx
        self.out_features = len(return_idx)

    def forward(self, x):
        return x[:,self.return_idx]





###################################################################################
####################################################################################
###################################################################################
####################################################################################

###################################################################################
####################################################################################








#############################################################################


###     DED CONTROL POLICY FUNCTIONS


##############################################################################





from neuromancer.policies import MLPPolicy









class Direct_Conv_policy_fn(nn.Module):
    def __init__(self,input_dim,output_dim,h_dim,parm_dim,nsteps,ndsteps):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.h_dim = h_dim
        self.parm_dim = parm_dim
        self.nsteps = nsteps
        self.ndsteps = ndsteps
        
        self.u1_ub = 2.445
        self.u1_lb = .815
        
        self.u2_ub = 1.275
        self.u2_lb = .425
        
        self.encoding = nn.LSTM(input_size=input_dim,hidden_size=h_dim,batch_first=True)
        
        self.Conv_ReLU_stack = nn.Sequential(
            nn.Conv1d(self.input_dim + self.parm_dim,self.h_dim,5,stride = 1,padding='same'),
            nn.ReLU(),
            nn.Conv1d(self.h_dim,self.h_dim,10,stride =1, padding = 'same'),
            nn.ReLU(), 
            nn.Conv1d(self.h_dim,self.output_dim,5,stride=1, padding = 'same')
        )
        
        self.decoding = nn.LSTM(input_size=h_dim,hidden_size = output_dim , batch_first=False)
        
        self.upsampling = nn.Upsample(nsteps,mode='linear')

    def forward(self, x_in,u_in,d_in,c_in):
        
        c_dat = c_in[0,:,:]
        
        up_dat=u_in[0,:,:]
        
        parm_dat = torch.cat((x_in,c_dat,up_dat),axis=1)
        
        parm_dims = parm_dat.shape
        
        parm_dat = torch.reshape(parm_dat, (parm_dims[0],parm_dims[1],1))
        
        parm_seq = torch.tile(parm_dat,(1,1,self.ndsteps))
        
        
        #Downsample input data
        Dn = torch.transpose(d_in,0,1)
        Dn = torch.transpose(Dn,1,2)
        Dd = F.interpolate(Dn,self.ndsteps)
        
        
        seq_dat = torch.cat((Dd,parm_seq),dim=1)
        
        cntrl_seq = self.Conv_ReLU_stack(seq_dat) #predict downsampled control
        
        x_out = F.tanh(cntrl_seq)
        
        x_out = torch.transpose(x_out,1,2)
        x_out = torch.transpose(x_out,0,1)
        
        u1 = self.u1_lb + .5*(1+x_out[:,:,0])*(self.u1_ub - self.u1_lb)
        u2 = self.u2_lb + .5*(1+x_out[:,:,1])*(self.u2_ub - self.u2_lb)
        
        
               
        x_out = torch.stack((u1,u2),dim=2)
        
        
        x_out = torch.transpose(x_out,0,1)
        x_out = torch.transpose(x_out,1,2)
        
        x_out = self.upsampling(x_out) #upsample the control to original timestep
        x_out = torch.transpose(x_out,1,2)
        x_out = torch.transpose(x_out,0,1)
        
        
        return x_out







class Forecast_Policy(Component):
    DEFAULT_INPUT_KEYS = ["x0","Uf", "Df","coefsf"] 
    DEFAULT_OUTPUT_KEYS = ["U_pred","reg_error"]

    OPTIONAL_INPUT_KEYS = ["Up"]
    OPTIONAL_OUTPUT_KEYS = ["fE"]
    
    _ALL_INPUTS = DEFAULT_INPUT_KEYS + OPTIONAL_INPUT_KEYS
    _ALL_OUTPUTS = DEFAULT_OUTPUT_KEYS + OPTIONAL_OUTPUT_KEYS

    def __init__(self, f_policy, fe = None, residual=False, name='policy',
                 input_keys=dict()):
               
        input_keys = input_keys or Forecast_Policy.DEFAULT_INPUT_KEYS
        input_keys = Forecast_Policy.add_optional_inputs(
            [
                k for k in self.OPTIONAL_INPUT_KEYS
                if k in input_keys or (isinstance(input_keys, dict) and k in input_keys.values())
            ],
            remapping=input_keys,
        )

        output_keys = Forecast_Policy.add_optional_outputs(
            [x for x, c in zip(self.OPTIONAL_OUTPUT_KEYS, [fe]) if c is not None]
        )

        super().__init__(
            input_keys,
            output_keys,
            name,
        )

        
        self.f_policy = f_policy
        
        
        self.residual = residual

       

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        x_in, u_in, d_in, c_in = self.DEFAULT_INPUT_KEYS
               
        x_in = data[x_in]
        u_in = data[u_in]
        d_in = data[d_in]
        c_in = data[c_in]

        u_policy = self.f_policy(x_in,u_in,d_in,c_in)
        output = dict()
        
        
        for tensor, name in zip([u_policy],["U_pred"]):
                output[name] = tensor
                
        output['reg_error'] = self.reg_error()
        
        return output
    
    
    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])


















###################################################################################
####################################################################################


###################################################################################
####################################################################################




# SLIM FUNCTIONS


###################################################################################
####################################################################################





from slim.linear import LinearBase





class PowerBndLinear(LinearBase):
    """
    Linear map with constrained maximum magnitude eigenvalues via Power method.
   
    """
    def __init__(self, insize, outsize,device, bias=False, **kwargs):
       
        super().__init__(insize, outsize, bias=bias)
        self.in_features = insize
        self.out_features = outsize
        
        self.W = nn.Parameter(torch.eye(insize))
        self.device = device


    def eig_v_estimate(self):
        n_iterates = 200
        
        a = torch.normal(0,1,(self.in_features,1))
        a = a.to(self.device)
        b = torch.normal(0,1,(self.in_features,1))
        b = b.to(self.device)

         
        with torch.no_grad():
            for i in range(n_iterates):
                a = torch.mm(self.W,a)
                b = torch.mm(self.W,b)
                a_ib_nrm = torch.sqrt( torch.mm(torch.t(a),a) + torch.mm(torch.t(b),b)   )
                a = (1/a_ib_nrm)*a
                b = (1/a_ib_nrm)*b

        return [a,b]
        

    def reg_error(self):
        """
        Regularization error enforces upper bound on magnitude of eigenvalues
        """
                    
            
        [a,b] = self.eig_v_estimate()    
        a_ib_nsq = torch.mm(torch.t(a),a) + torch.mm(torch.t(b),b)
        v = torch.mm(torch.t(torch.mm(self.W,a)),torch.mm(self.W,a)) + torch.mm(torch.t(torch.mm(self.W,b)),torch.mm(self.W,b))
        v = v/a_ib_nsq
        v = v[0][0]
        return torch.nn.functional.relu( v - (1-1e-5)  )
        
    

    def effective_W(self):
        
        return self.W









