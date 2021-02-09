"""
TODO: eval_metric - evaluate closed loop metric based on the simulation results
# use the same interface for objectives as for the problem via _calculate_loss. if code is changed in problem possible mismatch
TODO: overwrite past after n-steps, continuously in first n steps. In initial simulation period first n-steps are treated as 0s when only first needs to be a 0 vector

"""

import torch
import torch.nn as nn
import numpy as np

from psl import EmulatorBase

from neuromancer.datasets import EmulatorDataset, FileDataset
from neuromancer.data.normalization import normalize_01 as normalize, denormalize_01 as min_max_denorm
from neuromancer.problem import Problem
from neuromancer.datasets import Dataset, DataDict


class Simulator:
    def __init__(self, model: Problem, dataset: Dataset, emulator: EmulatorBase = None, eval_sim=True):
        self.model = model
        self.dataset = dataset
        self.emulator = emulator
        self.eval_sim = eval_sim

    def dev_eval(self):
        if self.eval_sim:
            dev_loop_output = self.model(self.dataset.dev_loop)
        else:
            dev_loop_output = dict()
        return dev_loop_output

    def test_eval(self):
        all_output = dict()
        for data, dname in zip([self.dataset.train_loop, self.dataset.dev_loop, self.dataset.test_loop],
                               ['train', 'dev', 'test']):
            all_output = {**all_output, **self.simulate(data)}
        return all_output

    def simulate(self, data):
        pass


class OpenLoopSimulator(Simulator):
    def __init__(self, model: Problem, dataset: Dataset, emulator: [EmulatorBase, nn.Module] = None,
                 eval_sim=True):
        super().__init__(model=model, dataset=dataset, emulator=emulator, eval_sim=eval_sim)

    def simulate(self, data):
        return self.model(data)


class MHOpenLoopSimulator(Simulator):
    """
    moving horizon open loop simulator
    """
    def __init__(self, model: Problem, dataset: Dataset, emulator: [EmulatorBase, nn.Module] = None,
                 eval_sim=True):
        super().__init__(model=model, dataset=dataset, emulator=emulator, eval_sim=eval_sim)

    def horizon_data(self, data, i):
        """
        will work with open loop dataset
        :param data:
        :param i: i-th time step
        :return:
        """
        step_data = DataDict()
        for k, v in data.items():
            step_data[k] = v[i:self.dataset.nsteps+i, :, :]
        step_data.name = data.name
        return step_data

    def simulate(self, data):
        Y, X, L = [], [], []
        Yp, Yf, Xp, Xf = [], [], [], []
        yN = data['Yp'][:self.dataset.nsteps, :, :]
        nsim = data['Yp'].shape[0]
        for i in range(nsim-self.dataset.nsteps):
            step_data = self.horizon_data(data, i)
            step_data['Yp'] = yN
            step_output = self.model(step_data)
            # outputs
            y_key = [k for k in step_output.keys() if 'Y_pred' in k]
            y = step_output[y_key[0]][0].unsqueeze(0)
            Y.append(y)
            yN = torch.cat([yN, y])[1:]
            yp_key = [k for k in step_output.keys() if 'Yp' in k]
            yp = step_output[yp_key[0]][0].unsqueeze(0)
            Yp.append(yp)
            yf_key = [k for k in step_output.keys() if 'Yf' in k]
            yf = step_output[yf_key[0]][0].unsqueeze(0)
            Yf.append(yf)
            # states
            x_key = [k for k in step_output.keys() if 'X_pred' in k]
            x = step_output[x_key[0]][0].unsqueeze(0)
            X.append(x)
            xp_key = [k for k in step_output.keys() if 'Xp' in k]
            xp = step_output[xp_key[0]][0].unsqueeze(0)
            Xp.append(xp)
            xf_key = [k for k in step_output.keys() if 'Xf' in k]
            xf = step_output[xf_key[0]][0].unsqueeze(0)
            Xf.append(xf)
            loss_keys = [k for k in step_output.keys() if 'loss' in k]
            loss_item = step_output[loss_keys[0]]
            L.append(loss_item)
        output = dict()
        for tensor_list, name in zip([X, Y, L, Yp, Yf, Xp, Xf],
                                     [x_key[0], y_key[0], loss_keys[0],
                                      yp_key[0], yf_key[0],
                                      xp_key[0], xf_key[0]]):
            if tensor_list:
                output[name] = torch.stack(tensor_list)
        return {**data, **output}


class MultiSequenceOpenLoopSimulator(Simulator):
    def __init__(self, model: Problem, dataset: Dataset, emulator: [EmulatorBase, nn.Module] = None,
                 eval_sim=True, stack=False):
        super().__init__(model=model, dataset=dataset, emulator=emulator, eval_sim=eval_sim)
        self.stack = stack

    def agg(self, outputs):
        agg_outputs = dict()
        for k, v in outputs[0].items():
            agg_outputs[k] = []
        for data in outputs:
            for k in data:
                agg_outputs[k].append(data[k])
        for k in agg_outputs:
            if len(agg_outputs[k][0].shape) < 2:
                agg_outputs[k] = torch.mean(torch.stack(agg_outputs[k]))
            else:
                if self.stack:
                    agg_outputs[k] = torch.stack(agg_outputs[k])
                else:
                    agg_outputs[k] = torch.cat(agg_outputs[k])
        return agg_outputs

    def simulate(self, data):
        outputs = []
        for d in data:
            outputs.append(self.model(d))
        return self.agg(outputs)

    def dev_eval(self):
        if self.eval_sim:
            dev_loop_output = self.simulate(self.dataset.dev_loop)
        else:
            dev_loop_output = dict()
        return dev_loop_output


class ClosedLoopSimulator(Simulator):
    def __init__(self, model: Problem, dataset: Dataset, policy: nn.Module, emulator: [EmulatorBase, nn.Module] = None):
        super().__init__(model=model, dataset=dataset, emulator=emulator)
        assert isinstance(emulator, EmulatorBase) or isinstance(emulator,  nn.Module), \
            f'{type(emulator)} is not EmulatorBase or nn.Module.'
        self.emulator = emulator
        self.policy = policy
        self.ninit = 0
        self.nsim = self.dataset.nstep_data['Yf'].shape[1]
        if isinstance(emulator, EmulatorBase):
            self.x0 = self.emulator.x0
        elif isinstance(emulator, nn.Module):
            self.x0 = torch.zeros([1, self.emulator.nx])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def select_step_data(self, data, i):
        """
        for closed loop we want to pick i-th batch of the nstep data nstep_data[item][:, k, :]

        :param data:
        :param i:
        :return:
        """
        step_data = DataDict()
        for k, v in data.items():
            step_data[k] = v[:, i, :].reshape(v.shape[0], 1, v.shape[2])
        step_data.name = data.name
        return step_data

    def eval_metric(self):
        pass

    def simulate(self, data):
        """
          # CL eval steps:
          # 1, initialize emulator with initial conditions and initial input trajectories
          # 2, obtain i-th simulation step batch of nstep data generated by emulator
          # 3, forward pass i-th step batch through the model
          # 4, receding horizon - slect only first sampling instant of the i-th batch control actions
          # 5, simulate emulator with selected control actions and obtain new step batch - continue from step 2,

        :param data:
        :return:
        """
        self.model.eval()
        self.nsim = data['Yp'].shape[1]
        self.nsteps = data['Yp'].shape[0]
        Y, X, D, U, R = [], [], [], [], []  # emulator trajectories
        Ymin, Ymax, Umin, Umax = [], [], [], []
        Y_pred, X_pred, U_pred, U_opt = [], [], [], []   # model  trajectories
        for i in range(self.nsim):
            step_data = self.select_step_data(data, i)

            x = self.x0 if i == 0 else x
            if i > 0:
                # select current step disturbance, reference and constraints
                d = step_data['Df'][0].cpu().detach().numpy() if 'Df' in step_data else None
                r = step_data['Rf'][0].cpu().detach().numpy() if 'Rf' in step_data else None
                ymin = step_data['Y_minf'][0].cpu().detach().numpy() if 'Y_minf' in step_data else None
                ymax = step_data['Y_maxf'][0].cpu().detach().numpy() if 'Y_maxf' in step_data else None
                umin = step_data['U_minf'][0].cpu().detach().numpy() if 'U_minf' in step_data else None
                umax = step_data['U_maxf'][0].cpu().detach().numpy() if 'U_maxf' in step_data else None
                if 'Y' in self.dataset.norm:
                    r = min_max_denorm(r, self.dataset.min_max_norms['Ymin'],
                                       self.dataset.min_max_norms['Ymax']) if r is not None else None
                    ymin = min_max_denorm(ymin, self.dataset.min_max_norms['Ymin'],
                                       self.dataset.min_max_norms['Ymax']) if ymin is not None else None
                    ymax = min_max_denorm(ymax, self.dataset.min_max_norms['Ymin'],
                                          self.dataset.min_max_norms['Ymax']) if ymax is not None else None
                if 'U' in self.dataset.norm:
                    umin = min_max_denorm(umin, self.dataset.min_max_norms['Umin'],
                                       self.dataset.min_max_norms['Umax']) if umin is not None else None
                    umax = min_max_denorm(umax, self.dataset.min_max_norms['Umin'],
                                          self.dataset.min_max_norms['Umax']) if umax is not None else None
                if 'D' in self.dataset.norm:
                    d = min_max_denorm(d, self.dataset.min_max_norms['Dmin'],
                                       self.dataset.min_max_norms['Dmax']) if d is not None else None
                # simulate 1 step of the emulator model
                if isinstance(self.emulator, EmulatorBase):
                    x, y, _, _ = self.emulator.simulate(ninit=0, nsim=1, U=u, D=d, x0=x.flatten())
                elif isinstance(self.emulator, nn.Module):
                    step_data_0 = dict()
                    step_data_0['U_pred_policy'] = uopt.unsqueeze(0)
                    step_data_0['x0_estim'] = x
                    for k, v in step_data.items():
                        step_data_0[k] = v[0:1]

                    emulator_output = self.emulator(step_data_0)
                    x = emulator_output['X_pred_dynamics'][0]
                    y = emulator_output['Y_pred_dynamics'][0].cpu().detach().numpy()
                    if 'Y' in self.dataset.norm:
                        y = min_max_denorm(y, self.dataset.min_max_norms['Ymin'],
                                           self.dataset.min_max_norms['Ymax']) if y is not None else None
                # update u and y trajectory history
                if len(Y) > self.nsteps:
                    if 'Y' in self.dataset.norm:
                        Yp_np, _, _ = normalize(np.concatenate(Y[-self.nsteps:]),
                                                             Mmin=self.dataset.min_max_norms['Ymin'],
                                                             Mmax=self.dataset.min_max_norms['Ymax'])
                    else:
                        Yp_np = np.concatenate(Y[-self.nsteps:])
                    step_data['Yp'] = torch.tensor(np.concatenate(Yp_np, 0)).reshape(self.nsteps, 1, -1).float().to(self.device)
                if len(U_opt) > self.nsteps:
                    step_data['Up'] = torch.cat(U_opt[-self.nsteps:], dim=0).reshape(self.nsteps, 1, -1).float().to(self.device)

            # control policy model
            # step_output = self.model(step_data)
            step_output = self.policy(step_data)

            # # model trajectories
            # x_key = [k for k in step_output.keys() if 'X_pred' in k]
            # X_pred.append(step_output[x_key[0]])
            # y_key = [k for k in step_output.keys() if 'Y_pred' in k]
            # Y_pred.append(step_output[y_key[0]])
            u_key = [k for k in step_output.keys() if 'U_pred' in k]
            U_pred.append(step_output[u_key[0]])
            uopt = step_output[u_key[0]][0].detach()
            U_opt.append(uopt)

            # emulator trajectories
            if 'U' in self.dataset.norm:
                u = min_max_denorm(uopt.cpu().numpy(), self.dataset.min_max_norms['Umin'],
                                   self.dataset.min_max_norms['Umax'])
            else:
                u = uopt.cpu().numpy()
            if i > 0:
                U.append(u)
                Y.append(y)
                X.append(x) if isinstance(self.emulator, EmulatorBase) else X.append(x.detach().cpu().numpy())
                D.append(d) if d is not None else None
                R.append(r) if r is not None else None
                Ymin.append(ymin) if ymin is not None else None
                Ymax.append(ymax) if ymax is not None else None
                Umin.append(umin) if umin is not None else None
                Umax.append(umax) if umax is not None else None

        return {'Y': np.concatenate(Y, 0), 'X': np.concatenate(X, 0), 'U': np.concatenate(U, 0),
                'D': np.concatenate(D, 0) if D else None,
                'R': np.concatenate(R, 0) if R else None,
                'Ymin': np.concatenate(Ymin, 0) if Ymin else None,
                'Ymax': np.concatenate(Ymax, 0) if Ymax else None,
                'Umin': np.concatenate(Umin, 0) if Umin else None,
                'Umax': np.concatenate(Umax, 0) if Umax else None}


class CLSimulator(Simulator):
    """
    Closed loop simulation using pytorch nn.Module policy and dynamics models.
    """
    def __init__(self, model: Problem, dataset: Dataset, policy: nn.Module, emulator: [EmulatorBase, nn.Module] = None,
                 gt_emulator=None, eval_sim=True, Ki=0.1, integrator_steps=3, Kd=0.5, clamp_u=True, device='cpu'):
        super().__init__(model=model, dataset=dataset, emulator=emulator)
        self.policy = policy
        self.emulator = emulator
        self.gt_emulator = gt_emulator
        self.Ki = Ki
        self.Kd = Kd
        self.integrator_steps = integrator_steps
        self.eval_sim = eval_sim
        self.device = torch.device(device)
        self.clamp_u = clamp_u

    def psl_emulator(self, x, u, nstep_data):
        # print(x.shape)
        # print(self.gt_emulator.nx)

        # x = min_max_denorm(x.cpu().numpy(), self.dataset.norms['Ymin'], self.dataset.norms['Ymax'])
        u = min_max_denorm(u.cpu().numpy(), self.dataset.norms['Umin'], self.dataset.norms['Umax'])
        d = min_max_denorm(nstep_data['plant_Df'][0].cpu().numpy(), self.dataset.norms['plant_Dmin'], self.dataset.norms['plant_Dmax'])
        dta = self.gt_emulator.simulate(ninit=0, nsim=1, U=u, D=d, x0=x)
        x, y = dta['X'], dta['Y']
        # x, _, _ = normalize(x, Mmin=self.dataset.norms['Ymin'], Mmax=self.dataset.norms['Ymax'])
        y, _, _ = normalize(y, Mmin=self.dataset.norms['Ymin'], Mmax=self.dataset.norms['Ymax'])
        return (x[0], torch.tensor(y, dtype=torch.float32, device=self.device))

    def torch_emulator(self, x, u, nstep_data):
        # print(self.emulator.input_keys())
        one_step_data = {'U_pred_policy': u.unsqueeze(0),
                         'x0_estim': x,
                         **{k: v[0:1] for k, v in nstep_data.items()}}
        emulator_output = self.emulator(one_step_data)
        return emulator_output['X_pred_dynamics'][0], emulator_output['Y_pred_dynamics'][0]

    def select_step_data(self, data, i):
        """
        Creates an nstep snapshot of the data from a moving horizon dataset.

        :param data: (DataDict {str: torch.Tensor) A dictionary containing moving horizon data (nsteps X nbatches X dim) where each batch contains
                                                   nstep time series shifted 1 forward from the previous batch.
        :param i: (int) Time index
        :return: (DataDict {str: torch.Tensor) A moving horizon dataset with a single batch (nsteps X 1 X dim)
        """
        step_data = DataDict()
        for k, v in data.items():
            step_data[k] = v[:, i:i+1, :]
        step_data.name = data.name
        return step_data

    # def dev_eval(self):
    #     if self.eval_sim:
    #         dev_loop_output = self.simulate(self.dataset.dev_loop, sim=self.torch_emulator,
    #                                         name='dev_sim_', nx=self.emulator.nx)
    #     else:
    #         dev_loop_output = dict()
    #     return dev_loop_output

    def test_eval(self):
        output = {}
        output = {**output, **self.simulate(self.dataset.test_loop, sim=self.torch_emulator, name='model_', nx=self.emulator.nx)}
        output = {**output, **self.simulate(self.dataset.test_loop, sim=self.psl_emulator, name='plant_', nx=self.gt_emulator.nx)}
        return output

    def integrator(self, simulation, nsteps=32):
        y = torch.stack(simulation['Y'][-nsteps:])[:, :, 1:]
        r = torch.stack(simulation['R'][-nsteps:])
        return torch.clamp(self.Ki*torch.sum(r - y, dim=0), -5., 5.)

    def derivative_term(self, simulation):
        y = torch.stack(simulation['Y'][-2:])[:, :, 1:]
        r = torch.stack(simulation['R'][-2:])
        diff = r - y
        df = diff[1:2] - diff[0:1]
        return self.Kd*df.squeeze(0)

    def simulate(self, data, sim=None, name='model', nx=2, diff=False, df=False):
        """

        :param data: (DataDict {str: torch.Tensor) A dictionary containing moving horizon data (nsteps X nbatches X dim) where each batch contains
                                                   nstep time series shifted 1 forward from the previous batch.
        :return: (dict {str: np.array} A dictionary containing simulation time series (nsim X dim) with denormalized values
        """
        nsteps, nsim = data['Yp'].shape[0], data['Yp'].shape[1]
        simulation = {k: [] for k in ['Y', 'U', 'R', 'D', 'Ymin', 'Ymax', 'Umin', 'Umax']}
        with torch.no_grad():

            # Generate initial control action
            nstep_data = self.select_step_data(data, 0)
            policy_output = self.policy(nstep_data)
            u = policy_output['U_pred_policy'][0]
            simulation['U'].append(u)

            if name == 'plant_':
                x = self.gt_emulator.x0
            else:
                x = torch.zeros([1, nx])

            for i in range(1, nsim):
                nstep_data = self.select_step_data(data, i)

                # Simulate 1 step of dynamics with generated control input
                x, y = sim(x, u, nstep_data)

                # Update u and y trajectory history
                if len(simulation['Y']) > nsteps:
                    nstep_data['Yp'] = torch.stack(simulation['Y'][-nsteps:])
                    nstep_data['Up'] = torch.stack(simulation['U'][-nsteps:])

                # Generate sequence of control actions and keep first action
                policy_output = self.policy(nstep_data)
                u = policy_output['U_pred_policy'][0]

                if self.clamp_u:
                    # ad hoc clamping! works correctly only if U_min=0, U_max=1
                    u = torch.clamp(u, min=0.0, max=1.0)

                # if len(simulation['Y']) > self.integrator_steps and diff:
                #     u += self.integrator(simulation, nsteps=self.integrator_steps)
                # if len(simulation['Y']) > 2 and df:
                #     u += self.derivative_term(simulation)

                # Store simulation results
                for k, d in zip(['Y', 'U', 'R', 'D', 'Ymin', 'Ymax', 'Umin', 'Umax'],
                                [y, u, *[nstep_data[k][0] for k in ['Rf', 'Df', 'Y_minf', 'Y_maxf', 'U_minf', 'U_maxf']]]):
                    simulation[k].append(d)

            # Return completed simulation as dictionary of denormalized numpy arrays
            simulation = {
                name+k: min_max_denorm(torch.cat(v).cpu().numpy(), self.dataset.norms[f'{k}min'], self.dataset.norms[f'{k}max'])
                for k, v in simulation.items()}
            simulation[f'{name}error'] = np.mean(np.abs(simulation[f'{name}Y'] - simulation[f'{name}R']))
        return simulation


if __name__ == '__main__':

    systems = {'Reno_full': 'emulator'}
    for system, data_type in systems.items():
        if data_type == 'emulator':
            dataset = EmulatorDataset(system)
        elif data_type == 'datafile':
            dataset = FileDataset(system)
    nsim, ny = dataset.data['Y'].shape
    nu = dataset.data['U'].shape[1]
