"""
TODO: eval_metric - evaluate closed loop metric based on the simulation results
# use the same interface for objectives as for the problem via _calculate_loss. if code is changed in problem possible mismatch
TODO: overwrite past after n-steps, continuously in first n steps. In initial simulation period first n-steps are treated as 0s when only first needs to be a 0 vector

"""

import torch
import torch.nn as nn
import numpy as np

from psl import EmulatorBase

from neuromancer.dataset import normalize_01 as normalize, denormalize_01 as min_max_denorm
from neuromancer.problem import Problem
from neuromancer.trainer import move_batch_to_device


class Simulator:
    def __init__(
        self,
        model: Problem,
        train_data,
        dev_data,
        test_data,
        emulator: EmulatorBase = None,
        eval_sim=True,
        device="cpu",
    ):
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.emulator = emulator
        self.eval_sim = eval_sim
        self.device = device

    def dev_eval(self):
        if self.eval_sim:
            dev_loop_output = self.model(self.dev_data)
        else:
            dev_loop_output = dict()
        return dev_loop_output

    def test_eval(self):
        all_output = dict()
        for data, dname in zip([self.train_data, self.dev_data, self.test_data],
                               ['train', 'dev', 'test']):
            all_output = {
                **all_output,
                **self.simulate(data)
            }
        return all_output

    def simulate(self, data):
        pass


class OpenLoopSimulator(Simulator):
    def __init__(
        self,
        model: Problem,
        train_data,
        dev_data,
        test_data,
        emulator: EmulatorBase = None,
        eval_sim=True,
        device="cpu",
    ):
        super().__init__(
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            emulator=emulator,
            eval_sim=eval_sim,
            device=device,
        )

    def simulate(self, data):
        return self.model(move_batch_to_device(data, self.device))


class MHOpenLoopSimulator(Simulator):
    """
    moving horizon open loop simulator
    """
    def __init__(self, model: Problem, dataset, emulator: [EmulatorBase, nn.Module] = None,
                 eval_sim=True, device="cpu"):
        super().__init__(model=model, dataset=dataset, emulator=emulator, eval_sim=eval_sim, device=device)

    def horizon_data(self, data, i):
        """
        will work with open loop dataset
        :param data:
        :param i: i-th time step
        :return:
        """
        step_data = {}
        for k, v in data.items():
            step_data[k] = v[i:self.dataset.nsteps+i, :, :]
        step_data["name"] = data["name"]
        return step_data

    def simulate(self, data):
        Y, X, L = [], [], []
        Yp, Yf, Xp, Xf = [], [], [], []
        data = move_batch_to_device(data, self.device)
        yN = data['Yp'][:self.dataset.nsteps, :, :]
        nsim = data['Yp'].shape[0]
        for i in range(nsim-self.nsteps):
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
    def __init__(
        self,
        model: Problem,
        train_data,
        dev_data,
        test_data,
        emulator: EmulatorBase = None,
        eval_sim=True,
        stack=False,
        device="cpu",
    ):
        super().__init__(
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            emulator=emulator,
            eval_sim=eval_sim,
            device=device,
        )
        self.stack = stack

    def agg(self, outputs):
        agg_outputs = dict()
        for k, v in outputs[0].items():
            agg_outputs[k] = []

        for data in outputs:
            for k in data:
                agg_outputs[k].append(data[k])
        for k in agg_outputs:
            if type(agg_outputs[k][0]) == str: continue
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
            d = move_batch_to_device(d, self.device)
            outputs.append(self.model(d))
        return self.agg(outputs)

    def dev_eval(self):
        if self.eval_sim:
            dev_loop_output = self.simulate(move_batch_to_device(self.dev_data, self.device))
        else:
            dev_loop_output = dict()
        return dev_loop_output


class ClosedLoopSimulator(Simulator):
    def __init__(
        self,
        model: Problem,
        policy: nn.Module,
        emulator: [EmulatorBase, nn.Module],
        train_data,
        dev_data,
        test_data,
        norm_stats=None,
        eval_sim=True,
        device="cpu",
    ):
        super().__init__(
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            emulator=emulator,
            eval_sim=eval_sim,
            device=device,
        )
        assert isinstance(emulator, EmulatorBase) or isinstance(emulator,  nn.Module), \
            f'{type(emulator)} is not EmulatorBase or nn.Module.'
        self.emulator = emulator
        self.policy = policy
        self.ninit = 0
        self.nsim = self.train_data.nsteps
        self.train_loop = self.train_data.get_full_sequence()
        self.train_data = next(iter(self.train_data))
        self.dev_loop = self.dev_data.get_full_sequence()
        self.dev_data = next(iter(self.dev_data))
        self.test_loop = self.test_data.get_full_sequence()
        self.test_data = next(iter(self.test_data))
        self.norm_stats = norm_stats or {}
        if isinstance(emulator, EmulatorBase):
            self.x0 = self.emulator.x0
        elif isinstance(emulator, nn.Module):
            self.x0 = torch.zeros([1, self.emulator.nx])

        self.device = device

    def select_step_data(self, data, i):
        """
        for closed loop we want to pick i-th batch of the nstep data nstep_data[item][:, k, :]

        :param data:
        :param i:
        :return:
        """
        step_data = {}
        for k, v in data.items():
            if k == "name": continue
            step_data[k] = v[:, i, :].reshape(v.shape[0], 1, v.shape[2])

        step_data["name"] = data["name"]
        return step_data

    def eval_metric(self):
        pass

    def dev_eval(self):
        if self.eval_sim:
            dev_loop_output = self.model(self.dev_loop)
        else:
            dev_loop_output = dict()
        return dev_loop_output

    def test_eval(self):
        all_output = dict()
        for data, dname in zip([self.train_data, self.dev_data, self.test_data],
                               ['train', 'dev', 'test']):
            all_output = {
                **all_output,
                **self.simulate(data)
            }
        return all_output


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
                if 'Y' in self.norm_stats:
                    r = min_max_denorm(r, self.norm_stats['Ymin'],
                                       self.norm_stats['Ymax']) if r is not None else None
                    ymin = min_max_denorm(ymin, self.norm_stats['Ymin'],
                                       self.norm_stats['Ymax']) if ymin is not None else None
                    ymax = min_max_denorm(ymax, self.norm_stats['Ymin'],
                                          self.norm_stats['Ymax']) if ymax is not None else None
                if 'U' in self.norm_stats:
                    umin = min_max_denorm(umin, self.norm_stats['Umin'],
                                       self.norm_stats['Umax']) if umin is not None else None
                    umax = min_max_denorm(umax, self.norm_stats['Umin'],
                                          self.norm_stats['Umax']) if umax is not None else None
                if 'D' in self.norm_stats:
                    d = min_max_denorm(d, self.norm_stats['Dmin'],
                                       self.norm_stats['Dmax']) if d is not None else None
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
                    if 'Y' in self.norm_stats:
                        y = min_max_denorm(y, self.norm_stats['Ymin'],
                                           self.norm_stats['Ymax']) if y is not None else None
                # update u and y trajectory history
                if len(Y) > self.nsteps:
                    if 'Y' in self.norm_stats:
                        Yp_np, _, _ = normalize(np.concatenate(Y[-self.nsteps:]),
                                                             Mmin=self.norm_stats['Ymin'],
                                                             Mmax=self.norm_stats['Ymax'])
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
            if 'U' in self.norm_stats:
                u = min_max_denorm(uopt.cpu().numpy(), self.norm_stats['Umin'],
                                   self.norm_stats['Umax'])
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
