"""

"""

import torch
import torch.nn as nn

from neuromancer.psl.emulator import EmulatorBase

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
            dev_loop_output = self.simulate(self.dev_data)
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
            step_data[k] = v[:, i:self.dataset.nsteps+i, :]
        step_data["name"] = data["name"]
        return step_data

    def simulate(self, data):
        Y, X, L = [], [], []
        Yp, Yf, Xp, Xf = [], [], [], []
        data = move_batch_to_device(data, self.device)
        yN = data['Yp'][:, :self.dataset.nsteps, :]
        nsim = data['Yp'].shape[0]
        for i in range(nsim-self.nsteps):
            step_data = self.horizon_data(data, i)
            step_data['Yp'] = yN
            step_output = self.model(step_data)
            # outputs
            y_key = [k for k in step_output.keys() if 'Y_pred' in k][0]
            y = step_output[y_key][:, 0:1, :]
            Y.append(y)
            yN = torch.cat([yN, y], dim=1)[:, 1:, :]
            yp_key = [k for k in step_output.keys() if 'Yp' in k][0]
            yp = step_output[yp_key][:, 0:1, :]
            Yp.append(yp)
            yf_key = [k for k in step_output.keys() if 'Yf' in k][0]
            yf = step_output[yf_key][:, 0:1, :]
            Yf.append(yf)
            # states
            x_key = [k for k in step_output.keys() if 'X_pred' in k][0]
            x = step_output[x_key][:, 0:1, :]
            X.append(x)
            xp_key = [k for k in step_output.keys() if 'Xp' in k][0]
            xp = step_output[xp_key][:, 0:1, :]
            Xp.append(xp)
            xf_key = [k for k in step_output.keys() if 'Xf' in k][0]
            xf = step_output[xf_key][:, 0:1, :]
            Xf.append(xf)
            loss_keys = [k for k in step_output.keys() if 'loss' in k][0]
            loss_item = step_output[loss_keys]
            L.append(loss_item)
        output = dict()
        for tensor_list, name in zip([X, Y, L, Yp, Yf, Xp, Xf],
                                     [x_key, y_key, loss_keys,
                                      yp_key, yf_key,
                                      xp_key, xf_key]):
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
                agg_outputs[k] = torch.mean(torch.stack(agg_outputs[k]).float())
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
            dev_loop_output = self.simulate(self.dev_data)
        else:
            dev_loop_output = dict()
        return dev_loop_output


class ClosedLoopSimulator:
    def __init__(
            self,
            sim_data,
            policy: nn.Module,
            system_model: nn.Module,
            estimator: nn.Module = None,
            emulator: EmulatorBase = None,
            emulator_output_keys=None,
            emulator_input_keys=None,
            nsim=None,
    ):
        """

        :param sim_data:
        :param policy: nn.Module
        :param system_model: nn.Module
        :param estimator: nn.Module
        :param emulator: psl.EmulatorBase
        """
        assert isinstance(policy, nn.Module), \
            f'{type(policy)} is not nn.Module.'
        assert isinstance(system_model, nn.Module), \
            f'{type(system_model)} is not nn.Module.'
        if emulator is not None:
            assert isinstance(emulator, EmulatorBase), \
                f'{type(emulator)} is not EmulatorBase.'
        if estimator is not None:
            assert isinstance(estimator, nn.Module), \
                f'{type(estimator)} is not nn.Module.'
        self.sim_data = sim_data
        self.system_model = system_model
        self.policy = policy
        self.estimator = estimator
        self.emulator = emulator
        # ['Y_pred', 'X_pred']  - must be always in this order
        self.emulator_output_keys = emulator_output_keys
        # ['Uf', 'Df', 'x0']   - must be always in this order
        self.emulator_input_keys = emulator_input_keys
        key = list(sim_data.keys())[0]
        if nsim is None:
            self.nsim = sim_data[key].shape[1] - estimator.window_size - policy.nsteps
        else:
            self.nsim = nsim

    def test_eval(self):
        sim_out_model = self.simulate(nsim=self.nsim)
        if self.emulator is not None:
            sim_out_emul = self.simulate(nsim=self.nsim, use_emulator=True)
        else:
            sim_out_emul = None
        return sim_out_model, sim_out_emul

    def step_data_policy(self, data, k):
        """
        get one step input data for control policy
        :param data:
        :param k:
        :return:
        """
        step_data = {}
        for key in self.policy.input_keys:
            if key in data.keys():
                step_data[key] = data[key][:, k - self.policy.nsteps:k, :]
        return step_data

    def step_data_estimator(self, data, k):
        """
        get one step input data for state estimator
        :param data:
        :param k:
        :return:
        """
        step_data = {}
        for key in self.estimator.input_keys:
            if key in data.keys():
                step_data[key] = data[key][:, k-self.policy.nsteps:k, :]
        return step_data

    def step_data_model(self, data, k, input_keys):
        """
        get one step input data for system model and emulator
        :param data:
        :param k:
        :return:
        """
        step_data = {}
        for key in input_keys:
            if key in data.keys():
                step_data[key] = data[key][:, k, :]
        return step_data

    def step_emulator(self, step_data):
        """

        :param step_data:
        :return:
        """
        # load data U, D, x0 and transform tensors to numpy arrays
        U = step_data[self.emulator_input_keys[0]][0, :, :].detach().numpy() if len(self.emulator_input_keys) >= 1 else None
        D = step_data[self.emulator_input_keys[1]][0, :, :].detach().numpy() if len(self.emulator_input_keys) >= 2 else None
        x0 = step_data[self.emulator_input_keys[2]][0, :] if len(self.emulator_input_keys) >= 3 else None
        # simulate 1 step ahead
        emul_out = self.emulator.simulate(nsim=1, U=U, D=D, x0=x0)
        # transform to dict of torch tensors
        emul_step = {}
        emul_step[self.emulator_output_keys[0]] = torch.tensor([emul_out['Y']])
        emul_step[self.emulator_output_keys[1]] = torch.tensor([emul_out['X']])
        return emul_step

    def update_sim_data(self, sim_data, step_data, k):
        """
        update simulation data dictionary with new predicted values from step forward
        :param sim_data:
        :param step_data:
        :param k:
        :return:
        """
        sim_data['Yp'][:, k, :] = step_data['Y_pred_dynamics']
        sim_data['Up'][:, k, :] = step_data['U_pred_policy']
        return sim_data

    def rhc(self, policy_out):
        """
        Receding horizon control = select only first timestep of the control horizon
        :param policy_out:
        :return:
        """
        key = 'U_pred_policy'
        policy_out[key] = policy_out[key][:, 0:1, :]
        return policy_out

    def append_data(self, sim_data, step_data):
        for key in sim_data.keys():
            if key in step_data.keys():
                sim_data[key].append(step_data[key])
        return sim_data

    def simulate(self, nsim, use_emulator=False):
        cl_keys = self.estimator.output_keys + self.policy.output_keys + \
                  self.policy.input_keys
        if use_emulator:
            cl_keys = cl_keys+self.emulator_output_keys
        else:
            cl_keys = cl_keys+self.system_model.output_keys
        cl_keys = [k for k in cl_keys if not k.startswith('reg_error')]
        cl_data = {}
        for key in cl_keys:
            cl_data[key] = []

        # initial time index with offset determined by largest moving horizon window
        if self.estimator is not None and self.policy.nsteps < self.estimator.window_size:
            start_k = self.estimator.window_size
        else:
            start_k = self.policy.nsteps
        for k in range(start_k, start_k+nsim):

            # estimator step
            if self.estimator is not None:
                step_data = self.step_data_estimator(self.sim_data, k)
                estim_out = self.estimator(step_data)
            else:
                estim_out = {}

            # policy step
            policy_in = self.step_data_policy(self.sim_data, k)
            step_data = {**policy_in, **estim_out}
            policy_out = self.policy(step_data)     # calculate n-step ahead control
            policy_out = self.rhc(policy_out)       # apply reciding horizon control

            # model step
            if use_emulator:
                step_data = self.step_data_model(self.sim_data, k, self.emulator_input_keys)
                step_data = {**step_data, **estim_out, **policy_out}
                model_out = self.step_emulator(step_data)
            else:
                step_data = self.step_data_model(self.sim_data, k, self.system_model.input_keys)
                step_data = {**step_data, **estim_out, **policy_out}
                model_out = self.system_model(step_data)

            # closed-loop step
            cl_step_data = {**estim_out, **policy_in, **policy_out, **model_out}
            # update sim_data for next step
            self.sim_data = self.update_sim_data(self.sim_data, cl_step_data, k)

            # process batch data to have 2 dimensions: time x var. dim.
            for key in cl_step_data.keys():
                if len(cl_step_data[key].shape) == 3:
                    cl_step_data[key] = cl_step_data[key][:, 0, :]
            # if nstep ahead policy: select only each n-th step of policy keys for logging
            for key in self.policy.input_keys:
                cl_step_data[key] = cl_step_data[key][::self.policy.nsteps, :]

            # append closed-loop step to simulation data
            cl_data = self.append_data(cl_data, cl_step_data)
        # concatenate step data in a single tensor
        for key in cl_data.keys():
            cl_data[key] = torch.cat(cl_data[key])
        return cl_data
