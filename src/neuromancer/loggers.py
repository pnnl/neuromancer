"""

"""
import time
import os
import shutil
import torch
import dill
import numbers
import numpy as np


class BasicLogger:
    def __init__(self, args=None, savedir='test', verbosity=10,
                 stdout=('nstep_dev_loss', 'loop_dev_loss', 'best_loop_dev_loss',
                         'nstep_dev_ref_loss', 'loop_dev_ref_loss')):
        """
        :param args: (Namespace) returned by argparse.ArgumentParser.parse_args()
        :param savedir: (str) Folder to write results to.
        :param verbosity: (int) Print to stdout every verbosity epochs
        :param stdout: (list of str) Metrics to print to stdout. These should correspond to keys in the output dictionary of the Problem
        """
        os.makedirs(savedir, exist_ok=True)
        self.stdout = stdout
        self.savedir = savedir
        self.verbosity = verbosity
        self.start_time = time.time()
        self.step = 0
        self.args = args
        self.log_parameters()

    def log_parameters(self):
        """
        Print experiment parameters to stdout

        :param args: (Namespace) returned by argparse.ArgumentParser.parse_args()
        """
        print(self.args)

    def log_weights(self, model):
        """

        :param model: (nn.Module)
        :return: (int) The number of learnable parameters in the model
        """
        nweights = sum([i.numel() for i in list(model.parameters()) if i.requires_grad])
        print(f'Number of parameters: {nweights}')
        return nweights

    def log_metrics(self, output, step=None):
        """
        Print metrics to stdout.

        :param output: (dict {str: tensor}) Will only record 0d tensors (scalars)
        :param step: (int) Epoch of training
        """
        if step is None:
            step = self.step
        else:
            self.step = step
        if step % self.verbosity == 0:
            elapsed_time = time.time() - self.start_time
            entries = [f'epoch: {step}']
            for k, v in output.items():
                try:
                    if k in self.stdout:
                        entries.append(f'{k}: {v.item():.5f}')
                except (ValueError, AttributeError) as e:
                    pass
            entries.append(f'eltime: {elapsed_time: .5f}')
            print('\t'.join([e for e in entries if 'reg_error' not in e]))

    def log_artifacts(self, artifacts):
        """
        Stores artifacts created in training to disc.

        :param artifacts: (dict {str: Object})
        """
        for k, v in artifacts.items():
            savepath = os.path.join(self.savedir, k)
            torch.save(v, savepath, pickle_module=dill)

    def clean_up(self):
        pass

class LossLogger(BasicLogger):
    def __init__(self, args=None, savedir='test', verbosity=10,
                 stdout=('nstep_dev_loss', 'loop_dev_loss', 'best_loop_dev_loss',
                         'nstep_dev_ref_loss', 'loop_dev_ref_loss')):
        super().__init__(args, savedir, verbosity, stdout)
        self.losses = {'train': [], 'dev': [], 'test': []}  # Initialize losses dictionary

    def log_metrics(self, output, step=None):
        """
        Print metrics to stdout and store loss values.

        :param output: (dict {str: tensor}) Will only record 0d tensors (scalars)
        :param step: (int) Epoch of training
        """
        if step is None:
            step = self.step
        else:
            self.step = step
        if step % self.verbosity == 0:
            elapsed_time = time.time() - self.start_time 
            entries = [f'epoch: {step}']
            for k, v in output.items():
                try:
                    if k in self.stdout:
                        entries.append(f'{k}: {v.item():.5f}')
                        # Collect the loss values based on type
                        if 'loss' in k.lower():
                            if 'train' in k.lower():
                                self.losses['train'].append(v.item())
                            elif 'dev' in k.lower():
                                self.losses['dev'].append(v.item())
                            elif 'test' in k.lower():
                                self.losses['test'].append(v.item())
                except (ValueError, AttributeError) as e:
                    pass
            entries.append(f'eltime: {elapsed_time: .5f}')
            print('\t'.join([e for e in entries if 'reg_error' not in e]))

    def get_losses(self):
        """
        Returns a dictionary of recorded loss values for train, dev, and test.
        """
        return {k: v for k, v in self.losses.items() if v} 


class MLFlowLogger(BasicLogger):
    def __init__(self, args=None, savedir='test', verbosity=1, id=None,
                 stdout=('nstep_dev_loss','loop_dev_loss','best_loop_dev_loss',
                         'nstep_dev_ref_loss','loop_dev_ref_loss'),
                 logout=None):
        # Lazy import so module import works even if mlflow isn't installed
        try:
            import mlflow  # noqa: F401
        except Exception as e:
            raise ImportError(
                "MLFlowLogger requires mlflow. Install with "
                "`pip install neuromancer[tracking]` or `pip install mlflow>=2.12`."
            ) from e

        import mlflow  # use after we know it exists

        self._mlflow = mlflow
        self._mlflow.set_tracking_uri(args.location)
        self._mlflow.set_experiment(args.exp)
        self._mlflow.start_run(run_name=args.run, run_id=id)

        super().__init__(args=args, savedir=savedir, verbosity=verbosity, stdout=stdout)
        self.logout = logout

    def log_parameters(self):
        params = {k: getattr(self.args, k) for k in vars(self.args)}
        print({k: type(v) for k, v in params.items()})
        self._mlflow.log_params(params)

    def log_weights(self, model):
        nweights = super().log_weights(model)
        self._mlflow.log_metric('nparams', float(nweights))

    def log_metrics(self, output, step=0):
        super().log_metrics(output, step)
        keys = set(output.keys())
        if self.logout is not None:
            keys = {k for k in keys if any(p in k for p in self.logout)}
        for k in keys:
            v = output[k]
            if isinstance(v, torch.Tensor) and torch.numel(v) == 1:
                self._mlflow.log_metric(k, v.item())
            elif isinstance(v, np.ndarray) and v.size == 1:
                self._mlflow.log_metric(k, float(v))
            elif isinstance(v, numbers.Number):
                self._mlflow.log_metric(k, v)

    def log_artifacts(self, artifacts=dict()):
        super().log_artifacts(artifacts)
        self._mlflow.log_artifacts(self.savedir)

    def clean_up(self):
        shutil.rmtree(self.savedir)
        self._mlflow.end_run()



