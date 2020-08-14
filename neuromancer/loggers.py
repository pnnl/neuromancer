"""

"""
# python base imports
import time
import os

# machine learning/data science imports
import mlflow
# import wandb
import torch
import dill


class BasicLogger:
    def __init__(self, args=None, savedir='test', verbosity=10,
                 stdout=('nstep_dev_loss', 'loop_dev_loss', 'best_loop_dev_loss',
                         'nstep_dev_ref_loss', 'loop_dev_ref_loss')):
        """
        :param args: Namespace returned by argparse.ArgumentParser.parse_args()
        :param savedir: Folder to write results to.
        :param verbosity: Print to stdout every verbosity epochs
        :param stdout: Metrics to print to stdout. These should correspond to keys in the output dictionary of the Problem
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
        Pring experiment parameters to stdout

        :param args: Namespace returned by argparse.ArgumentParser.parse_args()
        """
        print(self.args)

    def log_weights(self, model):
        """

        :param model: nn.Module
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

        :param artifacts: dict {str: Object}
        """
        for k, v in artifacts.items():
            savepath = os.path.join(self.savedir, k)
            torch.save(v, savepath, pickle_module=dill)

    def clean_up(self):
        pass


class MLFlowLogger(BasicLogger):
    def __init__(self, args=None, savedir='test', verbosity=1, id=None,
                 stdout=('nstep_dev_loss', 'loop_dev_loss', 'best_loop_dev_loss',
                         'nstep_dev_ref_loss', 'loop_dev_ref_loss')):
        """

        :param args: Namespace returned by argparse.ArgumentParser.parse_args()
        :param savedir: Unique folder name to temporarily save artifacts
        :param verbosity: How often to print to stdout
        :param stdout: What variables to print to stdout
        """
        mlflow.set_tracking_uri(args.location)
        mlflow.set_experiment(args.exp)
        mlflow.start_run(run_name=args.run, run_id=id)
        super().__init__(args=args, savedir=savedir, verbosity=verbosity, stdout=stdout)

    def log_parameters(self):
        """
        Pring experiment parameters to stdout

        :param args: dict
        """
        params = {k: str(getattr(self.args, k)) for k in vars(self.args) if getattr(self.args, k)}
        mlflow.log_params(params)

    def log_weights(self, model):
        """

        :param model: nn.Module
        :return: (int) Number of learnable parameters in the model.
        """
        nweights = super().log_weights(model)
        mlflow.log_param('nparams',  nweights)

    def log_metrics(self, output, step=0):
        """
        Record metrics to mlflow

        :param output: dict {str: tensor} Will only record 0d tensors (scalars)
        :param step: Epoch of training
        """
        super().log_metrics(output, step)
        for k, v in output.items():
            try:
                mlflow.log_metric(k, v.item(), step=step)
            except:
                pass

    def log_artifacts(self, artifacts):
        """
        Stores artifacts created in training to mlflow.

        :param artifacts: dict {str: Object}
        """
        super().log_artifacts(artifacts)
        mlflow.log_artifacts(self.savedir)

    def clean_up(self):
        """
        Remove temporary files from file system
        """
        os.system(f'rm -rf {self.savedir}')
        mlflow.end_run()

