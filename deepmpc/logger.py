"""
# TODO: Finish wandb logger
"""
import mlflow
# import wandb
import time
import os
import torch
import dill


class BasicLogger:
    def __init__(self, savedir='test', verbosity=10):
        os.system(f'mkdir {savedir}')
        self.savedir = savedir
        self.verbosity = verbosity
        self.start_time = time.time()
        self.step = 0

    def log_weights(self, model):
        nweights = sum([i.numel() for i in list(model.parameters()) if i.requires_grad])
        print(f'Number of parameters: {nweights}')
        return nweights

    def log_metrics(self, output, step=None):
        if step is None:
            step = self.step
        else:
            self.step = step
        if step % self.verbosity == 0:
            elapsed_time = time.time() - self.start_time
            entries = [f'epoch: {step}']
            for k, v in output.items():
                try:
                    entries.append(f'{k}: {v.item():.5f}')
                except ValueError:
                    pass
            entries.append(f'eltime: {elapsed_time: .5f}')
            print('\t'.join([e for e in entries if 'reg_error' not in e]))

    def log_artifacts(self, artifacts):
        for k, v in artifacts.items():
            savepath = os.path.join(self.savedir, k)
            torch.save(v, savepath, pickle_module=dill)

    def clean_up(self):
        pass


# class WandBLogger(BasicLogger):
#     def __init__(self, savedir, verbosity):
#         super().__init__(savedir, verbosity)
#
#     def log_metrics(self, output, step):
#         super().log_metrics(output, step)
#         for k, v in output:
#             if isinstance(v.item(), numbers.Number):
#                 wandb.log({k: v.item()}, step=step)
#
#     def log_weights(self, model):
#         nweights = super().log_weights(model)
#         wandb.config({'nparams': nweights})
#
#     def log_artifacts(self, artifacts):
#         super().log_artifacts(artifacts)
#         wandb.save(os.path.join(self.savedir, '*'))


class MLFlowLogger(BasicLogger):
    def __init__(self, args, savedir, verbosity):
        super().__init__(savedir, verbosity)
        mlflow.set_tracking_uri(args.location)
        mlflow.set_experiment(args.exp)
        mlflow.start_run(run_name=args.run)
        params = {k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)}
        mlflow.log_params(params)

    def log_weights(self, model):
        nweights = super().log_weights(model)
        mlflow.log_param('nparams',  nweights)

    def log_metrics(self, output, step=0):
        super().log_metrics(output, step)
        for k, v in output.items():
            try:
                mlflow.log_metric(k, v.item(), step=step)
            except: # TODO catch only the exceptions we intend to here
                pass

    def log_artifacts(self, artifacts):
        super().log_artifacts(artifacts)
        mlflow.log_artifacts(self.savedir)

    def clean_up(self):
        """
        Remove temporary files from file system
        """
        os.system(f'rm -rf {self.savedir}')

