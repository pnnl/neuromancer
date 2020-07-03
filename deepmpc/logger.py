import mlflow
import wandb
import time
import os
import torch
import numbers


class BasicLogger:
    def __init__(self, savedir='test', verbosity=10):
        os.system(f'mkdir {savedir}')
        self.savedir = savedir
        self.verbosity = verbosity
        self.start_time = time.time()

    def log_metrics(self, output, step):
        if step % self.verbosity == 0:
            elapsed_time = time.time() - self.start_time
            entries = [f'epoch: {step}']
            for k, v in output.items():
                try:
                    entries += f'{k}: {v.item():.5f}'
                except AttributeError:
                    pass
            entries += {f'eltime: {elapsed_time}'}
            print('\t'.join(entries))

    def log_artifacts(self, artifacts):
        for k, v in artifacts.items():
            savepath = os.path.join(self.savedir, k)
            try:
                v.savefig(savepath)
            except AttributeError:
                pass
            try:
                v.make_and_save(savepath)
            except AttributeError:
                pass
            torch.save(v, savepath)


class WandBLogger(BasicLogger):
    def __init__(self, savedir, verbosity):
        super().__init__(savedir, verbosity)

    def log_metrics(self, output, step):
        super().log_metrics(output, step)
        for k, v in output:
            if isinstance(v.item(), numbers.Number):
                wandb.log({k: v.item()}, step=step)

    def log_artifacts(self, artifacts):
        super().log_artifacts(artifacts)
        wandb.save(os.path.join(self.savedir, '*'))


class MLFlowLogger(BasicLogger):
    def __init__(self, savedir, verbosity):
        super().__init__(savedir, verbosity)

    def log_metrics(self, output, step):
        super().log_metrics(output, step)
        for k, v in output:
            try:
                mlflow.log_metric(k, v.item(), step=step)
            except mlflow.exceptions.MlflowException:
                pass

    def log_artifacts(self, artifacts):
        super().log_artifacts(artifacts)
        mlflow.log_artifacts(self.savedir)
        os.system(f'rm {self.savedir}')
