"""
Callback classes for versatile behavior in the Trainer object at specified checkpoints.
"""

from copy import deepcopy


class Callback:
    """
    Callback base class which allows for bare functionality of Trainer
    """
    def __init__(self):
        pass

    def begin_train(self, trainer):
        pass

    def begin_epoch(self, trainer, output):
        pass

    def begin_eval(self, trainer, output):
        pass

    def end_batch(self, trainer, output):
        pass

    def end_eval(self, trainer, output):
        pass

    def end_epoch(self, trainer, output):
        pass

    def end_train(self, trainer, output):
        pass

    def begin_test(self, trainer):
        pass

    def end_test(self, trainer, output):
        pass


class SysIDCallback(Callback):
    """
    Callbacks for system ID training. Also works with control scripts. May refactor to put visualization and simulation
    functionality directly in the callback at which point there will be separate sysID and control callbacks.
    """
    def __init__(self, simulator, visualizer):
        super().__init__()
        self.simulator, self.visualizer = simulator, visualizer

    def begin_eval(self, trainer, output):
        output.update(self.simulator.dev_eval())

    def end_eval(self, trainer, output):
        if self.visualizer is not None:
            self.visualizer.train_plot(output, trainer.current_epoch)

    def end_train(self, trainer, output):
        plots = self.visualizer.train_output() if self.visualizer is not None else {}
        trainer.logger.log_artifacts(plots)

    def begin_test(self, trainer):
        self.simulator.model.load_state_dict(trainer.best_model)

    def end_test(self, trainer, output):
        output.update(self.simulator.test_eval())
        if self.visualizer is not None:
            plots = self.visualizer.eval(output)
            trainer.logger.log_artifacts(plots)


class ControlCallback(Callback):
    """
    Callbacks for closed loop control training. May refactor to put visualization and simulation
    functionality directly in the callback at which point there will be separate sysID and control callbacks.
    """
    def __init__(self, simulator=None, visualizer=None):
        super().__init__()
        self.simulator, self.visualizer = simulator, visualizer

    def end_test(self, trainer, output):
        plots = {}
        if self.simulator is not None:
            # simulate closed-loop control of system model and emulator (if specified)
            out_model, out_emul = self.simulator.test_eval()
            if self.visualizer is not None:
                # visualize closed-loop control of system model
                plots_model = self.visualizer.eval(out_model, plot_weights=True, figname='CL_model.png')
                plots["plt_model"] = plots_model
                if out_emul is not None:
                    # visualize closed-loop control of emulator
                    plots_emul = self.visualizer.eval(out_emul, figname='CL_emul.png')
                    plots["plt_emul"] = plots_emul
        trainer.logger.log_artifacts(plots)


class DoubleIntegratorCallback(Callback):
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.epoch_model = dict()
        self.epoch_policy = dict()

    def end_epoch(self, trainer, output):
        # save current copies of the system model and control policy
        self.epoch_model[trainer.current_epoch] = deepcopy(trainer.model.state_dict())
        self.epoch_policy[trainer.current_epoch] = \
            deepcopy(trainer.model.components[1].state_dict())

    def end_train(self, trainer, output):
        plots = self.visualizer.train_output(trainer, self.epoch_policy) \
            if self.visualizer is not None else {}
        if plots is not None:
            trainer.logger.log_artifacts(plots)

    def end_test(self, trainer, output):
        if self.visualizer is not None:
            plots = self.visualizer.eval(trainer)
            trainer.logger.log_artifacts(plots)