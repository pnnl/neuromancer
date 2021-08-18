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


# class ControlCallback_old(SysIDCallback):
#     """
#     Callbacks for closed loop control training. May refactor to put visualization and simulation
#     functionality directly in the callback at which point there will be separate sysID and control callbacks.
#     """
#     def __init__(self, simulator, visualizer):
#         super().__init__(simulator=simulator, visualizer=visualizer)
#         self.simulator, self.visualizer = simulator, visualizer
#         self.epoch_model = dict()
#         self.epoch_policy = dict()
#
#     def end_epoch(self, trainer, output):
#         # self.epoch_output[trainer.current_epoch] = output
#         self.epoch_model[trainer.current_epoch] = deepcopy(trainer.model.state_dict())
#         self.epoch_policy[trainer.current_epoch] = \
#             deepcopy(trainer.model.components[1].state_dict())
#
#     def end_train(self, trainer, output):
#         plots = self.visualizer.train_output(trainer, self.epoch_policy) if self.visualizer is not None else {}
#         if plots is not None:
#             trainer.logger.log_artifacts(plots)
#
#     def end_test(self, trainer, output):
#         output.update(self.simulator.test_eval())
#         if self.visualizer is not None:
#             plots = self.visualizer.eval(trainer)
#             trainer.logger.log_artifacts(plots)


class ControlCallback(Callback):
    """
    Callbacks for closed loop control training. May refactor to put visualization and simulation
    functionality directly in the callback at which point there will be separate sysID and control callbacks.
    """
    def __init__(self, simulator=None, visualizer=None):
        super().__init__()
        self.simulator, self.visualizer = simulator, visualizer

    # TODO update
    def end_test(self, trainer, output):
        if self.simulator is not None:
            out = self.simulator.test_eval()
        if self.visualizer is not None:
            plots = self.visualizer.eval(out)
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