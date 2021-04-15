"""
Callback classes for versatile behavior in the Trainer object at specified checkpoints.
"""


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

    def begin_test(self, trainer, output):
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

    def begin_test(self, trainer, output):
        self.simulator.model.load_state_dict(trainer.best_model)

    def end_test(self, trainer, output):
        output.update(self.simulator.test_eval())
        if self.visualizer is not None:
            plots = self.visualizer.eval(output)
            trainer.logger.log_artifacts(plots)







