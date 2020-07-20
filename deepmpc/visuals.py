"""

"""
import plot
from dataset import unbatch_data

class Visualizer:

    def train_plot(self, outputs, epochs):
        pass

    def train_output(self):
        return dict()

    def eval(self, outputs):
        return dict()


class VisualizerTrajectories(Visualizer):

    def __init__(self, dataset, model, plot_keys, verbosity):
        self.model = model
        self.dataset = dataset
        self.verbosity = verbosity
        self.plot_keys = plot_keys.intersection(set.union(*[model.input_keys, model.output_keys]))

        # if epoch % self.verbosity == 0:
        #     data = {k: v.squeeze().detach().cpu().numpy()
        #             for (k, v) in outputs.items() if any([plt_k in k for plt_k in self.plot_keys])}
        #     plot.plot_traj(data, figname=None)

    def eval(self, outputs):
        data = {k:  unbatch_data(v).squeeze(1).detach().cpu().numpy()
                for (k, v) in outputs.items() if any([plt_k in k for plt_k in self.plot_keys])}
        for k, v in data.items():
            plot.plot_traj({k: v}, figname=None)
        return dict()
