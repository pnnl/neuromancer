"""
Callback classes for versatile behavior in the Trainer object at specified checkpoints.
"""

import neuromancer as nm
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuromancer.trainer import move_batch_to_device
from tqdm import tqdm


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


class PCA_Callback(Callback):
    def __init__(
            self,
        ):
        super().__init__()
        self.parameter_vectors = []

    def begin_eval(self, trainer, output):

        # get the parameters -> flatten in reversible way -> append to self.parameter_vectors
        self.parameter_vectors.append(torch.nn.utils.parameters_to_vector(trainer.model.parameters()))

    def plot_pca_trajectory(self, trainer, num_points=100, max_plotted_loss=2000, save_name='pca.png'):

        best_parameter = torch.nn.utils.parameters_to_vector(trainer.model.parameters())

        def cost(parameter_vector):
            torch.nn.utils.vector_to_parameters(torch.tensor(parameter_vector, device=torch.device(trainer.device)), trainer.model.parameters())
            for t_batch in trainer.train_data: output = trainer.model(t_batch)

            losses = []
            for t_batch in trainer.train_data:
                t_batch = move_batch_to_device(t_batch, trainer.device)
                output = trainer.model(t_batch)
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.clip)
                losses.append(output["train_loss"])

            return torch.mean(torch.stack(losses))
        
        parameter_matrix = torch.vstack(self.parameter_vectors)

        # see what first parameters dtype is - assume that they are all the same
        dtype = next(trainer.model.parameters()).cpu().detach().numpy().dtype

        self.plot_optimization_landscape(parameter_matrix, best_parameter, cost, save_name, max_plotted_loss, num_points, dtype)
    
    def plot_optimization_landscape(self, parameter_matrix, best_parameter, cost, save_name, max_plotted_loss, num_points, dtype):

        parameter_matrix, best_parameter = parameter_matrix.cpu().detach().numpy(), best_parameter.cpu().detach().numpy()

        components, evals = self.pca((parameter_matrix[:-1] - parameter_matrix[1:]).T, num_components=2)
        components = components.astype(dtype)
        ne = parameter_matrix.shape[0]

        projected_vecs = np.zeros_like(parameter_matrix)
        coeffs = np.zeros([ne,3])
        xy = np.zeros([ne,2])
        losses_traj = np.zeros([ne])
        losses_true_traj = np.zeros([ne])
        for i in range(ne):
            projected_vecs[i] = self.project_vector(parameter_matrix[i], best_parameter, components)
            coeffs[i] = self.find_coefficients(best_parameter, components, projected_vecs[i])
            xy[i] = coeffs[i,1:]
            losses_traj[i] = cost(projected_vecs[i])
            losses_true_traj[i] = cost(parameter_matrix[i])

        x_delta = xy[:,0].max() - xy[:,0].min()
        y_delta = xy[:,1].max() - xy[:,1].min()
        x_max = max(np.abs(xy[:,0].min() - x_delta*0.1), np.abs(xy[:,0].max() + x_delta*0.1))
        x_min = -x_max
        
        y_min = xy[:,1].min() - y_delta*0.1
        y_max = xy[:,1].max() + y_delta*0.1    
        y_max = max(np.abs(xy[:,1].min() - y_delta*0.1), np.abs(xy[:,1].max() + y_delta*0.1))
        y_min = -y_max

        x = np.linspace(x_min, x_max, num_points)
        y = np.linspace(y_min, y_max, num_points)
        X, Y = np.meshgrid(x, y)

        losses = np.zeros_like(X)

        print('gathering losses...')
        for i in tqdm(range(num_points)):
            for j in range(num_points):
                vec = best_parameter + X[i,j] * components[:,0] + Y[i,j] * components[:,1]
                losses[i,j] = cost(vec)
        
        losses_clipped = np.clip(losses, a_min=0.0, a_max = max_plotted_loss)

        # Create a figure with two subplots
        fig = plt.figure(figsize=(25, 5), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.21)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        # Plot the 3D surface on the first subplot
        ax1.plot_surface(X, Y, np.log(losses_clipped), cmap='viridis')
        # fig.colorbar(surface, ax=ax1, shrink=0.5, aspect=10)
        ax1.set_xlabel('PC 2')
        ax1.set_ylabel('PC 1')
        ax1.set_zlabel('Log Loss')

        # Plot the 2D contour plot on the second subplot
        contour = ax2.contourf(X, Y, np.log(losses_clipped), levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax2)
        contour_lines = ax2.contour(X, Y, np.log(losses_clipped), colors='black', linestyles='dashed', linewidths=1)
        ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')
        ax2.plot(xy[:,0], xy[:,1], 'r-', label='Optimization Trajectory')
        ax2.legend()
        ax2.set_xlabel('PC 2')
        ax2.set_ylabel('PC 1')

        # plot the eigenvalues
        ax3.plot(evals)
        ax3.set_xlabel('Sorted Eigenvalue Indices')
        ax3.set_ylabel('Eigenvalues')        

        # Save the combined plot to a file
        plt.savefig(save_name)

    def project_vector(self, vec_to_project, base_vec, components):
        # Project a vector onto the hyperplane defined by a base vector and components.
        projector = components @ np.linalg.inv(components.T @ components) @ components.T
        projection = projector @ (vec_to_project - base_vec)
        projected_vec = base_vec + projection
        return projected_vec

    # https://medium.com/@nahmed3536/a-python-implementation-of-pca-with-numpy-1bbd3b21de2e
    def pca(self, data, num_components=2):
        standardized_data = (data - data.mean(axis = 0)) / data.std(axis = 0)
        covariance_matrix = np.cov(standardized_data, ddof = 1, rowvar = False)
        # we use eigh to ensure no complex eigenvalues from numerical errors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
        order_of_importance = np.argsort(eigenvalues)[::-1] 
        # utilize the sort order to sort eigenvalues and eigenvectors
        # sorted_eigenvalues = eigenvalues[order_of_importance]
        sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns
        # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
        # explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        reduced_data = standardized_data @ sorted_eigenvectors[:,:num_components] # transform the original data
        # total_explained_variance = sum(explained_variance[:k])
        return reduced_data, eigenvalues[order_of_importance]

    def find_coefficients(self, base_vec, components, projection):
        A = np.column_stack([base_vec, components[:,0], components[:,1]])
        coefficients, _, _, _ = np.linalg.lstsq(A, projection, rcond=None)
        return coefficients
