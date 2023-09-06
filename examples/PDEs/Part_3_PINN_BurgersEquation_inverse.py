"""
# Physics-Informed Neural Networks (PINNs) in Neuromancer

    This tutorial demonstrates the use of PINNs
    for solution and parameter estimation (inverse problem) of partial differential equations (PDEs) in the Neuromancer library.

References
    [1] [Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations.](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
    [2] This tutorial is based on the [Pytorch PINNs tutorial](https://github.com/jdtoscano94/Learning-Python-Physics-Informed-Machine-Learning-PINNs-DeepONets/blob/main/PINNs/4_DiffusionEquation.ipynb) made by [Juan Diego Toscano](https://github.com/jdtoscano94).
    [3] https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/diffusion.1d.html
    [4] https://en.wikipedia.org/wiki/Inverse_problem

---------------------------- Problem Setup -----------------------------------------

    Burgers' Equation
            \frac{\partial y}{\partial t}+ \lambda y\frac{\partial y}{\partial x}=\nu\frac{\partial^2 y}{\partial x^2}           x\in[-1,1]
            x\in[-1,1]
            t\in[0,1]

    Initial Condition:
            y(x,0)= -sin(\pi x)

    Boundary Conditions:
            y(-1,t)=0
            y(1,t)=0

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# data imports
from scipy.io import loadmat


def plot3D(X, T, y):
    X = X.detach().numpy()
    T = T.detach().numpy()
    y = y.detach().numpy()

    #     2D
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cm = ax1.contourf(T, X, y, 20, cmap="viridis")
    fig.colorbar(cm, ax=ax1)  # Add a colorbar to a plot
    ax1.set_title('u(x,t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_aspect('equal')
    #     3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T, X, y, cmap="viridis")
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u(x,t)')
    fig.tight_layout()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float)  # Set default dtype to float32
    torch.manual_seed(1234)  # PyTorch random number generator
    np.random.seed(1234)  # numpy Random number generators
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    ## Generate data of the exact solution
    """
    data = loadmat('./data/burgers_shock.mat')
    x = data['x']  # space:      256 points between -1 and 1 [256x1]
    t = data['t']  # time:       100 time points between 0 and 1 [100x1]
    ysol = data['usol']  # velocitu:   PDE solution [256x100]

    nu = 0.01 / np.pi  # diffusion coeficient

    X, T = np.meshgrid(x, t)  # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
    X = torch.tensor(X.T).float()
    T = torch.tensor(T.T).float()
    y_real = torch.tensor(ysol).float()

    #  -------- Test data -------
    X_test = X.reshape(-1, 1)
    T_test = T.reshape(-1, 1)
    Y_test = y_real.reshape(-1, 1)

    """
    ## Construct training datasets

    We construct training and development datasets containing collocation points (CP) 
    of the spatio-temporal domain (x,t), and samples of the initial conditions (IC), 
    and boundary conditions (BC).

    The dataset is given as:
        \Xi_{train} = [CP^i, IC^j, BC^j] 
            i = 1,...,N_f
            j = 1,...,N_u
        Where N_f defines number of collocation points, 
        and N_u number of initial and boundary condition samples.
    """
    total_points = len(x) * len(t)

    Nf = 1000  # Nf: Number of collocation points

    # Obtain random points of our PDE measurements y(x,t)
    id_f = np.random.choice(total_points, Nf, replace=False)  # Randomly chosen points for Interior
    X_train_Nu = X_test[id_f]
    T_train_Nu = T_test[id_f]
    Y_train_Nu = Y_test[id_f]

    print("We have", total_points, "points. We will select", X_train_Nu.shape[0], "points to train our model.")

    # visualize training points for 2D input space (x, t)
    plt.figure()
    plt.scatter(X_train_Nu.detach().numpy(), T_train_Nu.detach().numpy(),
                s=4., c='blue', marker='o', label='CP')
    plt.title('Samples of the PDE solution y(x,t) for training')
    plt.xlim(-1., 1.)
    plt.ylim(0., 1.)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend(loc='upper right')
    plt.show()
    plt.show(block=True)

    """
    # Create Neuromancer datasets
    """

    from neuromancer.dataset import DictDataset

    # turn on gradients for PINN
    X_train_Nu.requires_grad = True
    T_train_Nu.requires_grad = True

    # Training dataset
    train_data = DictDataset({'x': X_train_Nu, 't': T_train_Nu, 'y': Y_train_Nu}, name='train')
    # test dataset
    test_data = DictDataset({'x': X_test, 't': T_test, 'y': Y_test}, name='test')

    # torch dataloaders
    batch_size = X_train_Nu.shape[0]  # full batch training
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=train_data.collate_fn,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              collate_fn=test_data.collate_fn,
                                              shuffle=False)

    """
    # Neural Architecture in Neuromancer
    """

    from neuromancer.modules import blocks
    from neuromancer.system import Node

    # neural net to solve the PDE problem bounded in the PDE domain
    net = blocks.MLP(insize=2, outsize=1, hsizes=[32, 32], nonlin=nn.Tanh)

    # symbolic wrapper of the neural net
    pde_net = Node(net, ['x', 't'], ['y_hat'], name='net')

    print("symbolic inputs  of the pde_net:", pde_net.input_keys)
    print("symbolic outputs of the pde_net:", pde_net.output_keys)

    # evaluate forward pass on the train data
    net_out = pde_net(train_data.datadict)
    net_out['y_hat'].shape

    """
    Define Physics-informed terms of the PINN

    Our neural network approximation  must satisfy the PDE equations:
                NN_{\theta}(x,t) \approx y(x,t). 

    Thus we define the physics-informed layers as f_{PINN}:
                f_{PINN}(t,x)= \frac{\partial NN_{\theta}(x,t)}{\partial t} + 
                                NN_{\theta}(x,t) \frac{\partial NN_{\theta}(x,t)}{\partial x} 
                                -\nu\frac{\partial^2 NN_{\theta}(x,t)}{\partial x^2}

    We use Automatic Diferentiation to obtain the derivatives of the neural net:
        \frac{\partial NN_{\theta}}{\partial t}, \frac{\partial^2 NN_{\theta}}{\partial x^2}$ 
    
    In the inverse problem, we will jointly learn the neural network parameters $\theta$ 
    together with unknown PDE parameters $\lambda$, and $\nu$.
    
    To simplify the implementation of f_{PINN} we use the symbolic Neuromancer variable. 
    """

    from neuromancer.constraint import variable

    # symbolic Neuromancer variables
    y = variable('y')  # PDE measurements from the dataset
    y_hat = variable('y_hat')  # PDE solution generated as the output of a neural net (pde_net)
    t = variable('t')  # temporal domain
    x = variable('x')  # spatial domain
    # trainable parameters with initial values
    lamb = variable(torch.nn.Parameter(torch.tensor(2.0)), display_name='lambda')  # trainable PDE parameter lambda
    nu = variable(torch.nn.Parameter(torch.tensor(0.01)), display_name='nu')  # trainable PDE parameter nu

    # get the symbolic derivatives
    dy_dt = y_hat.grad(t)
    dy_dx = y_hat.grad(x)
    d2y_d2x = dy_dx.grad(x)
    # get the PINN form
    f_pinn = dy_dt + lamb * y_hat * dy_dx - nu * d2y_d2x

    # computational graph of the PINN neural network
    f_pinn.show()

    """
    PINNs' Loss function terms

    PDE Collocation Points Loss: 
        We evaluate our PINN $f_{PINN}$ over given number ($N_f$) of collocation points (CP) 
        and minimize the PDE residuals in the following loss function:
            \ell_{1}=\frac{1}{N_f}\sum^{N_f}_{i=1}|f_{PINN}(t^i,x^i)|^2

    PDE Initial and Boundary Conditions Loss:
        We select N_f points from our BC and IC and used them in the following 
        supervised learning loss function:
            \ell_{2} = \frac{1}{N_f}\sum^{N_f}_{i=1} |y(t^i,x^i) 
                                                  - NN_{\theta}(t^i,x^i)|^2

    Bound the PINN output in the PDE solution domain: 
        We expect the outputs of the neural net to be bounded in the PDE solution domain 
        NN_{\theta}(x,t) \in [-1.0, 1.0], 
        thus we impose the following inequality constraints via additional penalties:
            \ell_{3}=\frac{1}{N_f}\sum^{N_f}_{i=1} 
            (|RELU(NN_{\theta}(t^i,x^i) - y_{max})|^2 + 
             |RELU(-NN_{\theta}(t^i,x^i) + y_{min})|^2)

    Total Loss:
        The total loss is just a sum of PDE residuals over CP 
        and supervised learning residuals over IC and BC:
            \ell_{PINN}=\ell_{1}+\ell_{2} +\ell_{3}
    """

    # scaling factor for better convergence
    scaling = 1000.

    # PDE CP loss
    ell_1 = (f_pinn == 0.) ^ 2

    # PDE supervised learning loss
    ell_2 = scaling * (y_hat == y) ^ 2

    # ell_3 = output constraints to bound the PINN solution in the PDE output domain [-1.0, 1.0]
    con_1 = (y_hat <= 1.0) ^ 2
    con_2 = (y_hat >= -1.0) ^ 2

    # loss term names for nicer problem plot
    ell_1.name = 'PINN'
    ell_2.name = 'supervised'
    con_1.name = 'y <= ymax'
    con_2.name = 'y >= ymin'

    """
    PINN problem to solve the PDE

        We use stochastic gradient descent to optimize the parameters \theta of the neural network 
        NN_{\theta}(t,x) approximating the solution to the PDE equation y(t,x) 
        using the PINN loss \ell_{PINN} evaluated over sampled CP, IP, and BC.
    """

    from neuromancer.loss import PenaltyLoss
    from neuromancer.problem import Problem
    from neuromancer.trainer import Trainer

    # create Neuromancer optimization loss
    pinn_loss = PenaltyLoss(objectives=[ell_1, ell_2], constraints=[con_1, con_2])

    # construct the PINN optimization problem
    problem = Problem(nodes=[pde_net],  # list of nodes (neural nets) to be optimized
                      loss=pinn_loss,  # physics-informed loss function
                      grad_inference=True  # argument for allowing computation of gradients at the inference time)
                      )

    # show the PINN computational graph
    problem.show()

    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    epochs = 10000

    #  Neuromancer trainer
    trainer = Trainer(
        problem.to(device),
        train_loader,
        optimizer=optimizer,
        epochs=epochs,
        epoch_verbose=200,
        train_metric='train_loss',
        dev_metric='train_loss',
        eval_metric="train_loss",
        warmup=epochs,
    )

    # Train PINN
    best_model = trainer.train()
    # load best trained model
    problem.load_state_dict(best_model)

    print('True parameter lambda = ', 1.0)
    print('Estimated parameter lambda = ', float(lamb.value))
    print('True parameter nu = ', 0.01 / np.pi)
    print('Estimated parameter nu = ', float(nu.value))

    """
    Plot the results
    """
    # evaluate trained PINN on test data
    PINN = problem.nodes[0]
    y1 = PINN(test_data.datadict)['y_hat']

    # arrange data for plotting
    y_pinn = y1.reshape(shape=[256, 100]).detach().cpu()

    # plot PINN solution
    plot3D(X, T, y_pinn)
    # plot exact PDE solution
    plot3D(X, T, y_real)
    # plot residuals PINN - exact PDE
    plot3D(X, T, y_pinn - y_real)

