"""
# Physics-Informed Neural Networks (PINNs) in Neuromancer

    This tutorial demonstrates the use of PINNs
    for solving partial differential equations (PDEs) using Neuromancer.

References
    [1] [Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations.](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
    [2] This tutorial is based on the [Pytorch PINNs tutorial](https://github.com/jdtoscano94/Learning-Python-Physics-Informed-Machine-Learning-PINNs-DeepONets/blob/main/PINNs/4_DiffusionEquation.ipynb) made by [Juan Diego Toscano](https://github.com/jdtoscano94).
    [3] https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/diffusion.1d.html
---------------------------- Problem Setup -----------------------------------------

    Burgers' Equation
            \frac{\partial y}{\partial t}+ y\frac{\partial y}{\partial x}=\nu\frac{\partial^2 y}{\partial x^2}            x\in[-1,1]
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
    cm = ax1.contourf(T, X, y, 20,cmap="viridis")
    fig.colorbar(cm, ax=ax1) # Add a colorbar to a plot
    ax1.set_title('u(x,t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_aspect('equal')
        #     3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T, X, y,cmap="viridis")
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

    # Samples of Initial conditions (IC)
    #   Left Edge: u(x,0) = -sin(pi*x)
    left_X = X[:, [0]]
    left_T = T[:, [0]]
    left_Y = -torch.sin(np.pi * left_X[:, 0]).unsqueeze(1)

    #   Bottom Edge: x=min; tmin=<t=<max
    bottom_X = X[[0], :].T
    bottom_T = T[[0], :].T
    bottom_Y = torch.zeros(bottom_X.shape[0], 1)
    #   Top Edge: x=max; 0=<t=<1
    top_X = X[[-1], :].T
    top_T = T[[-1], :].T
    top_Y = torch.zeros(top_X.shape[0], 1)

    # Get all the initial and boundary condition data
    X_train = torch.vstack([left_X, bottom_X, top_X])
    T_train = torch.vstack([left_T, bottom_T, top_T])
    Y_train = torch.vstack([left_Y, bottom_Y, top_Y])

    # Choose (Nu) Number of training points for initial and boundary conditions
    Nu = 200

    # Randomly sample Nu points of our available initial and boundary condition data:
    idx = np.sort(np.random.choice(X_train.shape[0], Nu, replace=False))
    X_train_Nu = X_train[idx, :].float()  # Training Points  of x at (IC+BC)
    T_train_Nu = T_train[idx, :].float()  # Training Points  of t at (IC+BC)
    Y_train_Nu = Y_train[idx, :].float()  # Training Points  of y at (IC+BC)

    # x Domain bounds
    x_lb = X_test[0]  # [-1.]
    x_ub = X_test[-1]  # [1.]

    # t Domain bounds
    t_lb = T_test[0]  # [0.]
    t_ub = T_test[-1]  # [0.99]

    #  Choose (Nf) Collocation Points to Evaluate the PDE on
    Nf = 1000  # Nf: Number of collocation points (Evaluate PDE)

    # generate collocation points (CP)
    X_train_CP = torch.FloatTensor(Nf, 1).uniform_(float(x_lb), float(x_ub))
    T_train_CP = torch.FloatTensor(Nf, 1).uniform_(float(t_lb), float(t_ub))

    # add IC+BC to the collocation points
    X_train_Nf = torch.vstack((X_train_CP, X_train_Nu)).float()  # Collocation Points of x (CP)
    T_train_Nf = torch.vstack((T_train_CP, T_train_Nu)).float()  # Collocation Points of t (CP)

    print("Original shapes for X, T, and Y:", X.shape, T.shape, y_real.shape)
    print("Initial and Boundary condition shapes for X:", left_X.shape, bottom_X.shape, top_X.shape)
    print("Initial and Boundary condition shapes for T:", left_T.shape, bottom_T.shape, top_T.shape)
    print("Available training data of IC and BC (X,T,Y):", X_train.shape, T_train.shape, Y_train.shape)
    print("Selected training data of IC and BC (X,T,Y):", X_train_Nu.shape, T_train_Nu.shape, Y_train_Nu.shape)
    print("Final training data of CP (X,T):", X_train_Nf.shape, T_train_Nf.shape)
    print("Final test data (X,T,Y):", X_test.shape, T_test.shape, Y_test.shape)

    # visualize collocation points for 2D input space (x, t)
    plt.figure()
    plt.scatter(X_train_CP.detach().numpy(), T_train_CP.detach().numpy(),
                s=4., c='blue', marker='o', label='CP')
    plt.scatter(X_train_Nu.detach().numpy(), T_train_Nu.detach().numpy(),
                s=4., c='red', marker='o', label='IC+BC')
    plt.title('Sampled IC, BC, and CP (x,t) for training')
    plt.xlim(x_lb, x_ub)
    plt.ylim(t_lb, t_ub)
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
    X_train_Nf.requires_grad = True
    T_train_Nf.requires_grad = True

    # Training dataset
    train_data = DictDataset({'x': X_train_Nf, 't': T_train_Nf}, name='train')
    # test dataset
    test_data = DictDataset({'x': X_test, 't': T_test, 'y': Y_test}, name='test')

    # torch dataloaders
    batch_size = X_train_Nf.shape[0]  # full batch training
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

    To simplify the implementation of f_{PINN} we use the symbolic Neuromancer variable. 
    """

    from neuromancer.constraint import variable

    # symbolic Neuromancer variables
    y_hat = variable('y_hat')  # PDE solution generated as the output of a neural net (pde_net)
    t = variable('t')  # temporal domain
    x = variable('x')  # spatial domain

    # get the symbolic derivatives
    dy_dt = y_hat.grad(t)
    dy_dx = y_hat.grad(x)
    d2y_d2x = dy_dx.grad(x)
    # get the PINN form
    f_pinn = dy_dt + y_hat * dy_dx - nu * d2y_d2x

    # computational graph of the PINN neural network
    f_pinn.show()

    """
    PINNs' Loss function terms

    PDE Collocation Points Loss: 
        We evaluate our PINN f_{PINN} over given number (N_f) of collocation points (CP) and 
        minimize the PDE residuals in the following loss function:
            \ell_{f}=\frac{1}{N_f}\sum^{N_f}_{i=1}|f_{PINN}(t_f^i,x_f^i)|^2
        If f_{PINN} -> 0 then our PINN will be respecting the physical law.

    PDE Initial and Boundary Conditions Loss:
        We select N_u points from our BC and IC and used them in the following 
        supervised learning loss function:
            \ell_{u} = \frac{1}{N_u}\sum^{N_u}_{i=1} |y(t_{u}^i,x_u^i) 
                                                  - NN_{\theta}(t_{u}^i,x_u^i)|^2

    Bound the PINN output in the PDE solution domain: 
        We expect the outputs of the neural net to be bounded in the PDE solution domain 
        NN_{\theta}(x,t) \in [-1.0, 1.0], 
        thus we impose the following inequality constraints via additional penalties:
            \ell_{y}=\frac{1}{N_f}\sum^{N_f}_{i=1} 
            (|RELU(NN_{\theta}(t_{f}^i,x_f^i) - y_{max})|^2 + 
             |RELU(-NN_{\theta}(t_{f}^i,x_f^i) + y_{min})|^2)

    Total Loss:
        The total loss is just a sum of PDE residuals over CP 
        and supervised learning residuals over IC and BC:
            \ell_{PINN}=\ell_{f}+\ell_{u} +\ell_{y}
    """

    # scaling factor for better convergence
    scaling = 100.

    # PDE CP loss
    ell_f = scaling * (f_pinn == 0.) ^ 2

    # PDE IC and BC loss
    ell_u = scaling * (y_hat[-Nu:] == Y_train_Nu) ^ 2  # remember we stacked CP with IC and BC

    # output constraints to bound the PINN solution in the PDE output domain [-1.0, 1.0]
    con_1 = scaling * (y_hat <= 1.0) ^ 2
    con_2 = scaling * (y_hat >= -1.0) ^ 2

    # loss term names for nicer problem plot
    ell_f.name = 'CP'
    ell_u.name = 'IC+BC'
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
    pinn_loss = PenaltyLoss(objectives=[ell_f, ell_u],
                            constraints=[con_1, con_2])

    # construct the PINN optimization problem
    problem = Problem(nodes=[pde_net],  # list of nodes (neural nets) to be optimized
                      loss=pinn_loss,  # physics-informed loss function
                      grad_inference=True  # argument for allowing computation of gradients at the inference time)
                      )

    # show the PINN computational graph
    problem.show()

    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.003)
    epochs = 8000

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

