"""
# Physics-Informed Neural Networks (PINNs) for the 1D Wave Equation

    This tutorial demonstrates the use of PINNs
    for solving the 1D wave equation using Neuromancer.

References
    [1] [Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations.](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
    [2] https://en.wikipedia.org/wiki/Wave_equation
---------------------------- Problem Setup -----------------------------------------

    Wave equation
            \\frac{\\partial^2 y}{\\partial t^2} = c^2 \\frac{\\partial^2 y}{\\partial x^2}
            x \\in [0, 1]
            t \\in [0, 1]
            c = 1.0 (wave speed)

    Initial Conditions:
            y(x, 0) = sin(\\pi x)           (displacement)
            \\frac{\\partial y}{\\partial t}(x, 0) = 0   (velocity)

    Boundary Conditions:
            y(0, t) = 0
            y(1, t) = 0

    Exact solution:
            y(x, t) = sin(\\pi x) cos(\\pi t)

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# filter some user warnings from torch broadcast
import warnings
warnings.filterwarnings("ignore")


C = 1.0  # wave speed


def f_real(x, t):
    """Exact PDE solution y(x,t) = sin(pi*x) * cos(pi*c*t)."""
    return torch.sin(np.pi * x) * torch.cos(np.pi * C * t)


def plot3D(X, T, y, title='y(x,t)'):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cm = ax1.contourf(T.numpy(), X.numpy(), y.numpy(), 20, cmap="viridis")
    fig.colorbar(cm, ax=ax1)
    ax1.set_title(title)
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T.numpy(), X.numpy(), y.numpy(), cmap="viridis")
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel(title)
    fig.tight_layout()


if __name__ == "__main__":

    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    """
    ## Generate data of the exact solution
    """
    x_min = 0.0
    x_max = 1.0
    t_min = 0.0
    t_max = 1.0

    total_points_x = 200
    total_points_t = 200

    x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
    t = torch.linspace(t_min, t_max, total_points_t).view(-1, 1)

    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1))
    y_real = f_real(X, T)

    ymin = y_real.min()
    ymax = y_real.max()

    plot3D(X, T, y_real, title='Exact y(x,t)')

    # Test data
    X_test = X.transpose(1, 0).flatten()[:, None].float()
    T_test = T.transpose(1, 0).flatten()[:, None].float()
    Y_test = y_real.transpose(1, 0).flatten()[:, None].float()

    """
    ## Construct training datasets
    
    The wave equation is second-order in time, so we enforce:
      - Initial displacement: y(x,0) = sin(pi*x)
      - Initial velocity: dy/dt(x,0) = 0
      - Boundary conditions: y(0,t) = y(1,t) = 0
    """

    # Samples of Initial Condition (IC) for displacement: y(x,0) = sin(pi*x)
    ic_X = X[:, [0]]
    ic_T = T[:, [0]]
    ic_Y = torch.sin(np.pi * ic_X[:, 0]).unsqueeze(1)

    # Samples of Boundary Conditions (BC)
    #   Bottom Edge: x=0; tmin <= t <= tmax
    bc_bottom_X = X[[0], :].T
    bc_bottom_T = T[[0], :].T
    bc_bottom_Y = torch.zeros(bc_bottom_X.shape[0], 1)
    #   Top Edge: x=1; tmin <= t <= tmax
    bc_top_X = X[[-1], :].T
    bc_top_T = T[[-1], :].T
    bc_top_Y = torch.zeros(bc_top_X.shape[0], 1)

    # Combine IC and BC data
    X_train = torch.vstack([ic_X, bc_bottom_X, bc_top_X])
    T_train = torch.vstack([ic_T, bc_bottom_T, bc_top_T])
    Y_train = torch.vstack([ic_Y, bc_bottom_Y, bc_top_Y])

    # Choose (Nu) Number of training points for IC and BC
    Nu = 150

    idx = np.sort(np.random.choice(X_train.shape[0], Nu, replace=False))
    X_train_Nu = X_train[idx, :].float()
    T_train_Nu = T_train[idx, :].float()
    Y_train_Nu = Y_train[idx, :].float()

    # Domain bounds
    x_lb = X_test[0]
    x_ub = X_test[-1]
    t_lb = T_test[0]
    t_ub = T_test[-1]

    # Collocation Points (CP)
    Nf = 2000
    X_train_CP = torch.FloatTensor(Nf, 1).uniform_(float(x_lb), float(x_ub))
    T_train_CP = torch.FloatTensor(Nf, 1).uniform_(float(t_lb), float(t_ub))

    # Also sample points at t=0 for initial velocity constraint
    N_vel = 100
    X_train_vel = torch.FloatTensor(N_vel, 1).uniform_(float(x_lb), float(x_ub))
    T_train_vel = torch.zeros(N_vel, 1)

    # Stack all training points: CP + IC/BC + velocity IC
    X_train_Nf = torch.vstack((X_train_CP, X_train_Nu, X_train_vel)).float()
    T_train_Nf = torch.vstack((T_train_CP, T_train_Nu, T_train_vel)).float()

    print("Original shapes for X, T, and Y:", X.shape, T.shape, y_real.shape)
    print("Available IC+BC data (X,T,Y):", X_train.shape, T_train.shape, Y_train.shape)
    print("Selected IC+BC data (X,T,Y):", X_train_Nu.shape, T_train_Nu.shape, Y_train_Nu.shape)
    print("Velocity IC points:", N_vel)
    print("Final training data of CP+IC+BC+vel (X,T):", X_train_Nf.shape, T_train_Nf.shape)
    print("Final test data (X,T,Y):", X_test.shape, T_test.shape, Y_test.shape)

    # visualize training points
    plt.figure()
    plt.scatter(X_train_CP.detach().numpy(), T_train_CP.detach().numpy(),
                s=4., c='blue', marker='o', label='CP')
    plt.scatter(X_train_Nu.detach().numpy(), T_train_Nu.detach().numpy(),
                s=4., c='red', marker='o', label='IC+BC')
    plt.scatter(X_train_vel.detach().numpy(), T_train_vel.detach().numpy(),
                s=4., c='green', marker='o', label='vel IC')
    plt.title('Sampled IC, BC, vel IC, and CP (x,t) for training')
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
    # Test dataset
    test_data = DictDataset({'x': X_test, 't': T_test, 'y': Y_test}, name='test')

    # Dataloaders
    batch_size = X_train_Nf.shape[0]
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

    # neural net to approximate the PDE solution
    net = blocks.MLP(insize=2, outsize=1, hsizes=[50, 50, 50, 50], nonlin=nn.Tanh)

    # symbolic wrapper
    pde_net = Node(net, ['x', 't'], ['y_hat'], name='net')

    print("symbolic inputs  of the pde_net:", pde_net.input_keys)
    print("symbolic outputs of the pde_net:", pde_net.output_keys)

    # evaluate forward pass on train data
    net_out = pde_net(train_data.datadict)
    net_out['y_hat'].shape

    """
    Define Physics-informed terms of the PINN

    Wave equation PDE residual:
        f_{PINN}(t,x) = \\frac{\\partial^2 NN}{\\partial t^2} - c^2 \\frac{\\partial^2 NN}{\\partial x^2}

    Initial velocity constraint:
        g_{PINN}(x) = \\frac{\\partial NN}{\\partial t}(x, 0) = 0
    """

    from neuromancer.constraint import variable

    # symbolic Neuromancer variables
    y_hat = variable('y_hat')
    t_var = variable('t')
    x_var = variable('x')

    # first derivatives
    dy_dt = y_hat.grad(t_var)
    dy_dx = y_hat.grad(x_var)

    # second derivatives
    d2y_dt2 = dy_dt.grad(t_var)
    d2y_dx2 = dy_dx.grad(x_var)

    # Wave equation PDE residual: u_tt - c^2 * u_xx = 0
    f_pinn = d2y_dt2 - C ** 2 * d2y_dx2

    # Initial velocity residual: dy/dt at t=0 points (last N_vel entries)
    g_pinn = dy_dt

    """
    PINNs' Loss function terms

    PDE Collocation Points Loss (wave equation residual):
        \\ell_f = (1/N_f) \\sum |f_{PINN}|^2

    IC+BC supervised loss:
        \\ell_u = (1/N_u) \\sum |y - NN(t,x)|^2

    Initial velocity loss:
        \\ell_v = (1/N_vel) \\sum |dy/dt(x,0)|^2

    Output bounding constraints:
        \\ell_y = penalties for NN output outside [ymin, ymax]
    """

    scaling = 100.

    # PDE CP loss (wave equation residual = 0)
    ell_f = scaling * (f_pinn == 0.) ^ 2

    # IC+BC supervised loss
    ell_u = scaling * (y_hat[Nf:Nf + Nu] == Y_train_Nu) ^ 2

    # Initial velocity loss (dy/dt = 0 at t=0)
    ell_v = scaling * (g_pinn[Nf + Nu:] == 0.) ^ 2

    # output constraints
    con_1 = scaling * (y_hat <= ymax) ^ 2
    con_2 = scaling * (y_hat >= ymin) ^ 2

    # loss term names
    ell_f.name = 'PDE'
    ell_u.name = 'IC+BC'
    ell_v.name = 'vel IC'
    con_1.name = 'y <= ymax'
    con_2.name = 'y >= ymin'

    """
    PINN problem to solve the PDE
    """

    from neuromancer.loss import PenaltyLoss
    from neuromancer.problem import Problem
    from neuromancer.trainer import Trainer

    # create optimization loss
    pinn_loss = PenaltyLoss(objectives=[ell_f, ell_u, ell_v],
                            constraints=[con_1, con_2])

    # construct the PINN optimization problem
    problem = Problem(nodes=[pde_net],
                      loss=pinn_loss,
                      grad_inference=True)

    optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)
    epochs = 10000

    # Neuromancer trainer
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
        device=device,
    )

    # Train PINN
    best_model = trainer.train()
    problem.load_state_dict(best_model)

    """
    Plot the results
    """
    # evaluate trained PINN on test data
    PINN = problem.nodes[0].cpu()
    y1 = PINN(test_data.datadict)['y_hat']

    y_pinn = y1.reshape(shape=[total_points_t, total_points_x]).transpose(1, 0).detach().cpu()

    plot3D(X, T, y_pinn, title='PINN y(x,t)')
    plot3D(X, T, y_real, title='Exact y(x,t)')
    plot3D(X, T, y_pinn - y_real, title='Error: PINN - Exact')

    # compute and print relative L2 error
    l2_error = torch.norm(y_pinn - y_real) / torch.norm(y_real)
    print(f"\nRelative L2 error: {l2_error.item():.6f}")
