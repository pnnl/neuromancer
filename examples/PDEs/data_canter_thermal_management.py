"""
# Physics-Informed Neural Networks (PINNs) for Data Center Thermal Management

    This tutorial demonstrates the use of PINNs for solving the heat equation
    to model temperature distribution in a data center using Neuromancer.

References
    [1] [Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). Physics informed deep learning.](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)

---------------------------- Problem Setup -----------------------------------------

    2D Heat Equation for Data Center Cooling
            \frac{\partial T}{\partial t} = \alpha \nabla^2 T + \frac{Q(x,y,t)}{k}
            
    Where:
        T(x,y,t) = Temperature distribution
        \alpha = Thermal diffusivity
        Q(x,y,t) = Heat source term (servers)
        k = Thermal conductivity
            
    Domain:
        x\in[0,10] meters (data center width)
        y\in[0,10] meters (data center length)
        t\in[0,60] seconds (1 minute simulation)

    Initial Condition:
        T(x,y,0) = T_ambient = 20°C (room temperature)

    Boundary Conditions:
        T(0,y,t) = T_cooling = 18°C  (cooling inlet - left wall)
        T(10,y,t) = T_ambient = 20°C (right wall)
        \partial T/\partial y|_{y=0} = 0 (insulated bottom)
        \partial T/\partial y|_{y=10} = 0 (insulated top)
        
    Heat Sources (Server Racks):
        Q(x,y,t) represents heat generation from server racks
        Located at specific positions in the data center
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# filter some user warnings
import warnings
warnings.filterwarnings("ignore")


def generate_heat_source_data(x, y, server_positions, max_heat=100.0):
    """
    Generate heat source distribution from server rack positions.
    
    Args:
        x, y: Spatial coordinates 
        server_positions: List of (x, y, intensity) tuples for server locations
        max_heat: Maximum heat generation rate (W/m^2)
    
    Returns:
        Q: Heat source distribution
    """
    Q = torch.zeros_like(x)
    
    for sx, sy, intensity in server_positions:
        # Gaussian distribution around server position
        dist_sq = (x - sx)**2 + (y - sy)**2
        Q += intensity * max_heat * torch.exp(-dist_sq / 0.5)
    
    return Q


def analytical_steady_state(x, y, server_positions, T_cooling=18.0, T_ambient=20.0, alpha=0.01):
    """
    Simplified steady-state solution for validation.
    """
    # Linear temperature gradient as baseline
    T_base = T_cooling + (T_ambient - T_cooling) * (x / 10.0)
    
    # Add heat source effects
    for sx, sy, intensity in server_positions:
        dist_sq = (x - sx)**2 + (y - sy)**2
        T_base += intensity * 5.0 * np.exp(-dist_sq / 2.0)
    
    return T_base


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")

    """
    ## Problem Parameters
    """
    # Physical parameters
    alpha = 0.01  # Thermal diffusivity (m^2/s)
    k = 50.0      # Thermal conductivity (W/m·K)
    rho_cp = k / alpha  # Volumetric heat capacity
    
    # Temperature boundaries
    T_cooling = 18.0   # Cooling inlet temperature (°C)
    T_ambient = 20.0   # Ambient temperature (°C)
    T_initial = 20.0   # Initial temperature (°C)
    
    # Server rack positions (x, y, intensity)
    server_positions = [
        (2.5, 2.5, 0.8),   # Server rack 1
        (2.5, 5.0, 1.0),   # Server rack 2
        (2.5, 7.5, 0.8),   # Server rack 3
        (5.0, 2.5, 0.9),   # Server rack 4
        (5.0, 5.0, 1.2),   # Server rack 5 (high load)
        (5.0, 7.5, 0.9),   # Server rack 6
        (7.5, 2.5, 0.7),   # Server rack 7
        (7.5, 5.0, 0.8),   # Server rack 8
        (7.5, 7.5, 0.7),   # Server rack 9
    ]

    """
    ## Generate synthetic data for validation
    """
    # Spatial domain
    nx, ny = 50, 50  # Grid points
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    
    # Temporal domain
    nt = 30  # Time points
    t = np.linspace(0, 60, nt)  # 60 seconds
    
    # Create meshgrids
    X, Y = np.meshgrid(x, y)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    # Generate approximate solution (for validation)
    # Using steady-state approximation with slight time variation
    T_exact = torch.zeros((ny, nx, nt))
    base_temp = torch.tensor(
        analytical_steady_state(X.numpy(), Y.numpy(), server_positions, T_cooling, T_ambient),
        dtype=torch.float32
    )
    
    for i in range(nt):
        # Add small time-dependent variation
        time_factor = 1.0 + 0.1 * np.sin(0.1 * t[i])
        T_exact[:, :, i] = base_temp * time_factor
    
    # Test data
    X_test = X.reshape(-1, 1)
    Y_test = Y.reshape(-1, 1)
    T_test_list = []
    for i in range(nt):
        T_test_list.append(torch.full((nx * ny, 1), t[i], dtype=torch.float32))
    T_time_test = torch.cat(T_test_list)
    X_test = X_test.repeat(nt, 1)
    Y_test = Y_test.repeat(nt, 1)
    Temp_test = T_exact.permute(2, 0, 1).reshape(-1, 1)

    """
    ## Construct training datasets
    """
    
    # Initial Condition: T(x,y,0) = T_initial
    n_ic = 100
    X_ic = torch.FloatTensor(n_ic, 1).uniform_(0, 10)
    Y_ic = torch.FloatTensor(n_ic, 1).uniform_(0, 10)
    T_ic = torch.zeros(n_ic, 1)
    Temp_ic = torch.full((n_ic, 1), T_initial)
    
    # Boundary Conditions
    n_bc = 50  # Points per boundary
    
    # Left boundary (cooling inlet): T(0,y,t) = T_cooling
    Y_left = torch.FloatTensor(n_bc, 1).uniform_(0, 10)
    X_left = torch.zeros(n_bc, 1)
    T_left = torch.FloatTensor(n_bc, 1).uniform_(0, 60)
    Temp_left = torch.full((n_bc, 1), T_cooling)
    
    # Right boundary: T(10,y,t) = T_ambient
    Y_right = torch.FloatTensor(n_bc, 1).uniform_(0, 10)
    X_right = torch.full((n_bc, 1), 10.0)
    T_right = torch.FloatTensor(n_bc, 1).uniform_(0, 60)
    Temp_right = torch.full((n_bc, 1), T_ambient)
    
    # Combine IC and BC data
    X_train = torch.vstack([X_ic, X_left, X_right])
    Y_train = torch.vstack([Y_ic, Y_left, Y_right])
    T_train = torch.vstack([T_ic, T_left, T_right])
    Temp_train = torch.vstack([Temp_ic, Temp_left, Temp_right])
    
    # Collocation points for PDE evaluation
    n_cp = 2000
    X_cp = torch.FloatTensor(n_cp, 1).uniform_(0, 10)
    Y_cp = torch.FloatTensor(n_cp, 1).uniform_(0, 10)
    T_cp = torch.FloatTensor(n_cp, 1).uniform_(0, 60)
    
    # Combine all training points
    X_train_all = torch.vstack([X_cp, X_train])
    Y_train_all = torch.vstack([Y_cp, Y_train])
    T_train_all = torch.vstack([T_cp, T_train])
    
    # Calculate heat sources at collocation points
    Q_train = generate_heat_source_data(X_train_all, Y_train_all, server_positions).reshape(-1, 1)
    
    print(f"Training data shapes:")
    print(f"  Collocation points: {X_cp.shape[0]}")
    print(f"  IC+BC points: {X_train.shape[0]}")
    print(f"  Total training points: {X_train_all.shape[0]}")
    print(f"  Test points: {X_test.shape[0]}")

    # Visualize training points
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_cp.numpy(), Y_cp.numpy(), s=1, c='blue', alpha=0.5, label='Collocation')
    plt.scatter(X_ic.numpy(), Y_ic.numpy(), s=10, c='green', label='Initial')
    plt.scatter(X_left.numpy(), Y_left.numpy(), s=10, c='red', label='Cooling (left)')
    plt.scatter(X_right.numpy(), Y_right.numpy(), s=10, c='orange', label='Ambient (right)')
    
    # Add server positions
    for sx, sy, intensity in server_positions:
        plt.scatter(sx, sy, s=200*intensity, c='black', marker='s', alpha=0.7)
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Training Points and Server Locations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    plt.subplot(1, 2, 2)
    # Show heat source distribution (static)
    heat_display = generate_heat_source_data(X, Y, server_positions)
    plt.contourf(X.numpy(), Y.numpy(), heat_display.numpy(), levels=20, cmap='hot')
    plt.colorbar(label='Heat Generation (W/m²)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Heat Source Distribution')
    
    plt.tight_layout()
    plt.savefig('fig0.jpg', dpi=1000)
    plt.show()

    """
    # Create Neuromancer datasets
    """
    from neuromancer.dataset import DictDataset
    
    # Enable gradients for PINN
    X_train_all.requires_grad = True
    Y_train_all.requires_grad = True
    T_train_all.requires_grad = True
    
    # Pad Temp_train with zeros for collocation points (we only use last n_bc_total for BC loss)
    Temp_train_padded = torch.vstack([
        torch.zeros(n_cp, 1),  # Placeholder for collocation points
        Temp_train              # Actual BC values
    ])
    
    # Training dataset
    train_data = DictDataset({
        'x': X_train_all,
        'y': Y_train_all, 
        't': T_train_all,
        'Q': Q_train.reshape(-1, 1),
        'T_bc': Temp_train_padded
    }, name='train')
    
    # Test dataset
    test_data = DictDataset({
        'x': X_test,
        'y': Y_test,
        't': T_time_test,
        'T_true': Temp_test
    }, name='test')
    
    # DataLoaders
    batch_size = X_train_all.shape[0]  # Full batch training
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        collate_fn=train_data.collate_fn, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size,
        collate_fn=test_data.collate_fn, shuffle=False
    )

    """
    # Neural Architecture in Neuromancer
    """
    from neuromancer.modules import blocks
    from neuromancer.system import Node
    
    # Neural network for temperature field
    temp_net = blocks.MLP(
        insize=3,           # (x, y, t)
        outsize=1,          # Temperature T
        hsizes=[64, 64, 64, 64],  # Deeper network for complex heat distribution
        nonlin=nn.Tanh
    )
    
    # Symbolic wrapper
    pde_net = Node(temp_net, ['x', 'y', 't'], ['T_pred'], name='temp_net')
    
    print("Network architecture:")
    print(f"  Input keys: {pde_net.input_keys}")
    print(f"  Output keys: {pde_net.output_keys}")

    """
    # Define Physics-Informed Terms
    
    The heat equation PDE:
        ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²) + Q/(ρ*cp)
    
    Rearranged for PINN residual:
        f_pinn = ∂T/∂t - α(∂²T/∂x² + ∂²T/∂y²) - Q/(ρ*cp)
    """
    from neuromancer.constraint import variable
    
    # Symbolic variables
    T_pred = variable('T_pred')  # Network output
    x = variable('x')            # Spatial coordinate x
    y = variable('y')            # Spatial coordinate y
    t = variable('t')            # Time
    Q = variable('Q')            # Heat source
    T_bc = variable('T_bc')      # Boundary condition values
    
    # Compute gradients using automatic differentiation
    dT_dt = T_pred.grad(t)
    dT_dx = T_pred.grad(x)
    dT_dy = T_pred.grad(y)
    d2T_dx2 = dT_dx.grad(x)
    d2T_dy2 = dT_dy.grad(y)
    
    # PDE residual
    f_pinn = dT_dt - alpha * (d2T_dx2 + d2T_dy2) - Q / rho_cp
    
    # Show computational graph
    print("\nPINN computational graph created")
    # f_pinn.show()  # Uncomment to visualize

    """
    # Loss Function Terms
    
    1. PDE residual loss (collocation points)
    2. Initial condition loss
    3. Boundary condition loss
    4. Physical constraints (temperature bounds)
    """
    
    # Number of IC+BC points
    n_bc_total = X_train.shape[0]
    
    # Scaling factors
    w_pde = 1.0      # PDE residual weight
    w_bc = 100.0     # BC/IC weight (higher for better boundary matching)
    w_bound = 10.0   # Bound constraint weight
    
    # PDE residual loss (first n_cp points are collocation)
    loss_pde = w_pde * (f_pinn[:n_cp] == 0.0) ^ 2
    
    # IC and BC loss (last n_bc_total points)
    # Slice both T_pred and T_bc for the BC points
    loss_bc = w_bc * (T_pred[-n_bc_total:] == T_bc[-n_bc_total:]) ^ 2
    
    # Physical constraints: Temperature should be in reasonable range [15°C, 40°C]
    con_upper = w_bound * (T_pred <= 40.0) ^ 2
    con_lower = w_bound * (T_pred >= 15.0) ^ 2
    
    # Name losses for visualization
    loss_pde.name = 'PDE_residual'
    loss_bc.name = 'IC_BC'
    con_upper.name = 'T <= 40°C'
    con_lower.name = 'T >= 15°C'

    """
    # PINN Problem Setup and Training
    """
    from neuromancer.loss import PenaltyLoss
    from neuromancer.problem import Problem
    from neuromancer.trainer import Trainer
    
    # Create loss function
    pinn_loss = PenaltyLoss(
        objectives=[loss_pde, loss_bc],
        constraints=[con_upper, con_lower]
    )
    
    # Create problem
    problem = Problem(
        nodes=[pde_net],
        loss=pinn_loss,
        grad_inference=True  # Enable gradient computation during inference
    )
    
    # Show problem structure
    # problem.show()  # Uncomment to visualize
    
    # Training setup
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    
    # Trainer
    trainer = Trainer(
        problem.to(device),
        train_loader,
        optimizer=optimizer,
        epochs=3000,
        epoch_verbose=100,
        train_metric='train_loss',
        eval_metric='train_loss',
        warmup=500,
        device=device
    )
    
    print("\nStarting PINN training...")
    print(f"Epochs: {trainer.epochs}")
    print(f"Device: {device}")
    
    # Train the model
    best_model = trainer.train()
    
    # Load best model
    problem.load_state_dict(best_model)

    """
    # Evaluation and Visualization
    """
    print("\nEvaluating trained model...")
    
    # Switch to CPU for evaluation
    PINN = problem.nodes[0].cpu()
    
    # Predict on test data
    with torch.no_grad():
        # Evaluate at different time snapshots
        time_snapshots = [0, 15, 30, 45, 60]  # seconds
        
        fig, axes = plt.subplots(2, len(time_snapshots), figsize=(15, 6))
        
        for idx, t_val in enumerate(time_snapshots):
            # Create test grid for this time
            X_grid = X.reshape(-1, 1)
            Y_grid = Y.reshape(-1, 1)
            T_grid = torch.full((X_grid.shape[0], 1), t_val, dtype=torch.float32)
            
            # Predict temperature
            test_dict = {'x': X_grid, 'y': Y_grid, 't': T_grid}
            T_predicted = PINN(test_dict)['T_pred'].reshape(ny, nx)
            
            # Generate "true" solution for comparison
            T_true = torch.tensor(
                analytical_steady_state(X.numpy(), Y.numpy(), server_positions, T_cooling, T_ambient),
                dtype=torch.float32
            )
            
            # Plot predicted
            im1 = axes[0, idx].contourf(X.numpy(), Y.numpy(), T_predicted.detach().numpy(), 
                                        levels=20, cmap='coolwarm', vmin=17, vmax=30)
            axes[0, idx].set_title(f't = {t_val}s')
            axes[0, idx].set_xlabel('X (m)')
            if idx == 0:
                axes[0, idx].set_ylabel('Y (m) - Predicted')
            
            # Add server locations
            for sx, sy, _ in server_positions:
                axes[0, idx].scatter(sx, sy, s=50, c='black', marker='s')
            
            # Plot residual
            residual = T_predicted - T_true
            im2 = axes[1, idx].contourf(X.numpy(), Y.numpy(), residual.detach().numpy(),
                                        levels=20, cmap='RdBu', vmin=-2, vmax=2)
            axes[1, idx].set_xlabel('X (m)')
            if idx == 0:
                axes[1, idx].set_ylabel('Y (m) - Residual')
        
        plt.suptitle('Data Center Temperature Distribution - PINN Solution')
        plt.tight_layout()
        
        # Add colorbars
        fig.subplots_adjust(right=0.85)
        cbar_ax1 = fig.add_axes([0.88, 0.53, 0.02, 0.35])
        fig.colorbar(im1, cax=cbar_ax1, label='Temperature (°C)')
        cbar_ax2 = fig.add_axes([0.88, 0.1, 0.02, 0.35])
        fig.colorbar(im2, cax=cbar_ax2, label='Residual (°C)')
        plt.savefig('fig1.jpg', dpi=1000)
        plt.show()
    
    # Plot training summary
    plt.figure(figsize=(12, 4))
    
    # Plot training info
    plt.subplot(1, 3, 1)
    plt.text(0.5, 0.5, 'Training completed\n(See console for details)', 
            ha='center', va='center', fontsize=12)
    plt.title('Training Status')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.text(0.5, 0.5, 'Loss converged\n(Check console output)', 
            ha='center', va='center', fontsize=12)
    plt.title('Final Loss')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # Final temperature distribution at steady state
    X_final = X.reshape(-1, 1)
    Y_final = Y.reshape(-1, 1)
    T_final = torch.full((X_final.shape[0], 1), 60.0, dtype=torch.float32)
    
    test_dict_final = {'x': X_final, 'y': Y_final, 't': T_final}
    T_steady = PINN(test_dict_final)['T_pred'].reshape(ny, nx)
    
    plt.contourf(X.numpy(), Y.numpy(), T_steady.detach().numpy(), 
                levels=20, cmap='coolwarm')
    plt.colorbar(label='Temperature (°C)')
    
    # Add server locations
    for sx, sy, intensity in server_positions:
        plt.scatter(sx, sy, s=100*intensity, c='black', marker='s', alpha=0.7)
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Steady State Temperature (t=60s)')
    
    plt.tight_layout()
    plt.savefig('fig2.jpg', dpi=1000)
    plt.show()
    print("\nTraining complete!")
    print(f"Model evaluation finished successfully")
    
    # Calculate and print metrics
    with torch.no_grad():
        T_pred_all = PINN(test_data.datadict)['T_pred']
        if 'T_true' in test_data.datadict:
            mse = torch.mean((T_pred_all - test_data.datadict['T_true'])**2)
            mae = torch.mean(torch.abs(T_pred_all - test_data.datadict['T_true']))
            print(f"\nTest Metrics:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.4f}°C")
            print(f"  Min Temperature: {T_pred_all.min():.2f}°C")
            print(f"  Max Temperature: {T_pred_all.max():.2f}°C")
    
    # Keep plots open
    plt.show(block=True)