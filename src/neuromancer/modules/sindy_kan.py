import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromancer.modules.blocks import Block


# =============================================================================
# SINDy-KAN: Sparse Identification of Nonlinear Dynamics through KANs
# =============================================================================

class BasisLibrary(nn.Module):
    """
    Library of basis functions for SINDy-KAN.
    
    Provides a collection of univariate basis functions for the SINDy-KAN
    formulation. Default library: [1, x, x², sin(x), cos(x)]
    
    Reference:
        Howard, A. A., et al. (2025). SINDy-KANs: Sparse identification of 
        non-linear dynamics through Kolmogorov-Arnold networks.
    """
    
    def __init__(self, poly_order=2, include_trig=True):
        """
        Args:
            poly_order (int): Maximum polynomial order (default: 2 for 1, x, x²)
            include_trig (bool): Whether to include sin/cos functions
        """
        super().__init__()
        self.poly_order = poly_order
        self.include_trig = include_trig
        
        self.n_basis = poly_order + 1
        if include_trig:
            self.n_basis += 2
        
        self.function_names = self._get_function_names()
    
    def _get_function_names(self):
        """Get symbolic names for each basis function."""
        names = ['1']
        for i in range(1, self.poly_order + 1):
            names.append('x' if i == 1 else f'x^{i}')
        if self.include_trig:
            names.extend(['cos(x)', 'sin(x)'])
        return names
    
    def forward(self, x):
        """
        Evaluate all basis functions on input x.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features)
            
        Returns:
            torch.Tensor: shape (batch_size, n_features, n_basis)
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        basis_funcs = []
        for power in range(self.poly_order + 1):
            basis_funcs.append(torch.ones_like(x) if power == 0 else x ** power)
        
        if self.include_trig:
            basis_funcs.extend([torch.cos(x), torch.sin(x)])
        
        return torch.stack(basis_funcs, dim=-1)
    
    def get_symbolic_expression(self, coefficients, input_name='x', threshold=0.01):
        """Convert coefficient vector to symbolic expression string."""
        terms = []
        coeffs = coefficients.detach().cpu().numpy()
        
        for coeff, name in zip(coeffs, self.function_names):
            if abs(coeff) > threshold:
                func_name = name.replace('x', input_name)
                if name == '1':
                    terms.append(f'{coeff:.4f}')
                else:
                    terms.append(f'{coeff:.4f}*{func_name}')
        
        return ' + '.join(terms) if terms else '0'


class SINDyKANLinear(nn.Module):
    """
    Single layer of Direct SINDy-KAN.
    
    Learns coefficients for basis functions directly. Maintains two matrices:
    - xi: Dense coefficient matrix
    - lam: Sparse coefficient matrix (regularized with L1)
    """
    
    def __init__(self, in_features, out_features, basis_library=None, 
                 poly_order=2, include_trig=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.basis = basis_library or BasisLibrary(poly_order=poly_order, include_trig=include_trig)
        self.n_basis = self.basis.n_basis
        
        # Dense (Ξ) and sparse (Λ) coefficient matrices
        self.xi = nn.Parameter(torch.randn(out_features, in_features, self.n_basis) * 0.1)
        self.lam = nn.Parameter(torch.randn(out_features, in_features, self.n_basis) * 0.1)
    
    def forward(self, x, use_sparse=False):
        """Forward pass: output = einsum('bin,oin->bo', basis(x), coeffs)"""
        theta = self.basis(x)
        coeffs = self.lam if use_sparse else self.xi
        return torch.einsum('bin,oin->bo', theta, coeffs)
    
    def get_l1_loss(self):
        return torch.sum(torch.abs(self.lam))
    
    def get_diff_loss(self):
        return torch.mean((self.xi - self.lam) ** 2)
    
    def get_sparse_coefficients(self, threshold=0.01):
        with torch.no_grad():
            sparse = self.lam.clone()
            sparse[torch.abs(sparse) < threshold] = 0
            return sparse


class SINDyKANBlock(Block):
    """
    Direct SINDy-KAN Block for symbolic regression.
    
    Implements the Direct SINDy-KAN method which learns sparse symbolic
    representations of functions through trainable coefficient matrices.
    
    Reference:
        Howard, A. A., et al. (2025). SINDy-KANs: Sparse identification of 
        non-linear dynamics through Kolmogorov-Arnold networks.
    
    Example:
        >>> model = SINDyKANBlock(insize=2, outsize=1, hsizes=[2])
        >>> y = model(x)
        >>> model.plot_network(['x', 'y'])  # Visualize learned equations
    """
    
    def __init__(self, insize, outsize, hsizes=[2], poly_order=2, include_trig=True,
                 lambda_kan=0.0, lambda_s=1.0, lambda_lam=1.0, lambda_l1=0.001, lambda_diff=0.001):
        """
        Args:
            insize (int): Number of input features
            outsize (int): Number of output features
            hsizes (list[int]): Hidden layer sizes
            poly_order (int): Max polynomial order in basis
            include_trig (bool): Include sin/cos in basis
            lambda_kan: KAN loss weight (0 for direct method)
            lambda_s, lambda_lam, lambda_l1, lambda_diff: Loss weights
        """
        super().__init__()
        
        self.in_features = insize
        self.out_features = outsize
        self.hsizes = hsizes
        self.lambda_kan = lambda_kan
        self.lambda_s = lambda_s
        self.lambda_lam = lambda_lam
        self.lambda_l1 = lambda_l1
        self.lambda_diff = lambda_diff
        
        self.basis = BasisLibrary(poly_order=poly_order, include_trig=include_trig)
        
        layer_sizes = [insize] + list(hsizes) + [outsize]
        self.layers = nn.ModuleList([
            SINDyKANLinear(layer_sizes[i], layer_sizes[i+1], basis_library=self.basis)
            for i in range(len(layer_sizes) - 1)
        ])
    
    def block_eval(self, x):
        return self.forward(x)
    
    def forward(self, x, use_sparse=False):
        for layer in self.layers:
            x = layer(x, use_sparse=use_sparse)
        return x
    
    def compute_loss(self, x, y_true):
        """Compute SINDy-KAN loss with all regularization terms."""
        y_pred_xi = self.forward(x, use_sparse=False)
        y_pred_lam = self.forward(x, use_sparse=True)
        
        loss_s = F.mse_loss(y_pred_xi, y_true)
        loss_lam = F.mse_loss(y_pred_lam, y_true)
        l1_loss = sum(layer.get_l1_loss() for layer in self.layers)
        diff_loss = sum(layer.get_diff_loss() for layer in self.layers)
        
        total_loss = (self.lambda_s * loss_s + self.lambda_lam * loss_lam +
                      self.lambda_l1 * l1_loss + self.lambda_diff * diff_loss)
        
        return {'loss': total_loss, 'loss_s': loss_s.item(), 'loss_lam': loss_lam.item(),
                'l1_loss': l1_loss.item(), 'diff_loss': diff_loss.item()}
    
    def get_edge_equations(self, input_names=None, threshold=0.01):
        """
        Get symbolic equations for each edge in the network.
        
        Returns:
            list[list[list[str]]]: edge_eqs[layer][out_idx][in_idx] = equation string
        """
        if input_names is None:
            input_names = [f'x_{i}' for i in range(self.in_features)]
        
        edge_equations = []
        current_names = input_names
        
        for layer_idx, layer in enumerate(self.layers):
            sparse_coeffs = layer.get_sparse_coefficients(threshold)
            layer_eqs = []
            next_names = []
            
            for out_idx in range(sparse_coeffs.shape[0]):
                out_eqs = []
                terms = []
                for in_idx in range(sparse_coeffs.shape[1]):
                    expr = self.basis.get_symbolic_expression(
                        sparse_coeffs[out_idx, in_idx, :],
                        input_name=current_names[in_idx],
                        threshold=threshold
                    )
                    out_eqs.append(expr)
                    if expr != '0':
                        terms.append(expr)
                
                layer_eqs.append(out_eqs)
                next_names.append(' + '.join(terms) if terms else '0')
            
            edge_equations.append(layer_eqs)
            current_names = next_names
        
        return edge_equations, current_names
    
    def plot_network(self, input_names=None, output_names=None, threshold=0.01, 
                     figsize=(14, 8), title=None, scale=0.5, x=None):
        """
        Plot the SINDy-KAN network with learned equations on edges (like Figure 5).
        
        Uses actual data forward pass to plot activations (matching KAN_plot.py).
        
        Args:
            input_names (list[str]): Names for input variables
            output_names (list[str]): Names for output variables
            threshold (float): Coefficient threshold for sparsity
            figsize (tuple): Figure size
            title (str): Plot title
            scale (float): Scale factor for plot elements
            x (torch.Tensor): Input data for plotting - REQUIRED for correct plots.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import tempfile
        
        if x is None:
            # Generate default data if not provided
            n_pts = 100
            x = torch.linspace(-2.5, 2.5, n_pts).unsqueeze(1).repeat(1, self.in_features)
        
        if input_names is None:
            input_names = [f'x_{i}' for i in range(self.in_features)]
        if output_names is None:
            output_names = [f'y_{i}' for i in range(self.out_features)]
        
        # Network structure
        layer_sizes = [self.in_features] + list(self.hsizes) + [self.out_features]
        depth = len(layer_sizes) - 1
        
        # Create temp folder for inset plots
        folder = tempfile.mkdtemp()
        
        # Build equation labels for each edge (matching get_function in KAN_plot.py)
        sparse_coeffs_all = [layer.get_sparse_coefficients(threshold) for layer in self.layers]
        
        # Forward pass to compute activations AND build equation labels
        x_in = x
        current_var_names = list(input_names)  # Track variable names through layers
        all_edge_labels = []  # [layer][n_in * n_out] labels
        
        for layer_idx, layer in enumerate(self.layers):
            n_in = layer_sizes[layer_idx]
            n_out = layer_sizes[layer_idx + 1]
            coeffs = sparse_coeffs_all[layer_idx]  # [n_out, n_in, n_basis]
            
            # Compute per-edge activations and labels
            x_reshaped = torch.zeros(x_in.shape[0], n_out, n_in)
            layer_labels = []
            
            # Order: for each input k, for each output j
            for k in range(n_in):
                for j in range(n_out):
                    edge_coeffs = coeffs[j, k, :]  # [n_basis]
                    
                    # Build equation label dynamically based on actual basis
                    label_parts = []
                    for b_idx, coeff_val in enumerate(edge_coeffs):
                        cv = coeff_val.item()
                        if abs(cv) > threshold:
                            basis_name = self.basis.function_names[b_idx]
                            var_name = current_var_names[k]
                            if basis_name == '1':  # constant
                                label_parts.append(f'{cv:.4f}')
                            elif basis_name == 'x':  # linear
                                label_parts.append(f'{cv:.4f}{var_name}')
                            elif basis_name.startswith('x^'):  # power (x^2, x^3, etc.)
                                power = basis_name[2:]  # get the exponent
                                label_parts.append(f'{cv:.4f}{var_name}^{power}')
                            elif basis_name == 'cos(x)':
                                label_parts.append(f'{cv:.4f}cos({var_name})')
                            elif basis_name == 'sin(x)':
                                label_parts.append(f'{cv:.4f}sin({var_name})')
                            else:
                                # Generic fallback
                                func_name = basis_name.replace('x', var_name)
                                label_parts.append(f'{cv:.4f}*{func_name}')
                    
                    eq_label = '+'.join(label_parts).replace('+-', '-') if label_parts else '0'
                    layer_labels.append(eq_label)
                    
                    # Compute activation using the actual basis library
                    input_k = x_in[:, k:k+1]  # Keep 2D for basis
                    basis_vals = self.basis(input_k).squeeze(1)  # [batch, n_basis]
                    
                    activation = torch.sum(edge_coeffs.unsqueeze(0) * basis_vals, dim=1)
                    x_reshaped[:, j, k] = activation
            
            all_edge_labels.append(layer_labels)
            
            # Plot each edge (order: for each input k, for each output j) - NO TITLES in saved images
            label_idx = 0
            for k in range(n_in):
                for j in range(n_out):
                    fig_inset, ax_inset = plt.subplots(figsize=(2.5, 2.5))  # Square
                    
                    color = '#009e73'
                    
                    # Get x and y values for this edge
                    x_vals = x_in[:, k].numpy()
                    y_vals = (x_reshaped[:, j, k] * n_in).detach().numpy()
                    
                    # CRITICAL: Sort by x to create a clean line plot
                    sort_idx = x_vals.argsort()
                    x_vals = x_vals[sort_idx]
                    y_vals = y_vals[sort_idx]
                    
                    # Plot: x-axis = input to this layer, y-axis = edge activation
                    ax_inset.plot(x_vals, y_vals, color=color, lw=4)
                    
                    ax_inset.set_xticks([])
                    ax_inset.set_yticks([])
                    for spine in ax_inset.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.5)
                    
                    # NO TITLE HERE - titles will be added on skeleton
                    
                    plt.savefig(f'{folder}/sp_{layer_idx}_{k}_{j}.png', 
                               bbox_inches="tight", dpi=100)
                    plt.close()
                    label_idx += 1
            
            # Build next layer variable names
            next_var_names = []
            for j in range(n_out):
                terms = []
                for k in range(n_in):
                    idx = k * n_out + j
                    if layer_labels[idx] != '0':
                        terms.append(layer_labels[idx])
                combined = '+'.join(terms).replace('+-', '-') if terms else '0'
                if combined != '0':
                    combined = f'({combined})'
                next_var_names.append(combined)
            current_var_names = next_var_names
            
            # Next layer input = sum over input dimension
            x_in = torch.sum(x_reshaped, dim=2)
        
        # Plot final output function (like KAN_plot.py) - NO TITLE
        # For clean visualization, use diagonal data (x=y for all inputs)
        # This matches the original SINDy-KAN approach (line 60 in run_test.py)
        n_out = layer_sizes[-1]
        for j in range(n_out):
            fig_inset, ax_inset = plt.subplots(figsize=(2.5, 2.5))  # Square
            
            # Create diagonal input data: all inputs equal (like jnp.tile(linspace, [n, 1]).T)
            n_plot = 200
            x_plot = torch.linspace(-2.5, 2.5, n_plot).unsqueeze(1).repeat(1, self.in_features)
            
            # Forward pass through all layers to get output
            x_temp = x_plot
            for layer in self.layers:
                coeffs = layer.get_sparse_coefficients(threshold)
                n_in_l = coeffs.shape[1]
                n_out_l = coeffs.shape[0]
                x_out = torch.zeros(n_plot, n_out_l)
                for out_i in range(n_out_l):
                    for in_i in range(n_in_l):
                        edge_coeffs = coeffs[out_i, in_i, :]
                        input_vals = x_temp[:, in_i:in_i+1]  # Keep 2D for basis
                        basis_vals = self.basis(input_vals).squeeze(1)  # [batch, n_basis]
                        x_out[:, out_i] += torch.sum(edge_coeffs.unsqueeze(0) * basis_vals, dim=1)
                x_temp = x_out
            
            x_vals = x_plot[:, 0].numpy()
            y_vals = x_temp[:, j].detach().numpy()
            
            ax_inset.plot(x_vals, y_vals, color='black', lw=4)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            for spine in ax_inset.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)            
            plt.savefig(f'{folder}/sp_{depth}_{0}_{j}.png', 
                       bbox_inches="tight", dpi=100)
            plt.close()
        
        # Now draw the skeleton with insets
        # Dynamic sizing: compute y1 based on max edges to prevent overlap
        max_num_weights = max(layer_sizes[i] * layer_sizes[i+1] for i in range(depth))
        y0 = 0.4  # Tighter horizontal spacing between layers
        y1 = 0.35 / max(max_num_weights, 4)  # Dynamic inset size - smaller when many edges
        
        fig, ax = plt.subplots(figsize=(14, 10))  # Adjusted figure size
        
        # Plot nodes and edges
        for l in range(depth + 1):
            n = layer_sizes[l]
            for i in range(n):
                y_pos = 1 / (2 * n) + i / n
                ax.scatter(l * y0, y_pos, s=800 * scale**2, color='black', zorder=10)
                
                if l < depth:
                    n_next = layer_sizes[l + 1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j
                        y_mid = 1 / (2 * N) + id_ / N
                        y_next = 1 / (2 * n_next) + j / n_next
                        
                        ax.plot([l * y0, (l + 0.5) * y0 - y1], 
                               [y_pos, y_mid], color='black', lw=2 * scale)
                        ax.plot([(l + 0.5) * y0 + y1, (l + 1) * y0], 
                               [y_mid, y_next], color='black', lw=2 * scale)
        
        ax.set_ylim(-0.12, 1.12)  # Balanced margin
        ax.set_xlim(-0.15, depth * y0 + 0.2 + 2*y1)  # Extend to fit final output inset
        ax.axis('off')
        
        # Add inset images with equation labels above them
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        DC_to_NFC = lambda coords: FC_to_NFC(DC_to_FC(coords))
        
        for l in range(depth):
            n = layer_sizes[l]
            n_next = layer_sizes[l + 1]
            N = n * n_next
            
            label_idx = 0
            for i in range(n):
                for j in range(n_next):
                    id_ = i * n_next + j
                    
                    im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
                    
                    y_mid = 1 / (2 * N) + id_ / N
                    bottom = DC_to_NFC([(0, y_mid - y1)])[0, 1]
                    up = DC_to_NFC([(0, y_mid + y1)])[0, 1]
                    left = DC_to_NFC([((l + 0.5) * y0 - y1, 0)])[0, 0]
                    right = DC_to_NFC([((l + 0.5) * y0 + y1, 0)])[0, 0]
                    
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    newax.imshow(im)
                    newax.axis('off')
                    
                    # Add equation label above inset
                    eq_label = all_edge_labels[l][label_idx] if l < len(all_edge_labels) else '0'
                    if len(eq_label) > 25:
                        eq_label = eq_label[:22] + '...'
                    ax.text((l + 0.5) * y0, y_mid + y1 + 0.015, eq_label,
                           fontsize=11, ha='center', va='bottom', weight='bold')
                    label_idx += 1
        
        # Add final output inset (after last node) with connecting line
        n_out = layer_sizes[-1]
        final_inset_left = depth * y0 + 0.08  # Position for final inset
        for j in range(n_out):
            im = plt.imread(f'{folder}/sp_{depth}_{0}_{j}.png')
            
            # Position after the output node
            y_pos = 1 / (2 * n_out) + j / n_out
            
            # Draw horizontal line from output node to final inset
            ax.plot([depth * y0, final_inset_left], [y_pos, y_pos], 
                   color='black', lw=2 * scale)
            
            bottom = DC_to_NFC([(0, y_pos - y1)])[0, 1]
            up = DC_to_NFC([(0, y_pos + y1)])[0, 1]
            left = DC_to_NFC([(final_inset_left, 0)])[0, 0]
            right = DC_to_NFC([(final_inset_left + 2*y1, 0)])[0, 0]
            
            newax = fig.add_axes([left, bottom, right - left, up - bottom])
            newax.imshow(im)
            newax.axis('off')
            
            # Add equation label above final inset (same style as other insets)
            final_eq_label = current_var_names[j] if j < len(current_var_names) else ''
            if len(final_eq_label) > 50:
                final_eq_label = final_eq_label[:47] + '...'
            ax.text(final_inset_left + y1, y_pos + y1 + 0.015, final_eq_label,
                   fontsize=9, ha='center', va='bottom', weight='bold')
        
        # Add input labels (positioned further left)
        n = layer_sizes[0]
        for i in range(n):
            ax.text(-0.12, 1 / (2 * n) + i / n, input_names[i], 
                   fontsize=28, ha='right', va='center', weight='bold')
        
        # Add output labels (positioned after final inset)
        n = layer_sizes[-1]
        for i in range(n):
            ax.text(final_inset_left + 2*y1 + 0.04, 1 / (2 * n) + i / n, output_names[i],
                   fontsize=28, ha='left', va='center', weight='bold')
        
        if title:
            ax.set_title(title, fontsize=22, weight='bold', pad=20)
        
        # Add final equation at bottom
        final_eq = current_var_names[0] if current_var_names else ''
        ax.text(y0 * depth / 2, -0.08, f'{output_names[0]} = {final_eq}',
               fontsize=14, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Clean up temp files
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        
        return fig, ax
    
    def print_equations(self, input_names=None, threshold=0.01):
        """Print the learned equations in a readable format."""
        if input_names is None:
            input_names = [f'x_{i}' for i in range(self.in_features)]
        
        edge_equations, final_eqs = self.get_edge_equations(input_names, threshold)
        
        print("=" * 60)
        print("LEARNED SINDY-KAN EQUATIONS")
        print("=" * 60)
        
        layer_sizes = [self.in_features] + list(self.hsizes) + [self.out_features]
        
        for layer_idx, layer_eqs in enumerate(edge_equations):
            print(f"\nLayer {layer_idx} → {layer_idx + 1}:")
            print(f"  ({layer_sizes[layer_idx]} inputs → {layer_sizes[layer_idx + 1]} outputs)")
            
            for out_idx, out_eqs in enumerate(layer_eqs):
                terms = [eq for eq in out_eqs if eq != '0']
                if terms:
                    print(f"  Node {out_idx}: {' + '.join(terms)}")
        
        print("\n" + "=" * 60)
        print("FINAL OUTPUT EQUATIONS:")
        print("=" * 60)
        for i, eq in enumerate(final_eqs):
            print(f"  f_{i} = {eq}")
