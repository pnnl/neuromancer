"""
Neural ODEs
predefined gray-box priors of selected emulators
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.misc
#local imports
import linear
import rnn



class CSTR(nn.Module):
    def __init__(self, **linargs):
        """
        bilinear term from Torch
        """
        super().__init__()
        self.nx, self.nu, self.nd, self.ny = 2, 1, 0, 2
        self.error_matrix = nn.Parameter(torch.zeros(1), requires_grad=False)
        # Volumetric Flowrate (m^3/sec) / Volume of CSTR (m^3)
        self.qV = nn.Parameter(torch.rand(1))
        self.Tf = nn.Parameter(torch.rand(1))  # Feed Temperature (K)
        self.Caf = nn.Parameter(torch.rand(1))  # Feed Concentration (mol/m^3)
        # Lumped coefficient
        # Heat of reaction for A->B (J/mol)
        # Density of A-B Mixture (kg/m^3)
        # Heat capacity of A-B Mixture (J/kg-K)
        self.mrhoCP = nn.Parameter(torch.rand(1))
        # E - Activation energy in the Arrhenius Equation (J/mol)
        # R - Universal Gas Constant = 8.31451 J/mol-K
        self.EoverR = nn.Parameter(torch.rand(1))
        # Pre-exponential factor (1/sec)
        self.k0 = nn.Parameter(torch.rand(1))
        # U - Overall Heat Transfer Coefficient (W/m^2-K)
        # A - Area - this value is specific for the U calculation (m^2)
        self.UAVrhoCp = nn.Parameter(torch.rand(1))

    def reg_error(self):
        return self.error_matrix

    def forward(self, x, u, d):
        # Inputs (1):
        # Temperature of cooling jacket (K)
        Tc = u
        # Disturbances (2):
        # Tf = Feed Temperature (K)
        # Caf = Feed Concentration (mol/m^3)
        # States (2):
        # Concentration of A in CSTR (mol/m^3)
        Ca = x[0]
        # Temperature in CSTR (K)
        T = x[1]
        # reaction rate
        rA = self.k0 * torch.exp(-self.EoverR / T) * Ca
        # Calculate concentration derivative
        dCadt = self.qV * (self.Caf - Ca) - rA
        # Calculate temperature derivative
        dTdt = self.qV * (self.Tf - T) \
               + self.mrhoCP * rA \
               + self.UAVrhoCp * (Tc - T)
        xdot = torch.zeros(2)
        xdot[0] = dCadt
        xdot[1] = dTdt
        return xdot
