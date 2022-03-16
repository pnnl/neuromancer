


import numpy as np
import pyomo
from pyomo.environ import *
from pyomo.dae import * 
import itertools 
import pandas as pd
import numpy.random as nprd

import numpy.matlib

#import matplotlib.pyplot as plt

import scipy.optimize as scio


import pyutilib.services
from pyomo.opt import TerminationCondition

import cyipopt




def ED_cntsdyn_opt(ED_parms,Ld_parms,Pg_int,OMi,omega_bound):
    
    
    
    ############################################################################
    #############################################################################
    ## unpack the variable paramters
    
    
    
    
    
    cost_coef_g = ED_parms[0]
    cost_coef_s = ED_parms[1]
    rmp_tol = ED_parms[2]
    
    
    
    
    
    
    PLi = Ld_parms[0]
    t_strt_vec = Ld_parms[1]
    slope_vec = Ld_parms[2]
    duration_vec = Ld_parms[3]
    
    
    
    
    
        
        
        
        
    
    ############################################################################
    #############################################################################
    ### Fixed ED problem parameters
    ######################################
    
    
    
    #Time horizon and discretization
    Tf = 60   #final time
    ndt = 6000   #number of time discretizations points
    
    
    
    
    
    #generator power output bounds
    P_g_nom = np.array([1.63,.85])  #nominal generator loads 
    P_g_max = 1.5*P_g_nom
    P_g_min = .5*P_g_nom
    
    
    
    
    #generator frequency bounds

    Om_min = - omega_bound
    Om_max = omega_bound

    
    
    
    
    #Fixed initial conditions for the slack bus
    OMsi = [0]
    
    
    
    
    
    
    ##############################################################################
    ###########################################################################
    #Power System Model Definition
    ###################
    
    
    
    
    Y = np.zeros([9,9])
    
    
    #matrix giving value of admittance and reactance for lines between busses
    Y[0,3] = 1/0.0576  
    Y[3,4] = 1/sqrt(0.000017**2+0.092**2) 
    Y[4,5] = 1/sqrt(0.000039**2+0.17**2)
    Y[2,5] = 1/0.0586   
    Y[5,6] = 1/sqrt(0.0000119**2+0.1008**2)
    Y[6,7] = 1/sqrt(0.0000085**2+0.072**2)
    Y[7,1] = 1/0.0625   
    Y[7,8] = 1/sqrt(0.000032**2+0.161**2)
    Y[8,3] = 1/sqrt(0.00001**2+0.085**2)
    
    
    
    
    Y = Y + np.transpose(Y)
    
    V = np.ones([9,1])
    V[0] = 1.04; # this is voltage for the swing bus,
    V[1] = 1.02533 # voltage for the gerator bus 2
    V[2] = 1.025363 # voltage for the generator bus 3
    
    B=V@np.transpose(V)*Y
    
    B_dim = 9
    
    # create a dictionary mapping matrix coords to the values
    B_idx = list(itertools.product(np.arange(0,B_dim),np.arange(0,B_dim)))
    B_vals = list(np.reshape(B,[1,B_dim**2])[0])
    B_dict = dict(list(zip(B_idx,B_vals)))
    
    
    
    
    # Moments of inertia
    M = list(np.array([13.64,6.4,3.01]))  
    M_dim = 3
    M_idx = list(np.arange(0,M_dim))
    M_dict = dict(list(zip(M_idx,M)))
    
    
    
    # Damping constants
    D = list(np.array( [9.6,2.5,1.0])) 
    D_dim = 3
    D_idx = list(np.arange(0,D_dim))
    D_dict = dict(list(zip(D_idx,D)))
    
    
    
    #Node indices
    g_idx = [1,2]  #the generator indicies
    l_idx = [3,4,5,6,7,8] # the load bus indicies
    s_idx = [0]  #the swing bus index 
    
    
    theta_idx = [0,1,2,3,4,5,6,7,8] # all node indicies
    
    
    
    
    
    
    
    
    
    ############################################################################
    ###########################################################################
    # Solve for an equilbirum initial condition for all angles given loads
    # and generator inputs
    ###########################3
    
    from scipy.optimize import fsolve
    
    
    def equations(x):
        eq0 = x[0]
        eq1 = Pg_int[0] - sum(B[1,j]*sin(x[1]-x[j]) for j in theta_idx ) 
        eq2 = Pg_int[1] - sum(B[2,j]*sin(x[2]-x[j]) for j in theta_idx ) 
        eq3 = PLi[0] - sum(B[3,j]*sin(x[3]-x[j]) for j in theta_idx ) 
        eq4 = PLi[1] - sum(B[4,j]*sin(x[4]-x[j]) for j in theta_idx ) 
        eq5 = PLi[2] - sum(B[5,j]*sin(x[5]-x[j]) for j in theta_idx ) 
        eq6 = PLi[3] - sum(B[6,j]*sin(x[6]-x[j]) for j in theta_idx ) 
        eq7 = PLi[4] - sum(B[7,j]*sin(x[7]-x[j]) for j in theta_idx ) 
        eq8 = PLi[5] - sum(B[8,j]*sin(x[8]-x[j]) for j in theta_idx ) 
        return [eq0,eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8]
    
    thet_ini= fsolve(equations, [0,0,0,0,0,0,0,0,0] )
    
    thet_ini[0] = 0
    
    
    
    
    
    
    ##############################################################################
    ##############################################################################
    #Pyomo Model Definition
    ################
    
    
    m =  ConcreteModel()
    
    m.tf = Param(initialize = Tf)
    
    m.t = ContinuousSet(bounds=(0, m.tf))
    
    
    
    
    
    ########################################################################3
    ###########################################################################
    #Set parameters
    ###############
    
    
    
    
    # Model parameters
    ###########################3
    m.B=Param(B_idx ,initialize = B_dict)
    m.M=Param(M_idx,initialize = M_dict)
    m.D=Param(D_idx,initialize = D_dict)
    
    
    
    
    #constraint parameters
    ###########################
    
    #generator constraints
    Pg_max_init_dict = dict(list(zip(g_idx,P_g_max)))
    Pg_min_init_dict = dict(list(zip(g_idx,P_g_min)))
    m.Pg_max = Param(g_idx,initialize = Pg_max_init_dict)
    m.Pg_min = Param(g_idx,initialize = Pg_min_init_dict)
    
    
    
    
    
    #cost coefficient paramters
    ##################################
    cc_g_dict = dict(list(zip(g_idx,cost_coef_g)))
    cc_s_dict = dict(list(zip(s_idx,cost_coef_s)))
    m.cc_g = Param(g_idx,initialize = cc_g_dict)
    m.cc_s = Param(s_idx,initialize = cc_s_dict)
    
    
    
    
    
    
    
    #initial conditions
    ###########################
    
    #initial load value
    Load_init = PLi #[0,-0.9,0,-1.0,0,-1.25] 
    Load_dict = dict(list(zip(l_idx,Load_init)))
    m.Load_init = Param(l_idx,initialize = Load_dict)
    
    
    #initial generator value
    P_g_init = Pg_int
    P_g_init_dict = dict(list(zip(g_idx,P_g_init)))
    m.P_g_init = Param(g_idx, initialize = P_g_init_dict)
    
    
    #initial voltage angle values
    theta_init= thet_ini
    thet_init_dict = dict(list(zip(theta_idx,theta_init)))
    m.thet_init = Param(theta_idx, initialize = thet_init_dict)
    
    
    #initial generator frequency values
    omega_init = OMi
    om_init_dict = dict(list(zip(g_idx,omega_init)))
    m.om_init = Param(g_idx, initialize = om_init_dict)
    
    
    #initial swing frequency value
    omega_s_init = OMsi 
    om_s_init_dict = dict(list(zip(s_idx,omega_s_init)))
    m.om_s_init =  Param(s_idx, initialize = om_s_init_dict)
    
    
    
    
    
    
    ### Linear load function parameters
    ########################################################
    
    t_strt_dict = dict(list(zip(l_idx,t_strt_vec)))
    slope_dict = dict(list(zip(l_idx,slope_vec)))
    duration_dict = dict(list(zip(l_idx,duration_vec)))
    
    m.t_strt = Param(l_idx,initialize = t_strt_dict)
    m.slope = Param(l_idx,initialize = slope_dict)
    m.duration = Param(l_idx,initialize = duration_dict)
    
    
    
    
    
    
    
    
    #node indicies
    ###################
    m.g_idx = Set(initialize=g_idx )   #generator indicies
    m.l_idx = Set(initialize=l_idx)   #load bus indicies
    m.s_idx = Set(initialize=s_idx)   #swing bus index
    m.theta_idx = Set(initialize=theta_idx) #theta indicies
    
    
    
    
    
    
    
    
    
    
    
    
    
    ##########################################################################3
    ###########################################################################
    # Set model variables
    ##################
    
    
    m.theta = Var(m.theta_idx,m.t)  # node angles
    m.omega = Var(m.g_idx,m.t)  #first time derivative of generator node angles
    m.omega_s = Var(m.s_idx,m.t) #first tiem derivative of swing bus
    m.dtheta_dt = DerivativeVar(m.theta, wrt=(m.t)) 
    m.domega_dt = DerivativeVar(m.omega, wrt=(m.t))
    m.domega_dt_s = DerivativeVar(m.omega_s,wrt=(m.t))
    
    
    
    #set a variable swing/slack bus
    m.P_s = Var(m.s_idx,m.t) 
    
    
    
    
    
    
    
    
    
    
    
    
    #############################################################################
    ###########################################################################
    #Discretize the model
    ################
    
    
    
    discretizer = TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m,nfe = ndt, wrt=m.t, scheme='BACKWARD')
    
    
    
    
    
    
    
    
    
    
    
    ###########################################################################
    ###########################################################################
    #Set an initial iterate estimate for the generator inputs
    #################
    
    def gen_cntrl_init(m,i,t):
        return m.P_g_init[i]
    
    #the Control variable
    m.P_g = Var(m.g_idx,m.t,initialize = gen_cntrl_init)  # mechanical input to generators
    
    
    
    
    
    
    
    
    
    
    ###########################################################################
    ###########################################################################
    # Set the variable Loads
    ##################
    
    
    
    
    
    #linear ramping function
    def loading_fn(m,i,t):
        if t<=m.t_strt[i] : return m.Load_init[i]
        if t>m.t_strt[i] and t <=m.t_strt[i]+ m.duration[i]:
            return m.Load_init[i] - m.slope[i]*(t - m.t_strt[i])
        else: return m.Load_init[i] - m.duration[i]*m.slope[i]
    
    
    
    
    
    
    m.Var_Load = Param(l_idx,m.t,initialize = loading_fn)
    
    
    
    
    
    
    
    
    
    #############################################################################
    #############################################################################
    #Set the Dynamic Constraints
    #############
    
    
    
    #generator_dynamics
    def gen_dyn(m,i,t):
        return m.domega_dt[i,t]*m.M[i] + m.omega[i,t]*m.D[i] - m.P_g[i,t] + sum(m.B[(i,j)]*sin(m.theta[i,t]-m.theta[j,t]) for j in m.theta_idx ) == 0
    m.g_dyn = Constraint(m.g_idx,m.t,rule = gen_dyn)
    
    
    
    ### Slack constraint
    def slack_cns(m,i,t):
        return m.P_s[i,t] == sum(m.B[(i,j)]*sin(m.theta[i,t]-m.theta[j,t]) for j in m.theta_idx ) 
    m.slck_cns = Constraint(m.s_idx,m.t,rule = slack_cns)
    
    
    
    def fxswng(m,i,t):
        return m.domega_dt_s[i,t] == 0
    m.s_dyn = Constraint(m.s_idx,m.t,rule = fxswng)
    
    
    
    
    #generator derivative consistency eqn
    def gen_cns(m,i,t):
        return m.omega[i,t] - m.dtheta_dt[i,t] == 0
    m.g_cns = Constraint(m.g_idx,m.t,rule = gen_cns)
    
    
    
    
    #swing derivative consistency eqn
    def swng_cns(m,i,t):
        return m.omega_s[i,t] - m.dtheta_dt[i,t] == 0
    m.s_cns = Constraint(m.s_idx,m.t,rule = swng_cns)
    
    
    
    
    
    #Power flow equation with variable loads
    def pwr_flw_eqn_var(m,i,t):
        return m.Var_Load[i,t] - sum(m.B[(i,j)]*sin(m.theta[i,t]-m.theta[j,t]) for j in m.theta_idx ) == 0
    m.pwr_flw_var = Constraint(l_idx,m.t,rule = pwr_flw_eqn_var)
    
    
    
    
    
    
    
    
    #############################################################################3
    #############################################################################3
    #Set the variable Constraints
    ##################3
    
    
    # generator input constraints
    def P_upper_cnstrnts(m,i,t):
       return  m.P_g[i,t] <= m.Pg_max[i]
    m.P_ub = Constraint(m.g_idx,m.t,rule = P_upper_cnstrnts)
    
    def P_lower_cnstrnts(m,i,t):
       return  m.P_g[i,t] >= m.Pg_min[i]
    m.P_lb = Constraint(m.g_idx,m.t,rule = P_lower_cnstrnts)
    
    
    
    # generator angle derivative constraints
    def Om_upper_cnstrnts(m,i,t):
        return  m.omega[i,t] <= Om_max
    m.om_ub = Constraint(m.g_idx,m.t,rule = Om_upper_cnstrnts)
    
    
    def Om_lower_cnstrnts(m,i,t):
        return  m.omega[i,t] >= Om_min
    m.om_lb = Constraint(m.g_idx,m.t,rule = Om_lower_cnstrnts)
    
    
    
    
    
    
    
    
    #############################################################################
    #############################################################################
    #set ramping constraints on the generator inputs
    #######################3
    
    
    n_tm = len(m.t)
    
    time_idx = np.arange(1,n_tm+1)
    
    def ramp_cnstrnts_upper(m,i,j):
        if j>=2: 
            return m.P_g[i,m.t[j-1]] - m.P_g[i,m.t[j]] <= rmp_tol  
        else:
            return m.P_g[i,m.t[j]] - m.P_g[i,m.t[j+1]] <= rmp_tol
    
    def ramp_cnstrnts_lower(m,i,j):
        if j>=2: 
            return  - m.P_g[i,m.t[j-1]] + m.P_g[i,m.t[j]] <= rmp_tol  
        else:
            return - m.P_g[i,m.t[j]] + m.P_g[i,m.t[j+1]]  <= rmp_tol
       
    
        
    m.rmp_cns_up = Constraint(m.g_idx,time_idx,rule = ramp_cnstrnts_upper)
    m.rmp_cns_lw = Constraint(m.g_idx,time_idx,rule = ramp_cnstrnts_lower)
    
    
    
    
    
    
    
    
    
    
    
    ###########################################################################
    ###########################################################################
    #### Slack constraints to use to replace the absolute value in the objective
    ##############################
    
    m.a = Var(m.t)  # obj_slack_var
    
    
    def min_a_cnstrnt(m,t):
        return m.a[t] >= 0
    
    def max_a_cnstrnt(m,t) :
        return m.a[t] + 2* m.P_s[0,t] >= 0
     
    
    m.mna_cns = Constraint(m.t,rule=min_a_cnstrnt)
    m.mxa_cns = Constraint(m.t,rule=max_a_cnstrnt)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #############################################################################
    #############################################################################
    #Set the initial condition constraints
    ##################
    
    
    def _omega_init(m,i):
        return m.omega[i,m.t.first()] == m.om_init[i]
    m.omega_init = Constraint(m.g_idx,rule = _omega_init)
    
    
    
    def _omega_s_init(m,i):
        return m.omega_s[i,m.t.first()] == m.om_s_init[i]
    m.omega_s_init = Constraint(m.s_idx,rule = _omega_s_init)
    
    
    
    def _theta_init(m,i):
        return m.theta[i,m.t.first()] == m.thet_init[i]
    m.theta_init = Constraint(m.theta_idx,rule = _theta_init)
    
    
    
    def _Pg_init(m,i):
        return m.P_g[i,m.t.first()] == m.P_g_init[i]
    m.Pg_init = Constraint(m.g_idx,rule = _Pg_init)
    
    
    
    
    
    
    #################################################################################
    #################################################################################
    #Define the objective
    
    
    
    
    def _intObj(m): 
        return sum( sum(  m.cc_g[i] * m.P_g[i,t] for i in m.g_idx) for t in m.t) + sum(   sum( m.cc_s[i]*( m.P_s[i,t] + m.a[t] )  for i in m.s_idx )  for t in m.t) 
         
    
    
    
    
    
    
    
    
    
    ##############################################################################
    ##############################################################################
    #Solve the model
    
    max_iter = 1000
    max_cpu_time = 1000
    
    
    m.EDcost = Objective(rule = _intObj, sense= minimize)
    
    
    
    
    
    
    
    
    tempfile = pyutilib.services.TempfileManager.create_tempfile(suffix='ipopt_out', text=True)
    opts = {'output_file': tempfile,
            'max_iter': max_iter,
            'max_cpu_time': max_cpu_time}
    
    
    
    #find the solution
    opt=SolverFactory('ipopt', solver_io='nl')
    
    
    status_obj = opt.solve(m, options=opts )
    
    
    
    
    
    

    
    
    
    
    
    ##############################################################################
    ##############################################################################
    #Extract the outputs
    #########
    
    
    
    ## Get the Solver outputs
    #############################################################################
    
    # find out if it terminated on optimal solution
    solved = 1
    if status_obj.solver.termination_condition != TerminationCondition.optimal:
        solved = 0
    
    
    # get the number of iterations and total solution time
    iters = 0   
    sol_time = 0
    line_m_2 = None
    line_m_1 = None
    # parse the output file to get the iteration count, solver times, etc.
    with open(tempfile, 'r') as f:
        for line in f:
            if line.startswith('Number of Iterations....:'):
                tokens = line.split()
                iters = int(tokens[3])
                tokens_m_2 = line_m_2.split()
                regu = str(tokens_m_2[6])
            elif line.startswith('Total CPU secs in IPOPT (w/o function evaluations)   ='):
                tokens = line.split()
                sol_time += float(tokens[9])
            elif line.startswith('Total CPU secs in NLP function evaluations           ='):
                tokens = line.split()
                sol_time += float(tokens[8])
            line_m_2 = line_m_1
            line_m_1 = line
    
    
    
    obj_value = value(m.EDcost)
    
    
    solver_outputs = [solved,iters,sol_time,obj_value]
    
    
    
    
    
    
    
    
    #Record Load and Generator bus solution trajectories
    #####################################
    n_tm = len(m.t)
    
        
    time_idx = np.arange(1,n_tm+1)
    
    
    Om_trj = np.zeros([n_tm,len(g_idx)])
    Pg_trj = np.zeros([n_tm,len(g_idx)])
    
    Oms_trj = np.zeros([n_tm,len(s_idx)])
    
    Th_trj = np.zeros([n_tm,len(theta_idx)])
    
    Ps_trj = np.zeros([n_tm,len(s_idx)])
    
    Ld_trj = np.zeros([n_tm, 3])
    
    fullLd_trj = np.zeros([n_tm,6])
    
    for j in time_idx:
        
        for i in g_idx:
            Om_trj[j-1,i-1] =value( m.omega[i,m.t[j]] )
            Pg_trj[j-1,i-1] =value( m.P_g[i,m.t[j]] )
        
        for i in s_idx:
            #Oms_trj[j-1,i] =value( m.omega_s[i,m.t[j]] )
            Oms_trj[j-1,i] = 0
            
        for i in theta_idx:
            Th_trj[j-1,i] =value( m.theta[i,m.t[j]] )
            
        Th_trj[j-1,0] = 0
            
            
        for i in s_idx:
            Ps_trj[j-1,i] = value(m.P_s[i,m.t[j]])
        
        for i in [0,1,2]:
            ddx = [4,6,8]
            Ld_trj[j-1,i] = value(m.Var_Load[ddx[i],m.t[j]] )
            
        for i in [0,1,2,3,4,5]:
            ddf = [3,4,5,6,7,8]
            fullLd_trj[j-1,i] = value(m.Var_Load[ddf[i],m.t[j]] )
    

    
    
    #combine for the full P trajectory
    P_trj = np.concatenate((Ps_trj,Pg_trj,fullLd_trj),axis = 1 )
    
    
    State_trj = np.concatenate((Oms_trj,Om_trj,Th_trj),axis = 1)
    
    
    Neuro_trj = np.concatenate((State_trj,P_trj),axis=1)
    
    
    
    
    return [solver_outputs,Neuro_trj]
    
    
