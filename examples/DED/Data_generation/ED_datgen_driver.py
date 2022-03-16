




'''
##########################################################################

This scripts generates solutions to the Dynamics Aware Economic Dispatch 
subject to the swing equations on a 9-bus test case using ipopt via pyomo


##########################################################################
'''




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

from scipy.io import savemat

import pyutilib.services
from pyomo.opt import TerminationCondition





## load in pyomo optimization module
from ED_cntsdyn_opt_mod import ED_cntsdyn_opt



'''
#Set Problem parameters
#################################

'''

n_dat = 2  #number of problem intances to generate and solve



omega_bound = .314










#Specify where to save the data
dat_root = '../Data/Generated_data/'
save_folder = 'trial_data'




#set up storage
###################################################

cnts_slv_out = np.zeros((n_dat,7))
cnts_xini_out = np.zeros((n_dat,12))
cnts_ptrj_out = np.zeros((6001,9,n_dat))


KO_slv_out = np.zeros((n_dat,4))
KO_xini_out = np.zeros((n_dat,12))
KO_ptrj_out = np.zeros((6001,9,n_dat))




##############################################################################
############################################################################3
#Generate data
###############
loop_counter = 0

for i in range(n_dat):
    
        
        
        
        
    
    ###########################################################################
    ###########################################################################
    ## generate ED problem parameters
    ######################
    
    
    # 9 bus transmission system parms
    ng = 2 #number of generators
    ns = 1 #swing bus
    nL =6 #number of load nodes
    
    
    
    #Set the cost functional coefficients for each generator and swing bus
    c_1 = nprd.uniform(0,1,1)
    c_2 = nprd.uniform(0,1,1)
    c_s = nprd.uniform(0,1,1) 
    
    cost_coef_g = [c_1[0],c_2[0]]
    cost_coef_s = [c_s[0]]
    
    
    
    
    # set the generator ramping tolerance
    rmp_tol = 5E-4
    
    
    
    
    ED_parms=[cost_coef_g,cost_coef_s,rmp_tol]
    
    
    ###########################################################################
    ###########################################################################
    # generate the Load Values
    ########################
    
    
    prob_Ld_increase = .5 
    pb_ldi_r = nprd.uniform(0,1,1)
    
    
    #Set the initial load values
    Load_nom = [0,-0.9,0,-1.0,0,-1.25] #nominal load values
    Ld_rng = .25   #.25 #max proportion difference in load values allowed
    
    
    
    
    if pb_ldi_r[0] <= prob_Ld_increase:
    
        PLi = (1 + Ld_rng*nprd.uniform(-1,1,nL) )*Load_nom
        Ld_sign = 1
    else:
        PLi = ( (1+Ld_rng) + Ld_rng*nprd.uniform(-1,1,nL) )*Load_nom
        Ld_sign = -1
    
    
    
    
    ### Linear ramping Load function parameters
    ########################################################
    
    
    nrLd = 3 #number of nodes with a real load
    vldx = [1,3,5]  #indeces for nodes with load values
    
    
    prob_load_change = .85
    p_Ld_change = nprd.uniform(0,1,nrLd)
    
    t_strt_vec = 100*np.ones(nL)
    for i in np.arange(0,3):
        if p_Ld_change[i]<=prob_load_change:
            rnd_time = nprd.uniform(0,60,1)
            t_strt_vec[vldx[i]]= rnd_time[0]
    
    
    
    slope_vec = Ld_sign*nprd.uniform(.01,.05,nL)
    
    duration_vec = nprd.uniform(5,20,nL)
    
       
    
    
    
    
    Ld_parms = [PLi,t_strt_vec,slope_vec,duration_vec]
    
    
    
    
    ###########################################################################
    ###########################################################################
    #generate the initial values for the generator inputs
    #############################
    
    #generator power output bounds
    P_g_nom = np.array([1.63,.85]) 
    P_g_max = 1.5*P_g_nom
    P_g_min = .5*P_g_nom
    
    
    #slack bus bounds
    S_max = [100] ;
    S_min = [0] ;
    
    
    
    
    #solve for optimal initial set point
    
    #problem setup
    c=[c_s,c_1,c_2]
    u_b = np.concatenate((S_max,P_g_max))
    l_b = np.concatenate((S_min,P_g_min))
    bnds = list(zip(l_b,u_b))
    
    Aeq = np.reshape(np.array([1,1,1]),(1,3))
    
    beq = np.abs(sum(PLi))
    
    #solve linear programming problem
    lin_sol = scio.linprog(c,A_ub=None,b_ub=None,A_eq=Aeq,b_eq=beq, bounds=bnds )
    
    opt_P=lin_sol['x']
    
    
    
    #Set the initial generator inputs as the optimal values
    
    Pg_int = opt_P[1:3]
    
    Ps0=np.reshape(opt_P[0],[1,1])
    
    #peturb the initial generator values
    Pg_int = Pg_int - .5*(Pg_int - P_g_nom)*nprd.uniform(0,1)
    
    #reset Ps0
    #Ps0 = np.reshape(beq - sum(Pg_int), [1,1] )
    
    #peturb the initial generator phase angle frequencies
    OMi = [0,0] 
    OMi = OMi + .01*nprd.uniform(-1,1,2)
    
    
    #########################################################################
    ##########################################################################
    ## solve the optimization
    #######################
    
    
    
    
    #continuous dynamics
    [solver_outputs,N_trj] = ED_cntsdyn_opt(ED_parms,Ld_parms,Pg_int,OMi,omega_bound)
    
    
    
    
    solver_outputs=np.concatenate((solver_outputs,cost_coef_s,cost_coef_g))
    
    
    #Save the outputs
    
    
    #Neuromancer column names
    cols = ["y1","y2","y3","y4","y5","y6","y7","y8","y9","y10","y11","y12","y13","u1","u2","d1","d2","d3","d4","d5","d6"]    
    
    N_df = pd.DataFrame(N_trj,columns = cols)
    
    
    cnts_slv_out[loop_counter,:] = solver_outputs
    
    
    
    
    loop_counter = loop_counter + 1
    
    path = dat_root + save_folder + "/trial_{}.csv".format(loop_counter)
    

    N_df.to_csv(path,index=False)
    
    print(loop_counter)
    

    metdat_cols = ["solved","iters","sol_time","obj_value","cs","c1","c2"]
    metdat_df = pd.DataFrame(cnts_slv_out,columns = metdat_cols)
    met_path = dat_root + "opt_metadata.csv"
    metdat_df.to_csv(met_path,index=False)



