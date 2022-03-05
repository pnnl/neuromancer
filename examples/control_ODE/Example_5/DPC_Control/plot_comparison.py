import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def norm_max1(a):
    # Normalised [0,1]
    return a/np.max(a)

labels = ['DPC', 'MPC', 'RMPC', 'SMPC', 'LQR', 'LQI', 'PPO2', 'A2C', 'ACKTR']
# normalized metric nominal simulation
MSE_ref_nominal = norm_max1(np.asarray([1.244, 1.398, 1.405, 1.398, 2.024, 1.954, 16.978, 10.682, 9.556]))
MA_ene_nominal = norm_max1(np.asarray([1111, 897, 899, 897, 758, 899, 531, 732, 1496]))
MA_con_nominal = norm_max1(np.asarray([0.000, 0.000, 0.000, 0.000, 5.574, 1.893, 2.525, 1.608,  0.557]))
# normalized metric simulation with parametric and additive uncertainties
MSE_ref_uncertain = norm_max1(np.asarray([2.355, 3.711, 2.839, 3.579, 2.711, 4.957, 27.885, 14.682, 15.473]))
MA_ene_uncertain = norm_max1(np.asarray([1139, 866, 836, 856, 883, 676, 526, 731, 1510]))
MA_con_uncertain = norm_max1(np.asarray([0.000, 1.106, 0.066, 0.572, 7.567, 5.346, 3.546, 1.883, 0.768]))

x = np.arange(len(labels))  # the label locations
width = 0.30  # the width of the bars

# NOMINAL SIMULATION
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, MSE_ref_nominal, width, label='Reference')
rects2 = ax.bar(x, MA_ene_nominal, width, label='Energy')
rects3 = ax.bar(x + width, MA_con_nominal, width, label='Constraints')
ax.set_ylabel('Normalized Loss', fontsize=16)
ax.set_title('Performance Comparison Nominal Simulations', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=16)
ax.grid()
ax.legend(fontsize=16)

# UNCERTAIN SIMULATION
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, MSE_ref_uncertain, width, label='Reference')
rects2 = ax.bar(x, MA_ene_uncertain, width, label='Energy')
rects3 = ax.bar(x + width, MA_con_uncertain, width, label='Constraints')
ax.set_ylabel('Normalized Loss', fontsize=16)
ax.set_title('Performance Comparison Uncertain Simulations', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=16)
ax.grid()
ax.legend(fontsize=16)



# REFERENCE TRACKING
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, MSE_ref_nominal, width, label='Nominal')
rects2 = ax.bar(x + width/2, MSE_ref_uncertain, width, label='Uncertain')
ax.set_ylabel('Normalized Loss', fontsize=16)
ax.set_title('Performance Comparison Reference Tracking', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=16)
ax.grid()
ax.legend(fontsize=16)

# ENERGY USE
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, MA_ene_nominal, width, label='Nominal')
rects2 = ax.bar(x + width/2, MA_ene_uncertain, width, label='Uncertain')
ax.set_ylabel('Normalized Loss', fontsize=16)
ax.set_title('Performance Comparison Energy Use', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=16)
ax.grid()
ax.legend(fontsize=16)

# CONSTRAINTS VIOLATIONS
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, MA_con_nominal, width, label='Nominal')
rects2 = ax.bar(x + width/2, MA_con_uncertain, width, label='Uncertain')
ax.set_ylabel('Normalized Loss', fontsize=16)
ax.set_title('Performance Comparison Constraints Violations', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=16)
ax.grid()
ax.legend(fontsize=16)