# floating ball system
# https://engineering.purdue.edu/~andrisan/Courses/AAE421_Fall_2007/AAE421_Buffer_F07/HW4-5-6_Floatball.pdf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

df_raw1 = pd.read_csv('meas2.csv')
df_raw2 = pd.read_csv('meas1.csv')
df_raw = df_raw1.append(df_raw2)

u = df_raw.iloc[:,1].to_numpy()
y = df_raw.iloc[:,3].to_numpy()
r = df_raw.iloc[:,2].to_numpy()

# r - reference, position
# y - position
# dy - velocity
# ddy - acceleration
# u - fan power
# du - fan power speed
# du - fan power acceleration


# lowpass filter data
b, a = signal.butter(3, 0.05)
y_filter = signal.filtfilt(b, a, y)
u_filter = signal.filtfilt(b, a, u)
# 1st and 2nd order derivatives
dy_filter = np.diff(y_filter)
du_filter = np.diff(u_filter)
ddy_filter = np.diff(dy_filter)
ddu_filter = np.diff(du_filter)
# resample time series
new_samples = np.floor(y_filter.shape[0]/5).astype(int)
r_resampled = signal.resample(r, new_samples)
y_resampled = signal.resample(y_filter, new_samples)
u_resampled = signal.resample(u_filter, new_samples)
dy_resampled = signal.resample(dy_filter, new_samples)
du_resampled = signal.resample(du_filter, new_samples)
ddy_resampled = signal.resample(ddy_filter, new_samples)
ddu_resampled = signal.resample(ddu_filter, new_samples)
t_resampled = np.linspace(0, y_filter.shape[0], new_samples, endpoint=False)

fig, ax = plt.subplots(4, 1, figsize=(8, 8))
ax[0].plot(y, label='Y')
ax[0].plot(y_filter, label='Y filtered')
ax[0].plot(t_resampled, y_resampled, label='Y resampled')
ax[0].plot(t_resampled, r_resampled, '--', label='R')
ax[0].set(ylabel='Y')
ax[1].plot(u, label='U')
ax[1].plot(u_filter, label='U filtered')
ax[1].plot(t_resampled, u_resampled, label='U resampled')
ax[1].set(ylabel='U')
ax[2].plot(dy_filter, label='dY filtered')
ax[2].plot(ddy_filter, label='ddY filtered')
ax[2].plot(t_resampled, dy_resampled, label='dY resampled')
ax[2].plot(t_resampled, ddy_resampled, label='ddY resampled')
ax[2].set(ylabel='dY')
ax[3].plot(du_filter, label='dU filtered')
ax[3].plot(ddu_filter, label='ddU filtered')
ax[3].plot(t_resampled, du_resampled, label='dU resampled')
ax[3].plot(t_resampled, ddu_resampled, label='ddU resampled')
ax[3].set(ylabel='dU')
plt.tight_layout()

nsim = ddu_resampled.shape[0]
# save filtered data for system ID
# IO_data = pd.DataFrame(data=np.array([y_filter, u_filter]).T, columns=['y', 'u'])
IO_data = pd.DataFrame(data=np.array([y_resampled[0:nsim], dy_resampled[0:nsim], ddy_resampled[0:nsim],
                                      r_resampled[0:nsim], u_resampled[0:nsim]]).T,
                       columns=['y1', 'y2', 'y3', 'u1', 'd1'])
IO_data.to_csv('flexy_air_data.csv', index=False)

D = IO_data.filter(regex='d').values if IO_data.filter(regex='d').values.size != 0 else None
