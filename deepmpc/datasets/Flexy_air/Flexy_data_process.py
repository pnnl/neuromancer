

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

# lowpass filter data
b, a = signal.butter(3, 0.05)
y_filter = signal.filtfilt(b, a, y)
u_filter = signal.filtfilt(b, a, u)

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(y, label='Y')
ax[0].plot(y_filter, label='Y filtered')
ax[0].plot(r, '--', label='R')
ax[0].set(ylabel='Y')
ax[1].plot(u, label='U')
ax[1].plot(u_filter, label='U filtered')
ax[1].set(ylabel='U')
plt.tight_layout()

# save filtered data for system ID
IO_data = pd.DataFrame(data=np.array([y_filter, u_filter]).T, columns=['y', 'u'])
IO_data.to_csv('flexy_air_data.csv', index=False)

D = IO_data.filter(regex='d').values if IO_data.filter(regex='d').values.size != 0 else None
