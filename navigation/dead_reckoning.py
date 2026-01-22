import pandas as pd
df_nav = pd.read_csv("uav_navigation_dataset.csv")



# IMU-ONLY DEAD RECKONING VS GPS


import numpy as np

#Convert timestamp to datetime
df_nav['timestamp'] = pd.to_datetime(df_nav['timestamp'])

#Sorting the values by timestamp
df_nav = df_nav.sort_values('timestamp')

#Compute time difference (seconds)
df_nav['dt'] = df_nav['timestamp'].diff().dt.total_seconds()
df_nav['dt'] = df_nav['dt'].fillna(1.0)  # first timestep

ax = df_nav['imu_acc_x'].values
ay = df_nav['imu_acc_y'].values
dt = df_nav['dt'].values

#Integrating acceleration to get velocity and position
vx = np.zeros(len(ax))
vy = np.zeros(len(ay))

for i in range(1, len(ax)):
    vx[i] = vx[i-1] + ax[i] * dt[i]
    vy[i] = vy[i-1] + ay[i] * dt[i]

x_imu = np.zeros(len(vx))
y_imu = np.zeros(len(vy))

for i in range(1, len(vx)):
    x_imu[i] = x_imu[i-1] + vx[i] * dt[i]
    y_imu[i] = y_imu[i-1] + vy[i] * dt[i]

lat = df_nav['latitude'].values
lon = df_nav['longitude'].values

#Convert to relative position
x_gps = lat - lat[0]
y_gps = lon - lon[0]

#Plotting the results
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))

plt.plot(x_gps, y_gps, label='GPS Path', linewidth=2)
plt.plot(x_imu, y_imu, label='IMU Dead Reckoning', linestyle='--')

plt.xlabel('X Position (relative)')
plt.ylabel('Y Position (relative)')
plt.title('IMU-only Dead Reckoning vs GPS')
plt.legend()
plt.grid(True)
plt.show()