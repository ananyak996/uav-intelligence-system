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




# SENSOR FUSION WITH A COMPLEMENTARY FILTER


alpha = 0.98 

x_fused = np.zeros(len(x_imu))
y_fused = np.zeros(len(y_imu))

x_fused[0] = x_gps[0]
y_fused[0] = y_gps[0]

for i in range(1, len(x_imu)):
    #IMU-based prediction (incremental)
    dx_imu = x_imu[i] - x_imu[i-1]
    dy_imu = y_imu[i] - y_imu[i-1]

    x_pred = x_fused[i-1] + dx_imu
    y_pred = y_fused[i-1] + dy_imu

    # GPS correction
    x_fused[i] = alpha * x_pred + (1 - alpha) * x_gps[i]
    y_fused[i] = alpha * y_pred + (1 - alpha) * y_gps[i]


#Plotting the fused results
plt.figure(figsize=(7, 6))

plt.plot(x_gps, y_gps, label='GPS Path', linewidth=2)
plt.plot(x_imu, y_imu, label='IMU Dead Reckoning', linestyle='--')
plt.plot(x_fused, y_fused, label='Fused Trajectory', linewidth=2)

plt.xlabel('X Position (relative)')
plt.ylabel('Y Position (relative)')
plt.title('IMU vs GPS vs Fused Trajectory')
plt.legend()
plt.grid(True)
plt.show()


print("GPS range:", x_gps.min(), x_gps.max())
print("IMU range:", x_imu.min(), x_imu.max())
print("Fused range:", x_fused.min(), x_fused.max())





