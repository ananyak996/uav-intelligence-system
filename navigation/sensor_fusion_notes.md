# UAV Navigation Module – Sensor Fusion Analysis

## Dataset Overview

The navigation dataset consists of time-series sensor readings recorded at approximately 1 Hz.  
The following columns were used in this analysis:

- **IMU Accelerometer**:  
  - `imu_acc_x`, `imu_acc_y`
- **GPS Position**:  
  - `latitude`, `longitude`
- **Time**:  
  - `timestamp`

Only planar (2D) motion was considered to clearly demonstrate drift and correction without introducing vertical dynamics.


## IMU-Only Dead Reckoning

### Method

IMU acceleration data was numerically integrated in two stages:

1. Acceleration → Velocity  
2. Velocity → Position  

Time differences between samples (`dt`) were computed from timestamps to ensure physically correct integration.

### Observation

The IMU-only trajectory diverged rapidly from the GPS path.  
Small acceleration noise and bias accumulated over time, causing the estimated position to grow unbounded.

This demonstrates **dead-reckoning drift**, making IMU-only navigation unsuitable for long-term position estimation.


## GPS-Only Navigation Characteristics

GPS position estimates remained bounded and did not drift over time.  
However, the trajectory exhibited short-term noise and lacked smoothness, making it unsuitable for responsive navigation or control.

This highlights the complementary limitations of GPS and IMU sensors.



## Sensor Fusion Using a Complementary Filter

### Approach

A complementary filter was implemented to fuse IMU and GPS data incrementally.

- IMU data was used to estimate short-term motion increments
- GPS data was used to provide long-term positional correction

Rather than fusing absolute IMU position (which already contains drift), IMU-derived position changes were applied to the previously fused estimate and continuously corrected using GPS measurements.

A weighting factor of `alpha = 0.98` was used to prioritize IMU responsiveness while maintaining GPS stability.



## Results and Analysis

The fused trajectory demonstrated:

- Elimination of long-term drift
- Significantly smoother motion compared to GPS-only navigation
- Stability within the GPS reference frame

The fused estimate successfully retained the short-term smoothness of IMU integration while preventing unbounded error through continuous GPS correction.

