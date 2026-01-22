# UAV Intelligence System

An applied UAV intelligence project focused on behavioral security analysis,
navigation sensor fusion, and ground-station telemetry visualization under
noisy, real-world conditions.

The system demonstrates how classical machine learning and lightweight
sensor fusion techniques can be combined to improve robustness and
decision-making in autonomous aerial platforms.

---

## Overview

This repository implements a modular UAV intelligence system with three
cooperating components:

- **Security Module**  
  Machine learning–based classification of UAV behavior to identify
  potentially malicious or anomalous activity from telemetry data.

- **Navigation Module**  
  Sensor fusion of IMU and GPS data to mitigate drift and noise in UAV
  position estimation.

- **Ground Station Dashboard**  
  A lightweight telemetry dashboard that simulates live UAV data streaming
  and visualizes navigation outputs in real time.

The project emphasizes clarity, interpretability, and system-level reasoning
over black-box solutions.

---

## Security Module

- Performed exploratory data analysis on UAV behavioral telemetry
- Trained and compared Logistic Regression, KNN, and SVM classifiers
- Evaluated models using accuracy and confusion matrices
- Analyzed false-positive vs false-negative trade-offs from a UAV
  security perspective

---

## Navigation Module

- Demonstrated IMU-only dead reckoning drift through numerical integration
- Analyzed GPS stability and short-term noise characteristics
- Implemented a complementary filter to fuse inertial and GPS data
- Compared raw and fused trajectories to validate noise reduction

---

## Ground Station Dashboard

- Simulates live UAV telemetry by streaming navigation data row by row
- Dynamically updates plots to visualize navigation behavior in real time
- Demonstrates how fused navigation outputs can be monitored during flight
  operations from a ground station

---

## Repository Structure

uav-intelligence-system/
├── security/ # UAV behavioral classification
├── navigation/ # IMU–GPS sensor fusion
├── dashboard/ # Ground station telemetry visualization
├── data/sample/ # Small sample datasets
├── README.md
└── .gitignore

---

## Technologies

Python, NumPy, Pandas, Scikit-learn, Matplotlib, Streamlit

---

## Notes

- Full datasets are excluded due to size and licensing constraints
- The codebase is structured to support future extensions such as
  vision-based perception and real-time onboard integration
