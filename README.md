# UAV Intelligence System

An applied UAV intelligence project focused on behavioral security analysis
and navigation sensor fusion under noisy, real-world conditions.

The system demonstrates how classical machine learning and lightweight
sensor fusion techniques can be combined to improve robustness and
decision-making in autonomous aerial platforms.

--------

## Overview

This repository contains a modular UAV intelligence system with two core
components:

- **Security Module**  
  Machine learning–based classification of UAV behavior to identify
  potentially malicious or anomalous activity from telemetry data.

- **Navigation Module**  
  Sensor fusion of IMU and GPS data to mitigate drift and noise in
  position estimation.

The project emphasizes clarity, interpretability, and system-level
reasoning over black-box solutions.

--------

## Security Module

- Performed exploratory analysis of UAV behavioral data
- Trained and compared Logistic Regression, KNN, and SVM classifiers
- Evaluated models using accuracy and confusion matrices
- Analyzed false-positive vs false-negative trade-offs from a
  UAV security perspective

--------

## Navigation Module

- Demonstrated IMU-only dead reckoning drift through numerical integration
- Analyzed GPS stability and noise characteristics
- Implemented a complementary filter to fuse inertial and GPS data
- Compared raw and fused trajectories to validate noise reduction

--------

## Repository Structure

uav-intelligence-system/
├── security/ # UAV behavioral classification
├── navigation/ # IMU–GPS sensor fusion
├── data/sample/ # Small sample datasets
├── README.md
└── .gitignore

--------

## Technologies

Python, NumPy, Pandas, Scikit-learn, Matplotlib

--------

## Notes

- Full datasets are excluded due to size and licensing constraints
- The codebase is structured to support future extensions such as
  real-time telemetry visualization and vision-based perception
