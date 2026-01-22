# UAV Intelligence System

An applied UAV intelligence project integrating machine learning–based security analysis,
navigation sensor fusion, and computer vision–based situational awareness under noisy,
real-world conditions.

The system demonstrates how multiple intelligence layers—behavioral analysis, state
estimation, and visual perception—can work together to support decision-making in
autonomous aerial platforms.

---

## System Overview

This repository implements a modular UAV intelligence stack composed of three
cooperating subsystems:

- **Security Module**  
  Detects potentially malicious or anomalous UAV behavior using supervised
  machine learning models trained on behavioral telemetry.

- **Navigation Module**  
  Mitigates sensor noise and drift by fusing IMU and GPS data to obtain stable
  position estimates suitable for navigation and monitoring.

- **Perception Module**  
  Provides situational awareness using computer vision to detect, track, and
  reason about dynamic objects (pedestrians and vehicles) in urban scenes.

A lightweight ground-station dashboard visualizes navigation outputs in real time,
simulating live UAV telemetry during flight operations.

The project emphasizes clarity, interpretability, and system-level reasoning over
black-box solutions.

---

## Security Module

- Exploratory analysis of UAV behavioral telemetry
- Classification using Logistic Regression, KNN, and SVM
- Evaluation via accuracy and confusion matrices
- Analysis of false-positive vs false-negative trade-offs from a UAV security
  perspective

---

## Navigation Module

- Demonstration of IMU-only dead reckoning drift
- Analysis of GPS noise and stability
- Sensor fusion using a complementary filter
- Comparison of raw and fused trajectories to validate noise reduction

---

## Ground Station Dashboard

- Simulates live UAV telemetry by streaming navigation data row by row
- Dynamically updates plots to visualize navigation behavior in real time
- Demonstrates how fused navigation outputs can be monitored from a ground station

---

## Perception Module (Computer Vision & Situational Awareness)

- YOLOv8-based object detection for pedestrians and vehicles
- Centroid-based tracking to maintain persistent object identities across frames
- Temporal motion analysis to classify objects as MOVING or STATIONARY
- Generation of an annotated output video for perception validation

---

## Repository Structure

uav-intelligence-system/
├── security/ # UAV behavioral classification
├── navigation/ # IMU–GPS sensor fusion
├── dashboard/ # Ground station telemetry visualization
├── perception/ # CV-based detection, tracking, and situational awareness
├── data/sample/ # Small sample datasets
├── README.md
└── .gitignore

---

## Technologies

Python, NumPy, Pandas, Scikit-learn, Matplotlib, Streamlit, OpenCV, YOLOv8

---

## Notes

- Full datasets, model weights, and video files are excluded due to size and
  licensing constraints.
- The codebase is structured to support future extensions such as real-time
  onboard integration and multi-sensor fusion.
