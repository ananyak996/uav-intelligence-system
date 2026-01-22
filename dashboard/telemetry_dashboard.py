import streamlit as st
import pandas as pd
import time

# Page config
st.set_page_config(page_title="UAV Ground Station", layout="wide")

st.title("UAV Ground Station – Live Telemetry Dashboard")

# Load dataset
df = pd.read_csv(
    "uav_navigation_dataset.csv"
)


# Placeholders for live plots
altitude_chart = st.empty()
attitude_chart = st.empty()
env_chart = st.empty()

# Data buffers
altitude_data = []
roll_data = []
pitch_data = []
wind_data = []
speed_data = []

# Simulate live streaming
for i in range(len(df)):
    row = df.iloc[i]

    altitude_data.append(row["altitude"])
    roll_data.append(row["imu_gyro_x"])
    pitch_data.append(row["imu_gyro_y"])
    wind_data.append(row["wind_speed"])
    speed_data.append(row["speed"])

    # Altitude plot
    altitude_chart.line_chart(
        pd.DataFrame({"Altitude": altitude_data})
    )

    # Attitude indicator (Roll & Pitch)
    attitude_chart.line_chart(
        pd.DataFrame({
            "Roll (proxy)": roll_data,
            "Pitch (proxy)": pitch_data
        })
    )

    # Environmental monitor
    env_chart.line_chart(
        pd.DataFrame({
            "Wind Speed": wind_data,
            "Ground Speed": speed_data
        })
    )

    time.sleep(0.5)  # simulate telemetry delay
